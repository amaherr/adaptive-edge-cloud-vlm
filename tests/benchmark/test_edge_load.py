"""Edge-load levels and the supervised load-generator process."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from adaptive_vlm.benchmark.edge_load import (
    IN_PROCESS,
    EdgeLoadGenerator,
    InProcessEdgeLoad,
    LoadGeneratorFailure,
    LoadLevel,
    level_conditions,
    load_levels,
    make_generator,
)

LEVEL = LoadLevel("L1", working_set_mib=512, busy_ms=8.0, idle_ms=12.0)

FAKE_WORKER = """
import argparse, json, os, signal, sys, time
parser = argparse.ArgumentParser()
for name in ("--working-set-mib", "--busy-ms", "--idle-ms", "--heartbeat",
             "--device", "--level"):
    parser.add_argument(name)
parser.add_argument("--mode", default="run")
args, _ = parser.parse_known_args()
mode = os.environ.get("FAKE_WORKER_MODE", "run")
if mode == "crash":
    sys.exit(3)
stop = False
def handler(*_):
    global stop
    stop = True
signal.signal(signal.SIGTERM, handler)
state = {"status": "running", "pid": os.getpid(), "level": args.level,
         "allocated_mib": float(args.working_set_mib), "kernels": 0, "duty": 0.4,
         "started_unix_s": time.time()}
beats = 0
while not stop:
    state.update(updated_unix_s=time.time(), kernels=state["kernels"] + 1)
    if mode == "run" or beats == 0:
        tmp = args.heartbeat + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(state, handle)
        os.replace(tmp, args.heartbeat)
    beats += 1
    time.sleep(0.05)
state.update(status="stopped", updated_unix_s=time.time())
with open(args.heartbeat, "w") as handle:
    json.dump(state, handle)
"""


def _generator(tmp_path: Path, **kwargs: object) -> EdgeLoadGenerator:
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    return EdgeLoadGenerator(
        LEVEL,
        directory=tmp_path / "load",
        command_prefix=[sys.executable, str(worker)],
        start_timeout_s=20,
        **kwargs,  # type: ignore[arg-type]
    )


def test_levels_load_and_validate(tmp_path: Path) -> None:
    path = tmp_path / "levels.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "levels": {
                    "L1": {"working_set_mib": 512, "busy_ms": 8, "idle_ms": 12},
                    "L2": {"working_set_mib": 1024, "busy_ms": 8, "idle_ms": 12},
                },
            }
        )
    )
    levels = load_levels(path)
    assert levels["L1"] == LEVEL
    assert levels["L2"].working_set_mib == 1024
    for bad in (
        {"schema_version": 2, "levels": {}},
        {"schema_version": 1, "levels": {"L0": LEVEL.to_dict()}},
        {
            "schema_version": 1,
            "levels": {"L1": {"working_set_mib": 0, "busy_ms": 1, "idle_ms": 1}},
        },
    ):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_levels(path)


def test_level_conditions_record_the_load() -> None:
    assert level_conditions(None) == {"edge_load_level": "L0", "edge_load": None}
    assert level_conditions(LEVEL)["edge_load"] == LEVEL.to_dict()


def test_generator_starts_checks_snapshots_and_stops(tmp_path: Path) -> None:
    generator = _generator(tmp_path)
    beat = generator.start()
    assert beat["status"] == "running" and beat["allocated_mib"] >= 512
    generator.check()
    snapshot = generator.snapshot()
    assert snapshot["load_alive"] is True
    assert snapshot["load_level"] == "L1"
    assert snapshot["load_duty"] == 0.4
    final = generator.stop()
    assert final is not None and final["status"] == "stopped"
    assert generator.stop() is not None  # Idempotent, no second session entry.
    sessions = (tmp_path / "load" / "sessions.jsonl").read_text().splitlines()
    assert len(sessions) == 1
    with pytest.raises(LoadGeneratorFailure, match="not running"):
        generator.check()


def test_generator_detects_a_dead_worker(tmp_path: Path) -> None:
    with _generator(tmp_path) as generator:
        beat = generator.heartbeat()
        assert beat is not None
        os.kill(int(beat["pid"]), signal.SIGKILL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                generator.check()
            except LoadGeneratorFailure:
                break
            time.sleep(0.05)
        else:
            pytest.fail("dead worker was not detected")
        assert generator.snapshot()["load_alive"] is False


def test_generator_detects_a_stale_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "stall")
    now = [time.time()]
    generator = _generator(tmp_path, clock=lambda: now[0], stale_after_s=5)
    generator.start()
    try:
        generator.check()
        now[0] += 60
        with pytest.raises(LoadGeneratorFailure, match="old"):
            generator.check()
    finally:
        generator.stop()


def test_generator_reports_a_failed_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "crash")
    with pytest.raises(LoadGeneratorFailure, match="did not start"):
        _generator(tmp_path).start()


class FakeWorkload:
    """Stands in for TorchWorkload: each step sleeps one millisecond."""

    def __init__(self, level: LoadLevel, device: str, fail_after: int = -1) -> None:
        self.allocated_mib = float(level.working_set_mib)
        self.steps = 0
        self.released = False
        self.fail_after = fail_after

    def step(self) -> None:
        if self.steps == self.fail_after:
            raise RuntimeError("CUDA error: an illegal memory access")
        self.steps += 1
        time.sleep(0.001)

    def release(self) -> None:
        self.released = True


def _in_process(
    tmp_path: Path, level: LoadLevel = LEVEL, **kwargs: object
) -> tuple[InProcessEdgeLoad, list[FakeWorkload]]:
    made: list[FakeWorkload] = []
    fail_after = int(kwargs.pop("fail_after", -1))  # type: ignore[call-overload]

    def workload(level: LoadLevel, device: str) -> FakeWorkload:
        made.append(FakeWorkload(level, device, fail_after))
        return made[-1]

    generator = InProcessEdgeLoad(
        level,
        directory=tmp_path / "load",
        workload=workload,
        start_timeout_s=10,
        heartbeat_s=0.01,
        **kwargs,  # type: ignore[arg-type]
    )
    return generator, made


def test_levels_choose_the_generator(tmp_path: Path) -> None:
    path = tmp_path / "levels.json"
    levels = {"L1": {"working_set_mib": 256, "busy_ms": 8, "idle_ms": 12}}
    path.write_text(
        json.dumps({"schema_version": 1, "generator": IN_PROCESS, "levels": levels})
    )
    level = load_levels(path)["L1"]
    assert level.generator == IN_PROCESS
    assert level_conditions(level)["edge_load"]["generator"] == IN_PROCESS
    assert isinstance(make_generator(level, directory=tmp_path), InProcessEdgeLoad)
    assert isinstance(make_generator(LEVEL, directory=tmp_path), EdgeLoadGenerator)
    with pytest.raises(ValueError, match="only edge load"):
        load_levels(path, no_load="C0")
    levels["L1"]["kernel_mib"] = 16
    path.write_text(
        json.dumps({"schema_version": 1, "generator": IN_PROCESS, "levels": levels})
    )
    assert load_levels(path)["L1"].kernel_mib == 16
    for generator, kernel_mib in ((IN_PROCESS, 512), ("process", 16)):
        levels["L1"]["kernel_mib"] = kernel_mib
        path.write_text(
            json.dumps({"schema_version": 1, "generator": generator, "levels": levels})
        )
        with pytest.raises(ValueError, match="kernel_mib"):
            load_levels(path)
    del levels["L1"]["kernel_mib"]
    path.write_text(
        json.dumps({"schema_version": 1, "generator": "thread", "levels": levels})
    )
    with pytest.raises(ValueError, match="generator"):
        load_levels(path)


def test_in_process_load_runs_checks_and_stops(tmp_path: Path) -> None:
    generator, made = _in_process(tmp_path)
    beat = generator.start()
    assert beat["status"] == "running" and beat["generator"] == IN_PROCESS
    assert beat["pid"] == os.getpid() and beat["allocated_mib"] == 512
    deadline = time.monotonic() + 10
    while (generator.heartbeat() or {}).get("kernels", 0) < 5:
        assert time.monotonic() < deadline, "load thread made no progress"
        time.sleep(0.01)
    generator.check()
    snapshot = generator.snapshot()
    assert snapshot["load_alive"] is True and snapshot["load_level"] == "L1"
    assert 0 < float(snapshot["load_duty"] or 0) <= 1
    final = generator.stop()
    assert final is not None and final["status"] == "stopped"
    assert made[0].released and final["kernels"] == made[0].steps
    assert generator.stop() is not None  # Idempotent, no second session entry.
    sessions = (tmp_path / "load" / "sessions.jsonl").read_text().splitlines()
    assert len(sessions) == 1 and json.loads(sessions[0])["status"] == "stopped"
    with pytest.raises(LoadGeneratorFailure, match="not running"):
        generator.check()
    assert generator.snapshot()["load_alive"] is False


def test_in_process_load_failure_is_detected(tmp_path: Path) -> None:
    generator, made = _in_process(tmp_path, fail_after=3)
    generator.start()
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                generator.check()
            except LoadGeneratorFailure as exc:
                assert "illegal memory access" in str(exc)
                break
            assert time.monotonic() < deadline, "failed load thread not detected"
            time.sleep(0.01)
        assert generator.snapshot()["load_alive"] is False
        assert made[0].released
    finally:
        final = generator.stop()
    assert final is not None and final["status"] == "failed"


def test_in_process_load_reports_a_failed_start(tmp_path: Path) -> None:
    def out_of_memory(level: LoadLevel, device: str) -> FakeWorkload:
        raise MemoryError("CUDA out of memory. Tried to allocate 512.00 MiB")

    generator = InProcessEdgeLoad(
        LEVEL, directory=tmp_path / "load", workload=out_of_memory
    )
    with pytest.raises(LoadGeneratorFailure, match="did not start: MemoryError"):
        generator.start()
    assert generator.snapshot()["load_alive"] is False


def test_in_process_load_detects_a_stalled_thread(tmp_path: Path) -> None:
    import threading

    now = [time.time()]
    hang = threading.Event()
    stepped = threading.Event()

    class Stalling(FakeWorkload):
        def step(self) -> None:
            stepped.set()
            hang.wait(10)  # A kernel that never finishes.

    generator = InProcessEdgeLoad(
        LEVEL,
        directory=tmp_path / "load",
        workload=Stalling,
        clock=lambda: now[0],
        stale_after_s=5,
    )
    generator.start()
    try:
        assert stepped.wait(10)
        generator.check()
        now[0] += 60
        with pytest.raises(LoadGeneratorFailure, match="old"):
            generator.check()
    finally:
        hang.set()
        generator.stop()


def test_memory_only_level_launches_no_kernels(tmp_path: Path) -> None:
    memory_only = LoadLevel("L1", working_set_mib=64, busy_ms=0.0, idle_ms=5.0)
    generator, made = _in_process(tmp_path, memory_only)
    with generator:
        time.sleep(0.05)
        generator.check()
    assert made[0].steps == 0 and made[0].released
