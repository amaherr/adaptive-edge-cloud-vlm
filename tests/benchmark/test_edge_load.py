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
    EdgeLoadGenerator,
    LoadGeneratorFailure,
    LoadLevel,
    level_conditions,
    load_levels,
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
