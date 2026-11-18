"""Cloud GPU load (C1): control command, server probe, gate and lifecycle."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_g2v2_campaign import _in_process, fake_edge  # noqa: F401

from adaptive_vlm.benchmark import cloud_load as control
from adaptive_vlm.benchmark.edge_load import LoadLevel
from adaptive_vlm.campaign.cloud_load import CloudLoad, CloudLoadFailure
from adaptive_vlm.campaign.grid import load_grid
from adaptive_vlm.campaign.network import LoopbackNetwork
from adaptive_vlm.campaign.remote import RemoteCommandError
from adaptive_vlm.campaign.subblock import Gates, cloud_load_reasons, run_subblock
from adaptive_vlm.campaign.supervisor import CampaignSupervisor, load_settings
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.resource_state import combine
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "configs/profiles/rehearsal_grid_v2plus.json"
C1 = LoadLevel("C1", 64, 10, 10)

# A torch-free stand-in for the GPU worker: same arguments and heartbeat.
WORKER = """
import json, os, signal, sys, time
args = sys.argv[1:]
path = args[args.index("--heartbeat") + 1]
state = {"status": "running", "pid": os.getpid(),
         "level": args[args.index("--level") + 1],
         "allocated_mib": float(args[args.index("--working-set-mib") + 1]),
         "duty": 0.5}
stop = []
signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
while not stop:
    state["updated_unix_s"] = time.time()
    open(path + ".tmp", "w").write(json.dumps(state))
    os.replace(path + ".tmp", path)
    time.sleep(0.1)
state.update(status="stopped", updated_unix_s=time.time())
open(path, "w").write(json.dumps(state))
"""


def test_control_starts_reports_and_stops_a_detached_worker(tmp_path: Path) -> None:
    heartbeat = tmp_path / "load" / "heartbeat.json"
    worker = [sys.executable, "-c", WORKER]
    started = control.start(
        heartbeat,
        level="C1",
        working_set_mib=64,
        busy_ms=10,
        idle_ms=10,
        device="cpu",
        timeout_s=20,
        worker=worker,
    )
    try:
        assert started["started"] and started["heartbeat"]["level"] == "C1"
        state = control.status(heartbeat)
        assert state["alive"] and state["pids"] == [started["pid"]]
        probe = control.heartbeat_source(heartbeat)()
        assert probe["load_alive"] is True and probe["load_level"] == "C1"
        assert probe["load_heartbeat_age_s"] is not None
        assert probe["load_heartbeat_age_s"] < 5
        # Starting again replaces the worker instead of adding a second one.
        again = control.start(
            heartbeat,
            level="C1",
            working_set_mib=64,
            busy_ms=10,
            idle_ms=10,
            device="cpu",
            timeout_s=20,
            worker=worker,
        )
        assert control.status(heartbeat)["pids"] == [again["pid"]]
    finally:
        stopped = control.stop(heartbeat, grace_s=5)
    assert stopped["remaining"] == [] and stopped["stopped"]
    assert stopped["final_heartbeat"]["status"] == "stopped"
    assert control.heartbeat_source(heartbeat)()["load_alive"] is False
    assert control.stop(heartbeat)["stopped"] == []  # reap with nothing left
    with pytest.raises(ValueError, match="C0"):
        control.start(
            heartbeat,
            level="C0",
            working_set_mib=1,
            busy_ms=1,
            idle_ms=1,
            device="cpu",
            timeout_s=1,
        )


def test_server_probe_without_a_heartbeat_reports_no_load(tmp_path: Path) -> None:
    probe = control.heartbeat_source(tmp_path / "missing.json")()
    assert probe == {
        "load_level": "C0",
        "load_alive": False,
        "load_heartbeat_age_s": None,
        "load_duty": None,
    }
    beat = tmp_path / "dead.json"
    beat.write_text(
        json.dumps({"status": "running", "pid": 2**22 + 7, "updated_unix_s": 0})
    )
    assert control.heartbeat_source(beat)()["load_alive"] is False  # no such PID


class Recorder:
    def __init__(self, outputs: list[str] | None = None, fail: bool = False) -> None:
        self.commands: list[list[str]] = []
        self.outputs = outputs or []
        self.fail = fail

    def __call__(self, command: Any) -> str:
        self.commands.append(list(command))
        if self.fail:
            raise RemoteCommandError("ssm failed")
        return self.outputs.pop(0) if self.outputs else '{"stopped": []}'


def test_supervisor_wrapper_builds_commands_and_reports_failures() -> None:
    runner = Recorder(['log line\n{"started": true, "pid": 9}', '{"stopped": [9]}'])
    load = CloudLoad(
        runner,
        ["docker", "exec", "adaptive-vlm-inference", "python"],
        heartbeat="/tmp/hb.json",
    )
    assert load.start(C1)["pid"] == 9
    assert load.reap() == [9]
    start = runner.commands[0]
    assert start[:7] == [
        "docker",
        "exec",
        "adaptive-vlm-inference",
        "python",
        "-m",
        "adaptive_vlm.benchmark.cloud_load",
        "start",
    ]
    assert start[start.index("--working-set-mib") + 1] == "64"
    assert "--heartbeat-file" in start
    with pytest.raises(CloudLoadFailure, match="ssm failed"):
        CloudLoad(Recorder(fail=True), ["python"]).stop()
    with pytest.raises(CloudLoadFailure, match="no JSON"):
        CloudLoad(Recorder(["oops"]), ["python"]).status()
    with pytest.raises(CloudLoadFailure, match="did not start"):
        CloudLoad(Recorder(['{"started": false}']), ["python"]).start(C1)


def _row(alive: bool | None, age: float = 1.0, mode: str = "cloud") -> dict:
    metrics = (
        {}
        if alive is None
        else {
            "server_state_load_alive": alive,
            "server_state_load_level": "C1" if alive else "C0",
            "server_state_load_heartbeat_age_s": age,
        }
    )
    return {"status": "ok", "mode": mode, "metrics": metrics}


def test_cloud_load_gate() -> None:
    grid = load_grid(GRID)
    c1 = next(b for b in grid.subblocks() if b.cloud_load == "C1")
    c0 = next(b for b in grid.subblocks() if b.cloud_load == "C0")
    ok = [_row(True), _row(True, mode="distributed"), _row(None, mode="edge")]
    assert cloud_load_reasons(ok, c1, max_age_s=5) == []
    assert (
        "not alive in 1" in cloud_load_reasons([*ok, _row(False)], c1, max_age_s=5)[0]
    )
    assert "old in 1" in cloud_load_reasons([*ok, _row(True, 7)], c1, max_age_s=5)[0]
    assert "missing" in cloud_load_reasons([_row(None)], c1, max_age_s=5)[0]
    assert cloud_load_reasons([], c1, max_age_s=5)
    assert cloud_load_reasons([_row(False)], c0, max_age_s=5) == []
    assert "alive in 1 C0" in cloud_load_reasons([_row(True)], c0, max_age_s=5)[0]


def test_grid_cloud_cells_ids_and_calibration(tmp_path: Path) -> None:
    grid = load_grid(GRID)
    ids = {b.subblock_id for b in grid.subblocks()}
    assert "T01-L1-C1-r0" in ids and "T01-L1-r0" in ids and len(ids) == 7
    assert grid.cloud_loads() == {"C1": C1}
    payload = json.loads(GRID.read_text())
    for key in ("protocol_path", "split_path", "experiment_config_path"):
        payload[key] = str((GRID.parent / payload[key]).resolve())
    for task in payload["extra_tasks"].values():
        for key in ("split_path", "experiment_config_path"):
            task[key] = str((GRID.parent / task[key]).resolve())
    uncalibrated = tmp_path / "levels.json"
    uncalibrated.write_text(json.dumps({"schema_version": 1, "levels": {}}))
    payload["cloud_load"]["levels_path"] = str(uncalibrated)
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not calibrated"):
        load_grid(path).cloud_loads()
    payload["cloud_load"]["cells"] = [["V01", "L1", "short"]]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="same C0 cell"):
        load_grid(path)


# -- loopback campaign with a stand-in cloud worker ----------------------------


class FakeCloudLoad:
    """Writes the heartbeat the fake server reports; optionally misbehaves."""

    def __init__(self, heartbeat: Path) -> None:
        self.heartbeat = heartbeat
        self.events: list[str] = []
        self.keep_after_stop = False  # a stray that survives into C0
        self.die_after_start = False

    def _write(self, status: str) -> None:
        self.heartbeat.write_text(
            json.dumps(
                {
                    "status": status,
                    "pid": os.getpid(),
                    "level": "C1",
                    "updated_unix_s": time.time(),
                    "duty": 0.5,
                }
            )
        )

    def start(self, level: LoadLevel) -> dict[str, Any]:
        self.events.append(f"start {level.name}")
        self._write("stopped" if self.die_after_start else "running")
        return {"started": True, "pid": os.getpid()}

    def stop(self) -> dict[str, Any]:
        self.events.append("stop")
        if not self.keep_after_stop:
            self._write("stopped")
        return {"stopped": [os.getpid()], "remaining": []}

    def reap(self) -> list[int]:
        self.events.append("reap")
        return []

    def refresh(self) -> None:
        """Keep a running heartbeat fresh (the real worker beats every second)."""

        beat = json.loads(self.heartbeat.read_text())
        if beat["status"] == "running":
            self._write("running")


@pytest.fixture
def cloud(tmp_path: Path) -> Iterator[FakeCloudLoad]:
    config = load_experiment_config(
        ROOT / "configs/experiments/fake_g2v2_rehearsal.yaml"
    )
    heartbeat = tmp_path / "cloud-heartbeat.json"
    fake = FakeCloudLoad(heartbeat)
    probe = control.heartbeat_source(heartbeat)

    def state() -> dict[str, Any]:
        if heartbeat.exists():
            fake.refresh()
        return combine(probe)()

    adapter = FakeModelAdapter()
    server = CloudServer(
        host="127.0.0.1",
        port=config.transport.port,
        timeout_seconds=5.0,
        processor=CloudRequestProcessor(
            protocol_version=1,
            model_fingerprint=model_contract_fingerprint(config.model),
            inference_function=adapter.generate,
            tensor_inference_function=adapter.generate_from_payload,
            prompt_prepare_function=adapter.prepare_prompt,
            prepared_tensor_inference_function=adapter.generate_from_payload,
            state_provider=state,
        ),
    )
    try:
        server.start()
    except OSError as exc:  # pragma: no cover - sandbox dependent
        pytest.skip(f"loopback port unavailable: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield fake
    server.shutdown()
    thread.join(timeout=5)


def _supervisor(tmp_path: Path, fake: FakeCloudLoad) -> CampaignSupervisor:
    settings = json.loads(
        (ROOT / "configs/campaigns/g2v2plus_loopback_rehearsal.json").read_text()
    )
    settings.update(output_root=str(tmp_path / "campaign"))
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    supervisor = CampaignSupervisor(load_settings(path, ROOT), root=ROOT)
    supervisor.run_child = _in_process()  # type: ignore[method-assign]
    supervisor.cloud_load = fake  # type: ignore[assignment]
    return supervisor


def test_loopback_campaign_runs_five_configurations_and_a_c1_cell(
    tmp_path: Path,
    cloud: FakeCloudLoad,
    fake_edge: dict[str, Any],  # noqa: F811
) -> None:
    supervisor = _supervisor(tmp_path, cloud)
    supervisor.execute(preflight_only=False)
    output = tmp_path / "campaign"
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "completed" and status["valid_subblocks"] == 7
    assert cloud.events.count("start C1") == 1 and cloud.events.count("stop") == 1
    assert cloud.events.count("reap") >= 8  # preflight, each sub-block, exit
    c1 = output / "subblocks/T01-L1-C1-r0"
    assert json.loads((c1 / "plan.json").read_text())["cloud_load"] == "C1"
    rows = [
        json.loads(line)
        for line in (c1 / "run/distributed_int8/requests.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(row["metrics"]["server_state_load_alive"] for row in rows)
    events = [
        json.loads(line)["event"]
        for line in (c1 / "cloud_load.jsonl").read_text().splitlines()
    ]
    assert events == ["started", "stopped"]


def _block_args(tmp_path: Path, fake: FakeCloudLoad, cloud_load: str) -> dict:
    supervisor = _supervisor(tmp_path, fake)
    grid = supervisor.grid
    block = next(
        b
        for b in grid.subblocks()
        if b.load_level == "L1" and b.task == "short" and b.cloud_load == cloud_load
    )
    return {
        "grid": grid,
        "block": block,
        "output": tmp_path / cloud_load,
        "network": LoopbackNetwork(),
        "baseline_rtt_ms": None,
        "command_prefix": supervisor.command_prefix(),
        "run_child": _in_process(),
        "gates": Gates(require_spill_evidence=False),
        "max_restarts": 0,
        "log": lambda message: None,
        "cloud_load": fake,
        "cloud_level": C1,
    }


def test_dead_worker_or_stray_worker_invalidates_and_crash_still_stops(
    tmp_path: Path,
    cloud: FakeCloudLoad,
    fake_edge: dict[str, Any],  # noqa: F811
) -> None:
    cloud.die_after_start = True
    report = run_subblock(**_block_args(tmp_path, cloud, "C1"))
    assert not report["valid"] and "not alive" in " ".join(report["reasons"])

    cloud.die_after_start = False
    cloud.keep_after_stop = True
    run_subblock(**{**_block_args(tmp_path, cloud, "C1"), "output": tmp_path / "x"})
    stray = run_subblock(**_block_args(tmp_path, cloud, "C0"))
    assert not stray["valid"] and "C0 responses" in " ".join(stray["reasons"])

    cloud.keep_after_stop = False
    cloud.events.clear()

    def crash(command: list[str], run_dir: Path | None = None) -> int:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_subblock(
            **{
                **_block_args(tmp_path, cloud, "C1"),
                "output": tmp_path / "crash",
                "run_child": crash,
            }
        )
    assert cloud.events == ["reap", "start C1", "stop"]


def test_grid_with_two_cloud_levels(tmp_path: Path) -> None:
    """Amendment A2a-5: C1 and C2 cells in one grid, each with its own level."""

    payload = json.loads(GRID.read_text())
    for key in ("protocol_path", "split_path", "experiment_config_path"):
        payload[key] = str((GRID.parent / payload[key]).resolve())
    for task in payload["extra_tasks"].values():
        for key in ("split_path", "experiment_config_path"):
            task[key] = str((GRID.parent / task[key]).resolve())
    levels = tmp_path / "levels.json"
    heavy = {**C1.to_dict(), "busy_ms": 20, "idle_ms": 0}
    calibrated = {"C1": C1.to_dict(), "C2": heavy}
    levels.write_text(json.dumps({"schema_version": 1, "levels": calibrated}))
    payload["cloud_load"] = {
        "levels_path": str(levels),
        "levels": {"C1": [["T01", "L1", "short"]], "C2": [["T01", "L0", "short"]]},
    }
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(payload))
    grid = load_grid(path)
    assert grid.cloud_load_levels == ("C1", "C2")
    ids = {b.subblock_id for b in grid.subblocks()}
    assert {"T01-L1-C1-r0", "T01-L0-C2-r0", "T01-L0-r0"} <= ids
    loads = grid.cloud_loads()
    assert loads["C1"] == C1 and loads["C2"].busy_ms == 20
    calibrated.pop("C2")
    levels.write_text(json.dumps({"schema_version": 1, "levels": calibrated}))
    with pytest.raises(ValueError, match="C2 is not calibrated"):
        grid.cloud_loads()
