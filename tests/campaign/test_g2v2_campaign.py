"""G2 v2 grid, netem control, sub-block gates and the loopback supervisor."""

from __future__ import annotations

import itertools
import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.campaign.grid import check_split, load_grid
from adaptive_vlm.campaign.network import (
    LoopbackNetwork,
    PhysicalNetwork,
    check_netem,
    measurement_failures,
    netem_command,
    parse_qdisc,
)
from adaptive_vlm.campaign.subblock import Gates, SubBlockAborted, run_subblock
from adaptive_vlm.campaign.supervisor import CampaignSupervisor, load_settings
from adaptive_vlm.cli import main
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "configs/profiles/physical_network_grid_v2.json"
REHEARSAL_GRID = ROOT / "configs/profiles/rehearsal_grid_v2.json"
NETEM = (
    "qdisc netem 8001: root refcnt 2 limit 10000 delay 10ms rate 45Mbit\n"
    " Sent 81234 bytes 120 pkt (dropped {dropped}, overlimits 0 requeues 0)\n"
    " backlog 0b 0p requeues 0\n"
)


# -- grid ------------------------------------------------------------------


def test_grid_matches_the_protocol_counts_and_order_rules() -> None:
    grid = load_grid(GRID)
    split = json.loads(
        (ROOT / "configs/evaluation/physical_vqa_split_v2.json").read_text()
    )
    check_split(grid, split)
    assert len(grid.cells()) == 34
    order = grid.execution_order()
    assert len(order) == 102 == len({b.subblock_id for b in order})
    assert all(a.cell_id != b.cell_id for a, b in itertools.pairwise(order))
    assert grid.expected_attempts(split) == {"measured": 14580, "warmups": 306}
    assert [b.subblock_id for b in order] == [
        b.subblock_id for b in load_grid(GRID).execution_order()
    ]
    seeds = {grid.subblock_seed(b) for b in order}
    assert len(seeds) == 102
    t04 = grid.conditions["T04"]
    assert (t04.rate_mbps, t04.egress_delay_ms) == (45, 10)


def test_rate_fallback_replaces_only_the_predeclared_conditions() -> None:
    fallback = load_grid(GRID, rate_fallback=True)
    assert fallback.conditions["T08"].rate_mbps == 150
    assert fallback.conditions["E04"].rate_mbps == 140
    assert fallback.conditions["T07"] == load_grid(GRID).conditions["T07"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "schema_version 2"),
        ({"loaded_conditions": ["T99"]}, "not in the grid"),
        ({"load_levels": ["L0", "L1"]}, "implicit"),
        ({"repetitions": [0, 0]}, "unique"),
        ({"netem_limit_packets": 100}, "netem_limit"),
    ],
)
def test_grid_rejects_invalid_definitions(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    payload = {**json.loads(GRID.read_text()), **change}
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        load_grid(path)


def test_split_and_grid_must_agree_on_conditions() -> None:
    grid = load_grid(GRID)
    split = json.loads(
        (ROOT / "configs/evaluation/physical_vqa_split_v2.json").read_text()
    )
    with pytest.raises(ValueError, match="disagree"):
        check_split(grid, {**split, "validation_network_condition_ids": ["V01"]})
    with pytest.raises(ValueError, match="no evaluation"):
        check_split(grid, {**split, "evaluation_request_ids": []})


# -- network -----------------------------------------------------------------


def test_netem_command_and_qdisc_parsing() -> None:
    assert netem_command("eth0", rate_mbps=45, delay_ms=10, limit_packets=10000) == [
        "sudo", "-n", "tc", "qdisc", "replace", "dev", "eth0", "root", "netem",
        "delay", "10ms", "rate", "45mbit", "limit", "10000",
    ]  # fmt: skip
    state = parse_qdisc(NETEM.format(dropped=3))
    assert (state.kind, state.rate_mbps, state.delay_ms) == ("netem", 45.0, 10.0)
    assert (state.limit_packets, state.sent_packets, state.dropped) == (10000, 120, 3)
    check_netem(state, interface="eth0", rate_mbps=45, delay_ms=10, limit_packets=10000)
    for kwargs, message in (
        ({"rate_mbps": 50}, "rate"),
        ({"delay_ms": 25}, "delay"),
        ({"limit_packets": 1000}, "limit"),
    ):
        arguments: dict[str, Any] = {
            "interface": "eth0",
            "rate_mbps": 45,
            "delay_ms": 10,
            "limit_packets": 10000,
            **kwargs,
        }
        with pytest.raises(RuntimeError, match=message):
            check_netem(state, **arguments)
    default = parse_qdisc(
        "qdisc mq 0: root\n Sent 1 bytes 1 pkt (dropped 0, overlimits 0)"
    )
    assert default.kind == "mq"
    with pytest.raises(RuntimeError, match="expected netem"):
        check_netem(
            default, interface="eth0", rate_mbps=45, delay_ms=10, limit_packets=1
        )
    assert parse_qdisc("qdisc netem 1: root delay 500us rate 1500Kbit").rate_mbps == 1.5
    with pytest.raises(ValueError):
        parse_qdisc("")


def test_measurement_tolerances_follow_g2() -> None:
    good = {
        "edge_to_cloud_receiver_bps": 44e6,
        "cloud_to_edge_receiver_bps": 46e6,
        "echo_median_rtt_ms": 21.6,
    }
    assert not measurement_failures(
        good, rate_mbps=45, added_rtt_ms=20, baseline_rtt_ms=1.6
    )
    bad = {**good, "cloud_to_edge_receiver_bps": 30e6, "echo_median_rtt_ms": 40.0}
    failures = measurement_failures(
        bad, rate_mbps=45, added_rtt_ms=20, baseline_rtt_ms=1.6
    )
    assert len(failures) == 2


class FakeHost:
    def __init__(self, qdisc: str = "qdisc mq 0: root") -> None:
        self.qdisc = qdisc
        self.commands: list[list[str]] = []

    def __call__(self, command: Any) -> str:
        command = list(command)
        self.commands.append(command)
        if "replace" in command:
            self.qdisc = NETEM.format(dropped=0)
        elif "del" in command:
            self.qdisc = "qdisc mq 0: root"
        return self.qdisc


def _physical(edge: FakeHost, cloud: FakeHost) -> PhysicalNetwork:
    return PhysicalNetwork(
        edge_interface="eth0",
        cloud_interface="eth1",
        cloud_host="10.0.0.2",
        iperf_port=5051,
        echo_port=5052,
        limit_packets=10000,
        echo_script=ROOT / "scripts/network/tcp_rtt_probe.py",
        local=edge,
        remote=cloud,
    )


def test_physical_network_shapes_verifies_and_clears_both_hosts() -> None:
    edge, cloud = FakeHost(), FakeHost()
    network = _physical(edge, cloud)
    network.apply(rate_mbps=45, delay_ms=10, owned=False)
    states = network.verify(rate_mbps=45, delay_ms=10)
    assert {name: s.dropped for name, s in states.items()} == {"edge": 0, "cloud": 0}
    assert any("eth1" in c for c in cloud.commands if "replace" in c)
    assert network.clear() == []
    assert edge.qdisc.startswith("qdisc mq") and cloud.qdisc.startswith("qdisc mq")


def test_physical_network_refuses_a_foreign_qdisc() -> None:
    network = _physical(FakeHost("qdisc htb 1: root"), FakeHost())
    with pytest.raises(RuntimeError, match="unexpected edge qdisc"):
        network.apply(rate_mbps=45, delay_ms=10, owned=False)


# -- loopback sub-blocks and supervisor ---------------------------------------


@pytest.fixture
def fake_server() -> Iterator[None]:
    config = load_experiment_config(
        ROOT / "configs/experiments/fake_g2v2_rehearsal.yaml"
    )
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
        ),
    )
    try:
        server.start()
    except OSError as exc:  # pragma: no cover - sandbox dependent
        pytest.skip(f"loopback port unavailable: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def fake_edge(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand-ins for NVML, Windows counters and the GPU load worker."""

    import adaptive_vlm.commands.interleaved as command

    state: dict[str, Any] = {"dedicated": 3200.0, "load_alive": True, "starts": 0}

    class Reader:
        def __init__(self, index: int) -> None:
            pass

        def snapshot(self) -> dict[str, Any]:
            return {"gpu_temperature_c": 55}

    class Windows:
        def start(self) -> None:
            pass

        def wait_for_sample(self) -> bool:
            return True

        def snapshot(self) -> dict[str, Any]:
            return {"windows_gpu_dedicated_mib": state["dedicated"]}

        def stop(self) -> None:
            pass

    class Generator:
        def __init__(self, level: Any, *, directory: Path) -> None:
            self.level = level

        def start(self) -> None:
            state["starts"] += 1

        def check(self) -> None:
            pass

        def snapshot(self) -> dict[str, Any]:
            return {"load_level": self.level.name, "load_alive": state["load_alive"]}

        def stop(self) -> dict[str, Any]:
            return {"status": "stopped"}

    monkeypatch.setattr(command, "NvmlReader", Reader)
    monkeypatch.setattr(command, "WindowsGpuCounterSampler", Windows)
    monkeypatch.setattr(command, "EdgeLoadGenerator", Generator)
    return state


def _in_process(codes: list[int] | None = None) -> Any:
    """Run children through the CLI in-process; optionally fail first calls."""

    queued = list(codes or [])

    def run_child(command: list[str], run_dir: Path | None = None) -> int:
        if queued:
            return queued.pop(0)
        return main(command[1:])

    return run_child


def _supervisor(tmp_path: Path, **overrides: Any) -> CampaignSupervisor:
    settings = json.loads(
        (ROOT / "configs/campaigns/g2v2_loopback_rehearsal.json").read_text()
    )
    settings.update(output_root=str(tmp_path / "campaign"), **overrides)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    supervisor = CampaignSupervisor(load_settings(path, ROOT), root=ROOT)
    supervisor.run_child = _in_process()  # type: ignore[method-assign]
    return supervisor


def test_loopback_rehearsal_runs_every_subblock_and_skips_finished_ones(
    tmp_path: Path, fake_server: None, fake_edge: dict[str, Any]
) -> None:
    supervisor = _supervisor(tmp_path)
    supervisor.execute(preflight_only=False)
    output = tmp_path / "campaign"
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "completed"
    assert (
        status["measured_attempts"] == status["target_attempts"] == 3 * (6 + 3 + 3 + 6)
    )
    assert status["valid_subblocks"] == 4
    plan = json.loads((output / "campaign_plan.json").read_text())
    assert plan["protocol_pinned"] is False and plan["git_dirty"] in (True, False)
    loaded = json.loads((output / "subblocks/T01-L1-r0/run/edge/plan.json").read_text())
    assert loaded["conditions"]["edge_load_level"] == "L1"
    assert fake_edge["starts"] == 1
    # A second run validates the finished evidence and does no new work.
    again = _supervisor(tmp_path)
    again.run_child = _in_process([99])  # type: ignore[method-assign]
    again.execute(preflight_only=False)


def test_spill_makes_an_l0_subblock_invalid_and_it_is_rerun(
    tmp_path: Path, fake_server: None, fake_edge: dict[str, Any]
) -> None:
    fake_edge["dedicated"] = 3950.0
    supervisor = _supervisor(tmp_path, require_spill_evidence=True)
    with pytest.raises(RuntimeError, match="stayed invalid after 2 attempts"):
        supervisor.execute(preflight_only=False)
    invalid = list((tmp_path / "campaign" / "invalid").iterdir())
    assert len(invalid) == 1
    report = json.loads(
        (tmp_path / "campaign/subblocks" / invalid[0].name.rsplit("-", 1)[0]
         / "validation_report.json").read_text()
    )  # fmt: skip
    assert "edge spill" in report["reasons"][0]
    status = json.loads((tmp_path / "campaign/status.json").read_text())
    assert status["state"] == "failed"


def _block_args(tmp_path: Path) -> dict[str, Any]:
    grid = load_grid(REHEARSAL_GRID)
    block = next(b for b in grid.subblocks() if b.load_level == "L1")
    supervisor = _supervisor(tmp_path)
    return {
        "grid": grid,
        "block": block,
        "output": tmp_path / "block",
        "baseline_rtt_ms": None,
        "command_prefix": supervisor.command_prefix(),
        "gates": Gates(require_spill_evidence=False),
        "max_restarts": 1,
        "log": lambda message: None,
    }


def test_subblock_restarts_after_device_failure_then_aborts(
    tmp_path: Path, fake_server: None, fake_edge: dict[str, Any]
) -> None:
    arguments = _block_args(tmp_path)
    report = run_subblock(
        network=LoopbackNetwork(), run_child=_in_process([75]), **arguments
    )
    assert report["valid"] and report["restarts"] == 1
    with pytest.raises(SubBlockAborted, match="exited 76"):
        run_subblock(
            network=LoopbackNetwork(),
            run_child=_in_process([76, 76]),
            **{**arguments, "output": tmp_path / "other"},
        )


def test_dead_load_generator_and_netem_drops_invalidate(
    tmp_path: Path, fake_server: None, fake_edge: dict[str, Any]
) -> None:
    fake_edge["load_alive"] = False

    class DroppingNetwork(LoopbackNetwork):
        controlled = True
        calls = 0

        def verify(self, *, rate_mbps: float, delay_ms: float) -> dict[str, Any]:
            self.calls += 1
            state = parse_qdisc(NETEM.format(dropped=0 if self.calls == 1 else 7))
            return {"edge": state, "cloud": replace(state, dropped=0)}

        def measure(self) -> dict[str, Any]:
            return {
                "edge_to_cloud_receiver_bps": 5e6,
                "cloud_to_edge_receiver_bps": 5e6,
                "echo_median_rtt_ms": 21.5,
            }

    arguments = {**_block_args(tmp_path), "baseline_rtt_ms": 1.5}
    report = run_subblock(
        network=DroppingNetwork(), run_child=_in_process(), **arguments
    )
    assert not report["valid"]
    reasons = " | ".join(report["reasons"])
    assert "edge netem dropped packets (0 -> 7)" in reasons
    assert "load generator not alive at the first request" in reasons


def test_settings_validation(tmp_path: Path) -> None:
    base = json.loads(
        (ROOT / "configs/campaigns/g2v2_physical.example.json").read_text()
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="set cloud_ssh"):
        load_settings(path, ROOT)
    filled = {
        **base,
        "cloud_ssh": "edge@cloud",
        "cloud_repo": "/repo",
        "cloud_python": "/repo/.venv-model/bin/python",
        "same_device_reference_evidence": "artifacts/reference/index.json",
    }
    path.write_text(json.dumps({**filled, "allow_dirty_tree": True}))
    with pytest.raises(ValueError, match="only in loopback"):
        load_settings(path, ROOT)
    path.write_text(json.dumps({**filled, "stall_seconds": 60}))
    with pytest.raises(ValueError, match="stall_seconds"):
        load_settings(path, ROOT)
    path.write_text(json.dumps(filled))
    settings = load_settings(path, ROOT)
    assert settings["python"].endswith(".venv-model/bin/python")
    assert settings["max_subblock_attempts"] == 2
    path.write_text(json.dumps({**filled, "mode": "remote"}))
    with pytest.raises(ValueError, match="mode"):
        load_settings(path, ROOT)


def test_shaping_failure_still_cleans_up(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    cleared: list[bool] = []

    class Failing(LoopbackNetwork):
        def apply(self, *, rate_mbps: float, delay_ms: float, owned: bool) -> None:
            raise RuntimeError("tc failed")

        def clear(self) -> list[str]:
            cleared.append(True)
            return []

    supervisor.network = Failing()
    supervisor.output.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="tc failed"):
        supervisor.shape(45, 10)
    assert supervisor.shaped
    assert supervisor.cleanup() == [] and cleared == [True]
    assert not (supervisor.output / "shaping_active.json").exists()


# -- downstream: Predictor V2 partitions and measured-oracle tables ----------

REHEARSAL_CONTRACT = ROOT / "configs/predictor/rehearsal_dataset_contract.json"


def _derived(tmp_path: Path) -> tuple[Path, Path, Path]:
    from adaptive_vlm.prediction.v2.dataset.campaign_v2 import (
        profile_campaign,
        write_feature_bundle,
    )

    _supervisor(tmp_path).execute(preflight_only=False)
    campaign, derived = tmp_path / "campaign", tmp_path / "derived"
    write_feature_bundle(campaign, REHEARSAL_CONTRACT, derived / "features.json")
    profile_campaign(
        campaign=campaign, contract_path=REHEARSAL_CONTRACT, output=derived / "profiles"
    )
    return campaign, derived, derived / "profiles"


def test_rehearsal_produces_partitions_and_oracle_without_edits(
    tmp_path: Path, fake_server: None, fake_edge: dict[str, Any]
) -> None:
    from adaptive_vlm.evaluation.measured_oracle import build_measured_oracle
    from adaptive_vlm.prediction.v2.dataset.campaign_v2 import (
        EDGE_STATE_KEYS,
        prepare_campaign_partitions,
    )

    campaign, derived, profiles = _derived(tmp_path)
    assert sorted(p.name for p in profiles.iterdir() if p.is_dir()) == [
        "T01-L0-r0",
        "T01-L1-r0",
        "V01-L0-r0",
    ]  # The evaluation sub-block is never profiled by default.
    receipt = prepare_campaign_partitions(
        campaign=campaign,
        profiles=profiles,
        contract_path=REHEARSAL_CONTRACT,
        feature_path=derived / "features.json",
        output=derived / "partitions",
        sealed_root=tmp_path / "sealed",
    )
    assert receipt["classification"] == {
        "train": 36,
        "validation": 9,
        "final_test_protected": 9,
    }
    assert receipt["training_load_levels"] == {"L0": 18, "L1": 18}
    assert receipt["evaluation_profiles_opened"] == []
    sealed = Path(receipt["sealed_validation_path"])
    assert sealed.stat().st_mode & 0o777 == 0
    rows = [
        json.loads(line)
        for line in (derived / "partitions/training.jsonl").read_text().splitlines()
    ]
    inputs = rows[0]["inputs"]
    assert inputs["schema_version"] == 3 and inputs["split"] == "training"
    assert set(inputs["edge_state"]) == set(EDGE_STATE_KEYS)
    assert "server_state_gpu_temperature_c" not in json.dumps(inputs)
    loaded = [r for r in rows if r["inputs"]["edge_load_level"] == "L1"]
    assert loaded and all(r["inputs"]["edge_state"]["edge_load_alive"] for r in loaded)

    summary = build_measured_oracle(
        campaign=campaign, profiles=profiles, output=derived / "oracle"
    )
    assert summary["attempts"] == 12 and len(summary["cells"]) == 2
    cell = summary["cells"][0]
    assert cell["winner"] in cell["placements"]
    assert all(s["ok"] == 6 for s in cell["placements"].values())
    with pytest.raises(ValueError, match="sealed"):
        build_measured_oracle(
            campaign=campaign,
            profiles=profiles,
            output=derived / "oracle-eval",
            splits=("evaluation",),
        )


def test_partitions_reject_a_mixed_split_row(
    tmp_path: Path, fake_server: None, fake_edge: dict[str, Any]
) -> None:
    import hashlib

    from adaptive_vlm.prediction.v2.dataset.campaign_v2 import (
        prepare_campaign_partitions,
    )

    campaign, derived, profiles = _derived(tmp_path)
    split = json.loads(
        (ROOT / "configs/evaluation/rehearsal_split_v2.json").read_text()
    )
    path = profiles / "T01-L0-r0/profile/profile.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[-1])
    row["request_id"] = split["evaluation_request_ids"][0]
    lines[-1] = json.dumps(row, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    index_path = profiles / "profiles_index.json"
    index = json.loads(index_path.read_text())
    index["subblocks"]["T01-L0-r0"]["profile_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="mixed-split row"):
        prepare_campaign_partitions(
            campaign=campaign,
            profiles=profiles,
            contract_path=REHEARSAL_CONTRACT,
            feature_path=derived / "features.json",
            output=derived / "partitions",
            sealed_root=tmp_path / "sealed",
        )


def test_partitions_refuse_changed_profiles_and_uncontrolled_physical_contract(
    tmp_path: Path, fake_server: None, fake_edge: dict[str, Any]
) -> None:
    from adaptive_vlm.prediction.v2.dataset.campaign_v2 import (
        prepare_campaign_partitions,
        profile_campaign,
    )

    campaign, derived, profiles = _derived(tmp_path)
    path = profiles / "V01-L0-r0/profile/profile.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="profile changed"):
        prepare_campaign_partitions(
            campaign=campaign,
            profiles=profiles,
            contract_path=REHEARSAL_CONTRACT,
            feature_path=derived / "features.json",
            output=derived / "partitions",
            sealed_root=tmp_path / "sealed",
        )
    with pytest.raises(ValueError, match="physical"):
        profile_campaign(
            campaign=campaign,
            contract_path=ROOT / "configs/predictor/g2v2_dataset_contract.json",
            output=derived / "physical-profiles",
        )


def test_supervisor_stops_stray_campaign_processes(tmp_path: Path) -> None:
    import subprocess
    import sys

    supervisor = _supervisor(tmp_path)
    marker = supervisor.output / "subblocks" / "T01-L0-r0" / "run"
    stray = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", str(marker)]
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        supervisor.output.mkdir(parents=True)
        assert supervisor.reap_strays() == [stray.pid]
        assert stray.wait(timeout=20) != 0
        assert unrelated.poll() is None
        log = (supervisor.output / "stray_processes.jsonl").read_text()
        assert str(stray.pid) in log
        assert supervisor.reap_strays() == []
    finally:
        for process in (stray, unrelated):
            process.kill()
            process.wait()


def test_a_second_writer_cannot_join_a_running_benchmark(tmp_path: Path) -> None:
    from adaptive_vlm.benchmark.interleaved import run_interleaved_benchmark
    from adaptive_vlm.process import exclusive_lock

    run_dir = tmp_path / "run"
    with (
        exclusive_lock(tmp_path / ".run.writer.lock", "run writer"),
        pytest.raises(RuntimeError, match="run writer lock"),
    ):
        run_interleaved_benchmark(run_dir=run_dir)
    assert not run_dir.exists()
