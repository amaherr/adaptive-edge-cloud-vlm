"""G5 (amendment A4): sensing, the live adaptive policy, traces and loopback."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.interleaved import (
    ALL_PLACEMENTS,
    make_interleaved_order,
    run_interleaved_benchmark,
)
from adaptive_vlm.campaign.grid import load_grid
from adaptive_vlm.campaign.subblock import benchmark_arguments, cloud_load_reasons
from adaptive_vlm.cli import main
from adaptive_vlm.config import ExecutionMode, load_experiment_config
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.remote import check_remote_health
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.prediction.r2.live import (
    AdaptiveRecord,
    AdaptiveRunner,
    NoFeasibleConfiguration,
)
from adaptive_vlm.prediction.r2.models import fit_component
from adaptive_vlm.scheduler.configuration import SPEC_BY_PLACEMENT
from adaptive_vlm.scheduler.sensing import (
    SensingConfig,
    Sensor,
    classify_cloud,
    classify_edge,
    network_from_probe,
)
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer
from tests.benchmark.test_interleaved import IDS, StubPolicy
from tests.prediction.test_r2 import _world

ROOT = Path(__file__).resolve().parents[2]
SENSING = SensingConfig(cloud_power_threshold_w=58.0, edge_probe_threshold_ms=20.0)


def test_cloud_and_edge_classification() -> None:
    assert (
        classify_cloud({"gpu_power_w": 70.1, "sample_age_s": 0.2}, SENSING)[0] == "C2"
    )
    assert classify_cloud({"gpu_power_w": 50.6}, SENSING)[0] == "C0"
    assert classify_cloud({"gpu_power_w": 58.0}, SENSING)[0] == "C2"
    level, *_, source = classify_cloud(None, SENSING)
    assert (level, source) == ("C0", "missing")
    level, *_, source = classify_cloud({"gpu_utilization_percent": 99}, SENSING)
    assert (level, source) == ("C0", "no_power_reading")
    assert classify_edge(19.9, SENSING) == "L0"
    assert classify_edge(20.0, SENSING) == "L1"
    with pytest.raises(ValueError):
        SensingConfig(cloud_power_threshold_w=0.0, edge_probe_threshold_ms=1.0)


def test_sensing_config_loads_with_its_hash(tmp_path: Path) -> None:
    path = tmp_path / "sensing.json"
    path.write_text(
        json.dumps(
            {
                "schema": "g5-sensing",
                "cloud_power_threshold_w": 58.0,
                "edge_probe_threshold_ms": 12.5,
                "note": "ignored field",
            }
        )
    )
    config, digest = SensingConfig.load(path)
    assert config.edge_probe_threshold_ms == 12.5 and len(digest) == 64
    path.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(ValueError, match="sensing"):
        SensingConfig.load(path)


def test_network_from_probe(tmp_path: Path) -> None:
    probe = tmp_path / "network_pre.json"
    probe.write_text(
        json.dumps({"edge_to_cloud_receiver_bps": 2.5e7, "echo_median_rtt_ms": 50.2})
    )
    assert network_from_probe(probe) == (2.5e7, 50.2)


def _calibration() -> dict[str, Any]:
    return fit_component(_world(range(4))) | {
        "provenance": {"training_units_sha256": "a" * 64}
    }


def _features() -> dict[str, dict[str, Any]]:
    return {
        rid: {"image_width": 640, "image_height": 480, "image_bytes": 120_000}
        for rid in IDS
    }


def _stub_policies(log: list[str]) -> dict[str, ExecutionPolicy]:
    return {
        placement: StubPolicy(ExecutionMode(SPEC_BY_PLACEMENT[placement].mode), log)
        for placement in ALL_PLACEMENTS
    }


def _runner(
    log: list[str],
    *,
    probe_ms: float,
    power: float,
    rate: float = 5e6,
    rtt: float = 20.0,
) -> AdaptiveRunner:
    policies = _stub_policies(log)
    sensor = Sensor(
        SENSING,
        edge_probe=lambda: probe_ms,
        cloud_state=lambda: {"gpu_power_w": power, "sample_age_s": 0.3},
        network=(rate, rtt),
    )
    return AdaptiveRunner(
        calibration=_calibration(),
        record=AdaptiveRecord("c" * 64, "s" * 64, "n" * 64, "short"),
        sensor=sensor,
        policies={SPEC_BY_PLACEMENT[p].id.value: pol for p, pol in policies.items()},
        features=_features(),
    )


def test_adaptive_runner_senses_selects_and_runs_one_configuration() -> None:
    _, manifest = _setup()
    request = next(i for i in manifest.items if i.request_id == IDS[0]).to_request()
    unloaded = _runner([], probe_ms=5.0, power=30.0, rate=1.5e8, rtt=5.0).infer(request)
    metrics = unloaded.metrics
    assert metrics["adaptive_sensed_cloud_load"] == "C0"
    assert metrics["adaptive_sensed_edge_load"] == "L0"
    assert metrics["adaptive_selected_configuration_id"] == "cloud/raw_image"
    assert metrics["adaptive_overhead_ms"] >= metrics["adaptive_selection_ms"] >= 0
    loaded = _runner([], probe_ms=5.0, power=70.0, rate=5e6, rtt=150.0).infer(request)
    assert loaded.metrics["adaptive_sensed_cloud_load"] == "C2"
    assert loaded.metrics["adaptive_selected_configuration_id"] == "edge/full"
    both = _runner([], probe_ms=40.0, power=70.0, rate=1.5e8, rtt=5.0).infer(request)
    assert both.metrics["adaptive_sensed_edge_load"] == "L1"
    predicted = {
        k.removeprefix("adaptive_predicted_ms_"): v
        for k, v in both.metrics.items()
        if k.startswith("adaptive_predicted_ms_")
    }
    assert len(predicted) == 5
    assert both.metrics["adaptive_selected_configuration_id"] == min(
        predicted, key=lambda c: float(predicted[c] or 0.0)
    )


def test_adaptive_runner_refuses_other_models_and_missing_configurations() -> None:
    with pytest.raises(ValueError, match="r2_component"):
        AdaptiveRunner(
            calibration={"model": "r2_additive"},
            record=AdaptiveRecord("c", "s", "n", "short"),
            sensor=_runner([], probe_ms=1.0, power=1.0).sensor,
            policies={},
            features={},
        )
    runner = _runner([], probe_ms=1.0, power=1.0)
    runner.calibration = dict(runner.calibration, edge={}, server={}, residual={})
    _, manifest = _setup()
    with pytest.raises(NoFeasibleConfiguration):
        runner.infer(manifest.items[0].to_request())


def _setup() -> tuple[Any, Any]:
    config = load_experiment_config(ROOT / "configs/experiments/local_campaign.yaml")
    return config, load_frozen_manifest(config.dataset.manifest_path)


def test_order_and_run_include_the_adaptive_placement(tmp_path: Path) -> None:
    order = make_interleaved_order(
        IDS[:2],
        repetitions=[0],
        warmups=0,
        placements=[*ALL_PLACEMENTS, "adaptive"],
        seed=1,
    )
    assert sum(p == "adaptive" for p, _ in order) == 2
    log: list[str] = []
    policies = _stub_policies(log)
    runner = _runner(log, probe_ms=5.0, power=70.0)
    runner.policies = {
        SPEC_BY_PLACEMENT[p].id.value: pol for p, pol in policies.items()
    }
    config, manifest = _setup()
    run_interleaved_benchmark(
        config=config,
        manifest=manifest,
        policies=policies,
        run_manifests={
            p: create_run_manifest(config, {}, dataset_manifest_sha256=manifest.sha256)
            for p in [*ALL_PLACEMENTS, "adaptive"]
        },
        run_dir=tmp_path / "run",
        request_ids=IDS,
        repetitions=[0],
        warmups=1,
        seed=7,
        adaptive=runner,
    )
    plan = json.loads((tmp_path / "run" / "adaptive" / "plan.json").read_text())
    assert plan["mode"] == "adaptive" and plan["adaptive"]["task"] == "short"
    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "adaptive" / "requests.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 1 + len(IDS)
    assert all(row["status"] == "ok" for row in rows)
    assert {row["metrics"]["adaptive_selected_configuration_id"] for row in rows}
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert set(summary["placements"]) == {*ALL_PLACEMENTS, "adaptive"}


def test_cloud_load_gate_skips_adaptive_rows_that_ran_on_the_edge() -> None:
    grid = load_grid(ROOT / "configs/profiles/aws_network_grid_v2plus.json")
    block = next(b for b in grid.subblocks() if b.cloud_load == "C2")
    alive = {
        "server_state_load_alive": True,
        "server_state_load_level": "C2",
        "server_state_load_heartbeat_age_s": 0.5,
    }
    rows = [
        {"status": "ok", "mode": "cloud", "metrics": alive},
        {
            "status": "ok",
            "mode": "adaptive",
            "metrics": {"adaptive_selected_configuration_id": "edge/full"},
        },
    ]
    assert cloud_load_reasons(rows, block, max_age_s=5.0) == []


def _trace_grid(tmp_path: Path) -> Path:
    payload = json.loads(
        (ROOT / "configs/profiles/aws_network_grid_v2plus.json").read_text()
    )
    base = ROOT / "configs/profiles"
    for key in (
        "protocol_path",
        "split_path",
        "experiment_config_path",
        "edge_load_levels_path",
    ):
        payload[key] = str((base / payload[key]).resolve())
    payload["cloud_load"]["levels_path"] = str(
        (base / payload["cloud_load"]["levels_path"]).resolve()
    )
    payload.pop("extra_tasks", None)
    payload["repetitions"] = [0, 1]
    payload["traces"] = {
        "net": [["E04", "L0", "C2"], ["E01", "L0", "C2"], ["E05", "L0", "C2"]],
        "cloud": [["E01", "L0", "C0"], ["E01", "L0", "C2"], ["E01", "L0", "C0"]],
    }
    payload["adaptive"] = {
        "calibration_path": "calibration.json",
        "sensing_path": "sensing.json",
        "features_path": "features.json",
    }
    path = tmp_path / "g5_grid.json"
    path.write_text(json.dumps(payload))
    return path


def test_trace_grid_orders_segments_and_passes_adaptive_inputs(tmp_path: Path) -> None:
    grid = load_grid(_trace_grid(tmp_path))
    order = grid.execution_order()
    assert len(order) == 2 * 6
    ids = [b.subblock_id for b in order]
    assert len(set(ids)) == len(ids)
    for repetition in (0, 1):
        for trace in ("net", "cloud"):
            segments = [
                b.segment
                for b in order
                if b.repetition == repetition and str(b.segment).startswith(trace)
            ]
            assert segments == [f"{trace}-s0", f"{trace}-s1", f"{trace}-s2"]
    assert order[0].subblock_id.startswith(("net-s0-", "cloud-s0-"))
    counts = grid.expected_attempts()
    assert counts["measured"] == 12 * 30 * (len(grid.placements) + 1)
    block = order[0]
    arguments = benchmark_arguments(grid, block, tmp_path / "sb" / "run", resume=False)
    probe = arguments[arguments.index("--adaptive-network-probe") + 1]
    assert probe == str(tmp_path / "sb" / "network_pre.json")
    assert arguments[arguments.index("--adaptive-calibration") + 1].endswith(
        "calibration.json"
    )


def test_trace_grid_rejects_unknown_states(tmp_path: Path) -> None:
    path = _trace_grid(tmp_path)
    payload = json.loads(path.read_text())
    payload["traces"]["bad"] = [["E01", "L7", "C0"]]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown edge load"):
        load_grid(path)


def test_health_response_carries_the_server_state() -> None:
    adapter = FakeModelAdapter()
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint="f" * 64,
        inference_function=adapter.generate,
        tensor_inference_function=adapter.generate_from_payload,
        state_provider=lambda: {"gpu_power_w": 69.5, "sample_age_s": 0.2},
    )
    server = CloudServer(
        host="127.0.0.1", port=0, timeout_seconds=5.0, processor=processor
    )
    try:
        server.start()
    except OSError as exc:  # pragma: no cover - sandbox dependent
        pytest.skip(f"loopback binding unavailable: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config, _ = _setup()
        from dataclasses import replace

        transport = replace(config.transport, host="127.0.0.1", port=server.port)
        health = check_remote_health(
            transport, ModelMetadata(model_id="m", revision="r", fingerprint="f" * 64)
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert health.ready and health.server_state == {
        "gpu_power_w": 69.5,
        "sample_age_s": 0.2,
    }


def test_cli_adaptive_placement_over_real_loopback(tmp_path: Path) -> None:
    """Five fixed placements and the adaptive one, fake model, real loopback."""

    adapter = FakeModelAdapter()
    model_yaml = tmp_path / "fake32.yaml"
    model_yaml.write_text(
        (ROOT / "configs/model/fake.yaml")
        .read_text()
        .replace("max_new_tokens: 8", "max_new_tokens: 32")
    )
    from adaptive_vlm.config import model_contract_fingerprint

    config_text = f"""schema_version: 1
name: g5-fake-loopback
mode: distributed
model_config: {model_yaml.as_posix()}
edge_hardware_config: {(ROOT / "configs/hardware/edge_rtx3050.yaml").as_posix()}
cloud_hardware_config: {(ROOT / "configs/hardware/local_test_cloud.yaml").as_posix()}
dataset:
  manifest_path: {(ROOT / "data/manifests/correctness.jsonl").as_posix()}
  revision: synthetic-fixtures-v1
runtime:
  seed: 42
  warmup_requests: 1
  measured_repetitions: 1
transport:
  host: 127.0.0.1
  port: PORT
  timeout_seconds: 5.0
  protocol_version: 1
artifacts_dir: {tmp_path.as_posix()}
"""
    probe_config = tmp_path / "probe.yaml"
    probe_config.write_text(config_text.replace("PORT", "5050"))
    fingerprint = model_contract_fingerprint(load_experiment_config(probe_config).model)
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=fingerprint,
        inference_function=adapter.generate,
        tensor_inference_function=adapter.generate_from_payload,
        prompt_prepare_function=adapter.prepare_prompt,
        prepared_tensor_inference_function=adapter.generate_from_payload,
        state_provider=lambda: {"gpu_power_w": 70.0, "sample_age_s": 0.1},
    )
    server = CloudServer(
        host="127.0.0.1", port=0, timeout_seconds=5.0, processor=processor
    )
    try:
        server.start()
    except OSError as exc:  # pragma: no cover - sandbox dependent
        pytest.skip(f"loopback binding unavailable: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(_calibration()))
    sensing = tmp_path / "sensing.json"
    sensing.write_text(json.dumps({"schema": "g5-sensing", **SENSING.to_dict()}))
    features = tmp_path / "features.json"
    features.write_text(json.dumps({"task": "short", "tasks": _features()}))
    probe = tmp_path / "network_pre.json"
    probe.write_text(
        json.dumps({"edge_to_cloud_receiver_bps": 5e6, "echo_median_rtt_ms": 150.0})
    )
    try:
        config = tmp_path / "g5.yaml"
        config.write_text(config_text.replace("PORT", str(server.port)))
        split = tmp_path / "split.json"
        split.write_text(json.dumps({"evaluation_request_ids": IDS[:3]}))
        argv = [
            "interleaved-benchmark",
            "--config",
            str(config),
            "--run-dir",
            str(tmp_path / "run"),
            "--split",
            str(split),
            "--split-name",
            "evaluation",
            "--order-seed",
            "11",
            "--repetition-indices",
            "0",
            "--placements",
            *ALL_PLACEMENTS,
            "--adaptive-calibration",
            str(calibration),
            "--adaptive-sensing",
            str(sensing),
            "--adaptive-features",
            str(features),
            "--adaptive-network-probe",
            str(probe),
            "--adaptive-edge-probe",
            "fixed:5.0",
        ]
        assert main(argv) == 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "adaptive" / "requests.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 1 + 3 and all(row["status"] == "ok" for row in rows), rows
    measured = [row for row in rows if not row["warmup"]]
    assert {row["metrics"]["adaptive_sensed_cloud_load"] for row in measured} == {"C2"}
    assert {
        row["metrics"]["adaptive_selected_configuration_id"] for row in measured
    } == {"edge/full"}
    edge = {
        json.loads(line)["trial_id"]: json.loads(line)["output_text"]
        for line in (tmp_path / "run" / "edge" / "requests.jsonl")
        .read_text()
        .splitlines()
    }
    assert all(edge[row["trial_id"]] == row["output_text"] for row in rows)
