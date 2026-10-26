from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.adaptive_analysis import analyze_adaptive_run
from adaptive_vlm.benchmark.adaptive_runner import run_adaptive_benchmark
from adaptive_vlm.benchmark.monitor_eval import analyze_monitor_run
from adaptive_vlm.benchmark.runner import run_benchmark
from adaptive_vlm.config import (
    ExecutionMode,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.remote import RemoteCapabilities
from adaptive_vlm.scheduler.adaptive import RegistrySnapshot, execute_adaptive_request
from adaptive_vlm.scheduler.configuration import (
    CONFIGURATIONS,
    CapabilityRegistry,
    ConfigurationId,
)
from adaptive_vlm.scheduler.predictor import (
    Calibration,
    Prediction,
    PredictionState,
    PredictionTask,
    Predictor,
)
from adaptive_vlm.scheduler.selector import SelectionConstraints
from adaptive_vlm.transport.client import PreInferenceConnectionError

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = "a" * 64


class FakePolicy(ExecutionPolicy):
    def __init__(self, mode: ExecutionMode, *, fail: Exception | None = None) -> None:
        self._mode = mode
        self.fail = fail
        self.calls: list[str] = []

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self.calls.append(request.request_id)
        if self.fail is not None:
            error = self.fail
            self.fail = None
            raise error
        metrics: dict[str, int] = {}
        if self.mode is not ExecutionMode.EDGE:
            metrics = {"request_wire_bytes": 100, "response_wire_bytes": 20}
        return InferenceResult(request.request_id, "same", (1, 2), metrics)


def _prediction(configuration_id: str, total: float) -> Prediction:
    return Prediction(
        configuration_id,
        True,
        None,
        "in_distribution",
        "fit-1",
        2.0,
        "training_median",
        {"compute_ms": total},
        total,
        total,
        total,
        100.0,
        100.0,
        20.0,
        None,
    )


class FakePredictor:
    def __init__(self, _calibration: object = None) -> None:
        pass

    def predict(self, spec: Any, _state: Any, _task: Any) -> Prediction:
        totals = {
            ConfigurationId.EDGE_FULL: 20.0,
            ConfigurationId.CLOUD_RAW_IMAGE: 10.0,
            ConfigurationId.DISTRIBUTED_PROJECTED: 30.0,
        }
        return _prediction(spec.id.value, totals[spec.id])


def _policies(
    *, cloud_fail: Exception | None = None
) -> dict[ConfigurationId, FakePolicy]:
    return {
        spec.id: FakePolicy(
            spec.mode,
            fail=cloud_fail if spec.id is ConfigurationId.CLOUD_RAW_IMAGE else None,
        )
        for spec in CONFIGURATIONS
    }


def _registry(
    policies: dict[ConfigurationId, FakePolicy], fingerprint: str = FINGERPRINT
) -> CapabilityRegistry:
    remote = RemoteCapabilities(
        True, fingerprint, frozenset({"raw_image", "projected_synchronous"})
    )
    return CapabilityRegistry(
        policies=policies, expected_model_fingerprint=fingerprint, remote=remote
    )


def _state() -> PredictionState:
    return PredictionState("host", FINGERPRINT, "placement", "loopback", "replay")


def _task() -> PredictionTask:
    return PredictionTask("fixture", 200, 4, 4, 11, 32)


def _execute(
    policies: dict[ConfigurationId, FakePolicy],
    *,
    healthy: dict[ConfigurationId, bool] | None = None,
) -> Any:
    image = ROOT / "tests/fixtures/moondream_smoke.ppm"
    return execute_adaptive_request(
        request=InferenceRequest("case", image, "describe"),
        task=_task(),
        state=_state(),
        placement_hashes={spec.id: "placement" for spec in CONFIGURATIONS},
        healthy=healthy or {spec.id: True for spec in CONFIGURATIONS},
        constraints=SelectionConstraints(),
        predictor=cast(Predictor, FakePredictor()),
        registry_provider=lambda: _registry(policies),
    )


def test_safe_retry_reselects_once_and_logs_both_attempts() -> None:
    policies = _policies(cloud_fail=PreInferenceConnectionError("refused"))
    outcome = _execute(policies)
    assert outcome.result is not None
    assert outcome.selected_configuration_id is ConfigurationId.EDGE_FULL
    assert [attempt["status"] for attempt in outcome.attempts] == ["failed", "ok"]
    assert outcome.attempts[0]["retry_reason"] == "connection_failed_before_send"
    assert (
        outcome.attempts[1]["decision"]["candidates"][1]["exclusion_reason"]
        == "unhealthy"
    )
    assert outcome.retry_overhead_ms >= 0
    assert len(policies[ConfigurationId.CLOUD_RAW_IMAGE].calls) == 1


def test_unknown_stage_failure_does_not_retry() -> None:
    policies = _policies(cloud_fail=RuntimeError("after send"))
    outcome = _execute(policies)
    assert outcome.result is None
    assert len(outcome.attempts) == 1
    assert not policies[ConfigurationId.EDGE_FULL].calls


def test_no_feasible_candidate_is_explicit() -> None:
    policies = _policies()
    outcome = _execute(policies, healthy={spec.id: False for spec in CONFIGURATIONS})
    assert outcome.result is None
    assert outcome.attempts[0]["status"] == "no_feasible_candidate"
    assert not any(policy.calls for policy in policies.values())


def test_live_monitor_failure_excludes_remote_and_unknown_memory() -> None:
    policies = _policies()
    registry = CapabilityRegistry(
        policies=policies,
        expected_model_fingerprint=FINGERPRINT,
        remote=None,
        remote_error="health_probe_failed",
        edge_memory_budget_mib=500.0,
        edge_required_peak_mib=None,
    )
    monitoring = {
        "sampled_at_utc": "2026-10-26T00:00:00Z",
        "age_ms": 0.0,
        "sampling_overhead_ms": 0.1,
        "health_ready": None,
        "remote_ready": False,
        "error": "health_probe_failed",
    }
    outcome = execute_adaptive_request(
        request=InferenceRequest(
            "case", ROOT / "tests/fixtures/moondream_smoke.ppm", "describe"
        ),
        task=_task(),
        state=_state(),
        placement_hashes={spec.id: "placement" for spec in CONFIGURATIONS},
        healthy={spec.id: True for spec in CONFIGURATIONS},
        constraints=SelectionConstraints(edge_memory_budget_mib=500.0),
        predictor=cast(Predictor, FakePredictor()),
        registry_provider=lambda: RegistrySnapshot(registry, monitoring),
    )
    assert outcome.result is None
    assert outcome.attempts[0]["status"] == "no_feasible_candidate"
    assert outcome.monitor_overhead_ms == 0.1
    assert outcome.attempts[0]["monitoring"] == monitoring
    assert not any(policy.calls for policy in policies.values())


def test_mixed_run_matches_fixed_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_local_campaign.yaml"
    )
    manifest = load_frozen_manifest(ROOT / "data/manifests/correctness.jsonl")
    fingerprint = model_contract_fingerprint(config.model)
    policies = _policies()
    calibration = cast(
        Calibration, SimpleNamespace(calibration_id="fit-1", host_pair_id="host")
    )
    monkeypatch.setattr(
        "adaptive_vlm.benchmark.adaptive_runner.Predictor", FakePredictor
    )
    placement = {spec.id.value: {"mode": spec.mode.value} for spec in CONFIGURATIONS}
    ids = [item.request_id for item in manifest.items]
    replay = {
        "task_bucket": "fixture",
        "model_contract_sha256": fingerprint,
        "placement_profiles": placement,
        "default_network_state": {"bucket_id": "loopback", "source": "replay"},
        "unhealthy_by_request": {
            request_id: (
                [
                    ConfigurationId.EDGE_FULL.value,
                    ConfigurationId.DISTRIBUTED_PROJECTED.value,
                ]
                if index < 4
                else [
                    ConfigurationId.CLOUD_RAW_IMAGE.value,
                    ConfigurationId.DISTRIBUTED_PROJECTED.value,
                ]
                if index < 8
                else [
                    ConfigurationId.EDGE_FULL.value,
                    ConfigurationId.CLOUD_RAW_IMAGE.value,
                ]
            )
            for index, request_id in enumerate(ids)
        },
    }
    placement_environment = {
        "execution_placement": {
            "vision_device": "cpu",
            "text_device": "cpu",
            "cloud_device": "cpu",
            "remote": "127.0.0.1:5050",
        }
    }
    fixed_dirs: list[Path] = []
    for spec in CONFIGURATIONS:
        directory = tmp_path / spec.mode.value
        run_benchmark(
            config=config,
            manifest=manifest,
            policy=policies[spec.id],
            run_manifest=create_run_manifest(
                config, placement_environment, dataset_manifest_sha256=manifest.sha256
            ),
            run_dir=directory,
            warmups=1,
            repetitions=1,
            pipeline="synchronous",
        )
        fixed_dirs.append(directory)
    adaptive_dir = tmp_path / "adaptive"
    summary = run_adaptive_benchmark(
        config=config,
        manifest=manifest,
        calibration=calibration,
        replay=replay,
        replay_sha256="b" * 64,
        registry_provider=lambda: _registry(policies, fingerprint),
        task_builder=lambda _item: _task(),
        run_manifest=create_run_manifest(
            config, placement_environment, dataset_manifest_sha256=manifest.sha256
        ),
        run_dir=adaptive_dir,
        warmups=1,
        repetitions=1,
        activation="diagnostic_unpassed_gates",
    )
    assert summary["status_counts"] == {"ok": 12}
    assert summary["selected_counts"] == {
        "cloud/raw_image": 4,
        "edge/full": 4,
        "distributed/projected/synchronous": 4,
    }
    report = analyze_adaptive_run(adaptive_dir, fixed_dirs)
    assert report["g5_local_gate_passed"]
    assert not report["decision_trace_errors"]
    assert not report["same_placement_token_mismatches"]
    resumed = run_adaptive_benchmark(
        config=config,
        manifest=manifest,
        calibration=calibration,
        replay=replay,
        replay_sha256="b" * 64,
        registry_provider=lambda: _registry(policies, fingerprint),
        task_builder=lambda _item: _task(),
        run_manifest=create_run_manifest(
            config, placement_environment, dataset_manifest_sha256=manifest.sha256
        ),
        run_dir=adaptive_dir,
        warmups=1,
        repetitions=1,
        activation="diagnostic_unpassed_gates",
        resume=True,
    )
    assert resumed["measured_attempts"] == 12
    live_dir = tmp_path / "live"
    live_monitoring = {
        "sampled_at_utc": "2026-10-26T00:00:00Z",
        "age_ms": 0.0,
        "sampling_overhead_ms": 0.0001,
        "health_ready": True,
        "remote_ready": True,
        "error": None,
        "edge_memory_budget_mib": None,
        "edge_required_peak_mib": None,
    }
    run_adaptive_benchmark(
        config=config,
        manifest=manifest,
        calibration=calibration,
        replay=replay,
        replay_sha256="b" * 64,
        registry_provider=lambda: RegistrySnapshot(
            _registry(policies, fingerprint), live_monitoring
        ),
        task_builder=lambda _item: _task(),
        run_manifest=create_run_manifest(
            config, placement_environment, dataset_manifest_sha256=manifest.sha256
        ),
        run_dir=live_dir,
        warmups=1,
        repetitions=1,
        activation="diagnostic_unpassed_gates",
        monitor_config={"mode": "live", "max_age_ms": 100.0},
        passive_observer=lambda metrics: "request_wire_bytes" in metrics,
    )
    monitor_report = analyze_monitor_run(live_dir)
    assert monitor_report["local_live_monitor_gate_passed"]
    assert monitor_report["monitor_samples_recorded"] == 13
    assert monitor_report["passive_updates"] == 8
    assert analyze_adaptive_run(live_dir, fixed_dirs)["g5_local_gate_passed"]
    raw = adaptive_dir / "requests.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    selected = rows[1]["selected_configuration_id"]
    for candidate in rows[1]["attempts"][-1]["decision"]["candidates"]:
        if candidate["configuration_id"] == selected:
            candidate["eligible"] = False
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tampered = analyze_adaptive_run(adaptive_dir, fixed_dirs)
    assert not tampered["g5_local_gate_passed"]
    assert rows[1]["trial_id"] in tampered["decision_trace_errors"]
