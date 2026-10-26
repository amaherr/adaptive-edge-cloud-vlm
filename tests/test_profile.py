from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.predictor_eval import run_predictor_evaluation
from adaptive_vlm.benchmark.profile import build_profile
from adaptive_vlm.benchmark.runner import run_benchmark
from adaptive_vlm.config import ExecutionMode, load_experiment_config
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.scheduler.predictor import Calibration

ROOT = Path(__file__).resolve().parents[1]


class ProfileFixturePolicy(ExecutionPolicy):
    def __init__(self, mode: ExecutionMode) -> None:
        self._mode = mode

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def infer(self, request: InferenceRequest) -> InferenceResult:
        if self.mode is ExecutionMode.DISTRIBUTED and request.request_id.endswith(
            "012"
        ):
            raise MemoryError("fixture OOM")
        model = {
            "prompt_tokens": 12,
            "prompt_prefill_ms": 0.000001,
            "decode_ms": 0.000001,
            "output_tokens": 1,
            "image_width": 16,
            "image_height": 16,
        }
        if self.mode is ExecutionMode.EDGE:
            metrics = {**model, "end_to_end_ms": 0.000002}
        else:
            metrics = {
                **{f"server_model_{name}": value for name, value in model.items()},
                "network_round_trip_ms": 0.000002,
                "server_total_ms": 0.000001,
                "request_wire_bytes": 3_000_000,
                "server_request_wire_bytes": 3_000_000,
                "response_wire_bytes": 30,
            }
            if self.mode is ExecutionMode.CLOUD:
                metrics.update(
                    image_bytes=request.image_path.stat().st_size,
                    image_read_ms=0.000001,
                )
            else:
                metrics.update(
                    projected_payload_bytes=729 * 2048 * 2,
                    edge_vision_end_to_end_ms=0.000001,
                    tensor_serialization_ms=0.000001,
                    image_width=16,
                    image_height=16,
                )
        return InferenceResult(request.request_id, "red", (1,), metrics)


def _campaign(tmp_path: Path) -> tuple[list[Path], Path]:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_local_campaign.yaml"
    )
    frozen = load_frozen_manifest(config.dataset.manifest_path)
    dirs = []
    for mode in ExecutionMode:
        run_dir = tmp_path / mode.value
        run_benchmark(
            config=config,
            manifest=frozen,
            policy=ProfileFixturePolicy(mode),
            run_manifest=create_run_manifest(
                config, {}, dataset_manifest_sha256=frozen.sha256
            ),
            run_dir=run_dir,
            warmups=1,
            repetitions=1,
            pipeline="synchronous",
        )
        dirs.append(run_dir)
    spec = ROOT / "configs/profiles/phase2_local_fixture.json"
    return dirs, spec


def test_profile_validates_and_retains_failures(tmp_path: Path) -> None:
    dirs, spec = _campaign(tmp_path)
    summary = build_profile(
        run_dirs=dirs, specification=spec, output_dir=tmp_path / "profile"
    )
    assert summary["row_count"] == 39
    assert summary["warmup_rows"] == 3
    assert summary["local_gate_passed"]
    assert not summary["controlled_network_gate_passed"]
    assert summary["coverage"]["distributed/projected/synchronous"][
        "status_counts"
    ] == {"ok": 11, "oom": 1}
    rows = [
        json.loads(line)
        for line in (tmp_path / "profile/profile.jsonl").read_text().splitlines()
    ]
    failed = [row for row in rows if row["status"] == "oom"]
    assert len(failed) == 1
    assert failed[0]["null_reasons"]["prompt_tokens"] == "failed_trial"
    cloud = next(
        row
        for row in rows
        if row["configuration_id"] == "cloud/raw_image" and row["status"] == "ok"
    )
    assert cloud["fields"]["request_wire_bytes"] == 3_000_000
    assert len(cloud["run_manifest_sha256"]) == 64
    assert cloud["fields"]["transport_residual_ms"] == pytest.approx(0.000001)
    assert cloud["fields"]["prompt_tokens"] == 12
    assert cloud["task_type"] is None
    assert cloud["null_reasons"]["task_type"] == "not_recorded_in_frozen_manifest"
    split = next(
        row
        for row in rows
        if row["configuration_id"] == "distributed/projected/synchronous"
        and row["status"] == "ok"
    )
    assert split["fields"]["image_width"] == 16
    with pytest.raises(FileExistsError):
        build_profile(
            run_dirs=dirs, specification=spec, output_dir=tmp_path / "profile"
        )


def test_profile_rejects_tampered_wire_bytes(tmp_path: Path) -> None:
    dirs, spec = _campaign(tmp_path)
    path = dirs[1] / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["metrics"]["request_wire_bytes"] = 2
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="wire-byte"):
        build_profile(
            run_dirs=dirs, specification=spec, output_dir=tmp_path / "profile"
        )


def test_profile_rejects_incomplete_schedule(tmp_path: Path) -> None:
    dirs, spec = _campaign(tmp_path)
    path = dirs[0] / "requests.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="missing trials"):
        build_profile(
            run_dirs=dirs, specification=spec, output_dir=tmp_path / "profile"
        )


def test_controlled_network_requires_hashed_evidence(tmp_path: Path) -> None:
    dirs, original = _campaign(tmp_path)
    spec = json.loads(original.read_text())
    spec["network_state"].update(
        controlled=True, uplink_bps=1_000_000, downlink_bps=1_000_000, rtt_ms=5
    )
    path = tmp_path / "controlled.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="measurement evidence hash"):
        build_profile(
            run_dirs=dirs, specification=path, output_dir=tmp_path / "profile"
        )


def test_predictor_fit_holds_out_requests_and_preserves_failures(
    tmp_path: Path,
) -> None:
    dirs, specification = _campaign(tmp_path)
    profile_dir = tmp_path / "profile"
    build_profile(run_dirs=dirs, specification=specification, output_dir=profile_dir)
    profile_path = profile_dir / "profile.jsonl"
    split = {
        "schema_version": 1,
        "campaign_id": "test-predictor",
        "source_profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "training_request_ids": [
            f"correctness-{number:03d}" for number in range(1, 11)
        ],
        "holdout_request_ids": ["correctness-011", "correctness-012"],
        "holdout_network_bucket_ids": [],
        "minimum_training_rows_per_configuration": 10,
        "interval_absolute_error_quantile": 0.9,
        "edge_memory_headroom_fraction": 0.2,
        "target_median_ape": 0.15,
        "target_p90_ape": 0.3,
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    output_dir = tmp_path / "fit"
    report = run_predictor_evaluation(
        profile_path=profile_path, split_path=split_path, output_dir=output_dir
    )
    calibration = json.loads((output_dir / "calibration.json").read_text())
    assert (output_dir / "environment.json").is_file()
    assert (
        Calibration.load(output_dir / "calibration.json").calibration_id
        == calibration["calibration_id"]
    )
    assert set(calibration["training_request_ids"]).isdisjoint(
        calibration["holdout_request_ids"]
    )
    assert all(
        candidate["sample_count"] == 10
        for candidate in calibration["candidates"].values()
    )
    evaluated = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert len(evaluated) == 6
    assert all(
        row["task_state_before_output"]["expected_output_tokens"] is None
        for row in evaluated
        if row["status"] == "ok"
    )
    assert sum(row["status"] != "ok" for row in evaluated) == 1
    assert not report["full_g3_gate_evaluable"]
