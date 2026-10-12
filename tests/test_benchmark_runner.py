from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.runner import make_schedule, run_benchmark
from adaptive_vlm.config import ExecutionMode, load_experiment_config
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.base import ExecutionPolicy

ROOT = Path(__file__).resolve().parents[1]


class IntermittentPolicy(ExecutionPolicy):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.EDGE

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self.calls += 1
        if self.calls == 2:
            raise MemoryError("test OOM")
        return InferenceResult(request.request_id, "red", (1,), {"inference_ms": 1.0})


def _manifest(tmp_path: Path) -> Path:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    path = tmp_path / "requests.jsonl"
    rows = [
        {
            "request_id": f"request-{index}",
            "image_path": image.name,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "prompt": "What color?",
            "references": ["red"],
            "scoring": "exact_match",
            "source": "test",
            "revision": "synthetic-fixtures-v1",
        }
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_schedule_is_seeded_and_balanced(tmp_path: Path) -> None:
    manifest = load_frozen_manifest(_manifest(tmp_path))
    schedule = make_schedule(manifest, warmups=1, repetitions=3, seed=42)
    assert schedule == make_schedule(manifest, warmups=1, repetitions=3, seed=42)
    assert len(schedule) == 7
    assert all(
        {trial.request_id for trial in schedule if trial.repetition == repetition}
        == {"request-0", "request-1"}
        for repetition in range(3)
    )


def test_runner_records_failures_and_resumes_without_duplicate_trials(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"
    )
    manifest = load_frozen_manifest(_manifest(tmp_path))
    policy = IntermittentPolicy()
    run_dir = tmp_path / "run"
    kwargs = dict(
        config=config,
        manifest=manifest,
        policy=policy,
        run_manifest=create_run_manifest(
            config, {}, dataset_manifest_sha256=manifest.sha256
        ),
        run_dir=run_dir,
        warmups=1,
        repetitions=2,
        pipeline="synchronous",
    )
    summary = run_benchmark(**kwargs)
    assert summary["measured_attempts"] == 4
    assert summary["warmup_attempts"] == 1
    assert summary["status_counts"] == {"oom": 1, "ok": 3}
    assert summary["failure_rate"] == 0.25
    assert summary["latency_ms_p95_success"] is not None
    assert summary["latency_ms_iqr_success"] is not None
    assert summary["quality_mean_success"] == 1.0
    assert summary["quality_scored_attempts"] == 3
    assert policy.calls == 5
    resumed = run_benchmark(**kwargs, resume=True)
    assert resumed == summary
    assert policy.calls == 5
    raw_lines = (run_dir / "requests.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in raw_lines]
    assert len(rows) == 5
    assert len({row["trial_id"] for row in rows}) == 5
    assert all(row["run_id"] == rows[0]["run_id"] for row in rows)
    with pytest.raises(FileExistsError):
        run_benchmark(**kwargs)


def test_resume_rejects_mismatched_plan(tmp_path: Path) -> None:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"
    )
    manifest = load_frozen_manifest(_manifest(tmp_path))
    kwargs = dict(
        config=config,
        manifest=manifest,
        policy=IntermittentPolicy(),
        run_manifest=create_run_manifest(config, {}),
        run_dir=tmp_path / "run",
        warmups=0,
        repetitions=1,
        pipeline="synchronous",
    )
    run_benchmark(**kwargs)
    with pytest.raises(ValueError, match="plan"):
        run_benchmark(**{**kwargs, "repetitions": 2}, resume=True)
