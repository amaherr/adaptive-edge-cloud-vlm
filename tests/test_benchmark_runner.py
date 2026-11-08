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
    # A process interrupted before its last append can resume just that trial.
    raw_path = run_dir / "requests.jsonl"
    raw_path.write_text("\n".join(raw_lines[:-1]) + "\n", encoding="utf-8")
    recovered = run_benchmark(**kwargs, resume=True)
    assert recovered["measured_attempts"] == 4
    assert policy.calls == 6
    recovered_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert len(recovered_rows) == 5
    assert all(row["run_id"] == rows[0]["run_id"] for row in recovered_rows)
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
        run_manifest=create_run_manifest(
            config, {}, dataset_manifest_sha256=manifest.sha256
        ),
        run_dir=tmp_path / "run",
        warmups=0,
        repetitions=1,
        pipeline="synchronous",
    )
    run_benchmark(**kwargs)
    with pytest.raises(ValueError, match="plan"):
        run_benchmark(**{**kwargs, "repetitions": 2}, resume=True)


def test_resume_recovers_torn_append_but_rejects_committed_corruption(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"
    )
    manifest = load_frozen_manifest(_manifest(tmp_path))
    policy = IntermittentPolicy()
    kwargs = dict(
        config=config,
        manifest=manifest,
        policy=policy,
        run_manifest=create_run_manifest(
            config, {}, dataset_manifest_sha256=manifest.sha256
        ),
        run_dir=tmp_path / "run",
        warmups=0,
        repetitions=1,
        pipeline="synchronous",
    )
    run_benchmark(**kwargs)
    path = tmp_path / "run/requests.jsonl"
    rows = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(rows[0] + rows[1][:30])
    run_benchmark(**kwargs, resume=True)
    assert policy.calls == 3
    assert len(path.read_text().splitlines()) == 2
    backups = list(path.parent.glob("requests.torn-*.bin"))
    assert len(backups) == 1 and backups[0].read_bytes() == rows[1][:30]
    path.write_bytes(rows[0] + b"broken\n")
    with pytest.raises(ValueError, match="invalid raw result line"):
        run_benchmark(**kwargs, resume=True)


def test_failure_circuit_breaker_saves_attempt_before_stopping(tmp_path: Path) -> None:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"
    )
    manifest = load_frozen_manifest(_manifest(tmp_path))
    policy = IntermittentPolicy()
    kwargs = dict(
        config=config,
        manifest=manifest,
        policy=policy,
        run_manifest=create_run_manifest(
            config, {}, dataset_manifest_sha256=manifest.sha256
        ),
        run_dir=tmp_path / "run",
        warmups=0,
        repetitions=2,
        pipeline="synchronous",
        max_consecutive_failures=1,
    )
    with pytest.raises(RuntimeError, match="attempts are saved"):
        run_benchmark(**kwargs)
    path = tmp_path / "run/requests.jsonl"
    assert len(path.read_text().splitlines()) == 2
    summary = run_benchmark(**kwargs, resume=True)
    assert policy.calls == 4
    assert summary["status_counts"] == {"ok": 3, "oom": 1}


def test_sigkill_after_durable_attempt_resumes_without_duplicate(
    tmp_path: Path,
) -> None:
    import os
    import signal
    import subprocess
    import sys

    manifest_path = _manifest(tmp_path)
    script = tmp_path / "crash.py"
    script.write_text("""
import os
import signal
import sys
from pathlib import Path
from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark import runner
from adaptive_vlm.config import load_experiment_config
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.models.fake import FakeModelAdapter

config = load_experiment_config(sys.argv[1])
manifest = load_frozen_manifest(sys.argv[2])
original_append = runner.append_jsonl
if sys.argv[4] == "crash":
    def append_and_die(path, row):
        original_append(path, row)
        os.kill(os.getpid(), signal.SIGKILL)
    runner.append_jsonl = append_and_die
runner.run_benchmark(
    config=config, manifest=manifest,
    policy=LocalExecutionPolicy(FakeModelAdapter()),
    run_manifest=create_run_manifest(config, {},
                                    dataset_manifest_sha256=manifest.sha256),
    run_dir=Path(sys.argv[3]), warmups=1, repetitions=3,
    pipeline="synchronous", resume=sys.argv[4] == "resume",
)
""")
    directory = tmp_path / "killed-run"
    command = [
        sys.executable,
        str(script),
        str(ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"),
        str(manifest_path),
        str(directory),
    ]
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [*command, "crash"], env=environment, capture_output=True, timeout=30
    )
    assert result.returncode == -signal.SIGKILL
    path = directory / "requests.jsonl"
    first = path.read_bytes()
    assert len(first.splitlines()) == 1
    result = subprocess.run(
        [*command, "resume"], env=environment, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr.decode()
    assert path.read_bytes().startswith(first)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 7
    assert len({row["trial_id"] for row in rows}) == 7
