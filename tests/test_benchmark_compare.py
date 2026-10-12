from __future__ import annotations

import json
from pathlib import Path

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.compare import compare_runs
from adaptive_vlm.benchmark.runner import run_benchmark
from adaptive_vlm.config import ExecutionMode, load_experiment_config
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.base import ExecutionPolicy

ROOT = Path(__file__).resolve().parents[1]


class StaticPolicy(ExecutionPolicy):
    def __init__(self, mode: ExecutionMode, token: int) -> None:
        self._mode = mode
        self._token = token

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def infer(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(request.request_id, "same", (self._token,))


def test_compare_reports_exact_mismatch(tmp_path: Path) -> None:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"
    )
    manifest = load_frozen_manifest(config.dataset.manifest_path)
    directories = []
    for mode, token in ((ExecutionMode.EDGE, 1), (ExecutionMode.CLOUD, 2)):
        directory = tmp_path / mode.value
        run_benchmark(
            config=config, manifest=manifest, policy=StaticPolicy(mode, token),
            run_manifest=create_run_manifest(
                config, {}, dataset_manifest_sha256=manifest.sha256
            ),
            run_dir=directory, warmups=0, repetitions=1,
            pipeline="synchronous",
        )
        directories.append(directory)
    report_path = tmp_path / "comparison.json"
    report = compare_runs(directories, output=report_path)
    assert report["common_first_repetition_requests"] == len(manifest.items)
    assert len(report["mismatches"]) == len(manifest.items)
    assert report["exact_token_matches"] == 0
    assert report["complete_and_equal"] is False
    assert json.loads(report_path.read_text()) == report
