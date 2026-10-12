from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.compare import compare_runs
from adaptive_vlm.benchmark.runner import run_benchmark, semantic_config_sha256
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
    assert report["common_measured_pairs"] == len(manifest.items)
    assert len(report["mismatches"]) == len(manifest.items)
    assert report["mismatches"][0]["first_differing_token_index"] == {
        "cloud": 0
    }
    assert report["exact_token_matches"] == 0
    assert report["text_mismatches"] == []
    assert report["quality_by_mode"]["edge"]["quality_scored_attempts"] > 0
    assert report["complete_and_equal"] is False
    assert json.loads(report_path.read_text()) == report


def test_semantic_config_hash_ignores_only_file_locations() -> None:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"
    ).to_dict()
    alternate = json.loads(json.dumps(config))
    alternate["source_path"] = "/another/path/config.yaml"
    alternate["artifacts_dir"] = "/another/path/artifacts"
    alternate["dataset"]["manifest_path"] = "/another/path/manifest.jsonl"
    assert semantic_config_sha256(config) == semantic_config_sha256(alternate)
    alternate["model"]["revision"] = "other-revision"
    assert semantic_config_sha256(config) != semantic_config_sha256(alternate)


def test_compare_three_identical_policies(tmp_path: Path) -> None:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"
    )
    manifest = load_frozen_manifest(config.dataset.manifest_path)
    directories = []
    for mode in ExecutionMode:
        directory = tmp_path / mode.value
        run_benchmark(
            config=config, manifest=manifest, policy=StaticPolicy(mode, 7),
            run_manifest=create_run_manifest(
                config, {}, dataset_manifest_sha256=manifest.sha256
            ),
            run_dir=directory, warmups=1, repetitions=2,
            pipeline="synchronous",
        )
        directories.append(directory)
    report = compare_runs(directories)
    assert report["complete_and_equal"] is True
    assert report["exact_token_matches"] == 2 * len(manifest.items)
    assert report["mismatches"] == []


def test_compare_rejects_tampered_result_hash(tmp_path: Path) -> None:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"
    )
    manifest = load_frozen_manifest(config.dataset.manifest_path)
    directories = []
    for mode in (ExecutionMode.EDGE, ExecutionMode.CLOUD):
        directory = tmp_path / mode.value
        run_benchmark(
            config=config, manifest=manifest, policy=StaticPolicy(mode, 7),
            run_manifest=create_run_manifest(
                config, {}, dataset_manifest_sha256=manifest.sha256
            ),
            run_dir=directory, warmups=0, repetitions=1,
            pipeline="synchronous",
        )
        directories.append(directory)
    raw_path = directories[0] / "requests.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    rows[0]["model_contract_sha256"] = "0" * 64
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="raw results do not match plan"):
        compare_runs(directories)
