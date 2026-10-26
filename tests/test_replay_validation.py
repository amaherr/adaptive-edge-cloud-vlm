"""Frozen adaptive replay validation, including mixed legacy fixture inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from adaptive_vlm.benchmark.adaptive_runner import load_adaptive_replay
from adaptive_vlm.benchmark.calibrate import canonical_sha256
from adaptive_vlm.config import load_experiment_config
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.scheduler.predictor import Calibration

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/adaptive/phase5_local_replay.json"


def _context() -> tuple[dict[str, Any], Calibration]:
    replay = json.loads(SOURCE.read_text(encoding="utf-8"))
    candidates = {
        identifier: SimpleNamespace(
            task_bucket=replay["task_bucket"],
            placement_profile_sha256=canonical_sha256(profile),
        )
        for identifier, profile in replay["placement_profiles"].items()
    }
    calibration = cast(
        Calibration,
        SimpleNamespace(
            calibration_id=replay["calibration_id"],
            source_profile_sha256=replay["source_profile_sha256"],
            host_pair_id=replay["host_pair_id"],
            candidates=candidates,
        ),
    )
    return replay, calibration


def _load(path: Path, calibration: Calibration) -> tuple[dict[str, Any], str]:
    return load_adaptive_replay(
        path,
        manifest=load_frozen_manifest(ROOT / "data/manifests/correctness.jsonl"),
        calibration=calibration,
        config=load_experiment_config(
            ROOT / "configs/experiments/chunk6_local_campaign.yaml"
        ),
        vision_device="cpu",
        text_device="cpu",
        cloud_device="cpu",
    )


def test_existing_mixed_fixture_replay_remains_loadable() -> None:
    _, calibration = _context()
    replay, digest = _load(SOURCE, calibration)
    assert replay["task_bucket"] == "correctness-fixtures-all-prompts-32-token-budget"
    assert digest == hashlib.sha256(SOURCE.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value.update(schema_version=2), "schema"),
        (lambda value: value.update(dataset_manifest_sha256="x"), "dataset"),
        (lambda value: value.update(calibration_id="x"), "calibration"),
        (lambda value: value.update(placement_profiles={}), "placement profiles"),
        (
            lambda value: value["placement_profiles"]["edge/full"].update(
                text_device="cuda:0"
            ),
            "runtime placement",
        ),
        (lambda value: value.update(task_bucket="other"), "task bucket"),
        (lambda value: value.update(task_type="vqa_short_answer"), "task type"),
        (lambda value: value.update(default_network_state=[]), "network state"),
        (
            lambda value: value.update(unhealthy_by_request={"missing": []}),
            "health request IDs",
        ),
        (
            lambda value: value.update(
                unhealthy_by_request={"correctness-001": ["edge/full", "edge/full"]}
            ),
            "unhealthy configuration",
        ),
        (lambda value: value.update(trial_network_states=[]), "network states"),
        (lambda value: value.update(quality_floor=0.5), "quality scores"),
    ],
)
def test_replay_rejects_invalid_predecision_inputs(
    tmp_path: Path, change: Any, message: str
) -> None:
    replay, calibration = _context()
    modified = copy.deepcopy(replay)
    change(modified)
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load(path, calibration)
