from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from adaptive_vlm.config import load_experiment_config
from adaptive_vlm.models.moondream import (
    _load_artifact_manifest,
    _validate_manifest,
    _validate_model_config,
    _verify_file,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs/experiments/baseline_edge.yaml"
ARTIFACT_MANIFEST = ROOT / "configs/model/moondream2_artifact.yaml"


def test_pinned_model_and_manifest_are_adapter_compatible() -> None:
    config = load_experiment_config(MODEL_CONFIG).model
    manifest = _load_artifact_manifest(ARTIFACT_MANIFEST)

    _validate_model_config(config)
    _validate_manifest(config, manifest)


def test_adapter_rejects_lora_variant() -> None:
    config = load_experiment_config(MODEL_CONFIG).model

    with pytest.raises(ValueError, match="LoRA"):
        _validate_model_config(replace(config, variant="some-variant"))


def test_adapter_rejects_manifest_revision_mismatch(tmp_path: Path) -> None:
    config = load_experiment_config(MODEL_CONFIG).model
    manifest = yaml.safe_load(ARTIFACT_MANIFEST.read_text())
    manifest["revision"] = "different"
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="revision"):
        _validate_manifest(config, _load_artifact_manifest(path))


def test_verify_file_rejects_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verify_file(artifact, "0" * 64, "test artifact")
