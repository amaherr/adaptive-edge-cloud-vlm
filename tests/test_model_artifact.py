from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from adaptive_vlm.model_artifact import classify_tensor_name, sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "component"),
    [
        ("model.vision.blocks.0.weight", "vision_encoder"),
        ("model.vision.proj_mlp.fc1.weight", "projector"),
        ("model.text.blocks.0.weight", "language_model"),
        ("model.region.coord.weight", "region_model"),
        ("unexpected.weight", "other"),
    ],
)
def test_classify_tensor_name(name: str, component: str) -> None:
    assert classify_tensor_name(name) == component


def test_sha256_file_streams_content(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abcdefgh")

    expected = hashlib.sha256(b"abcdefgh").hexdigest()
    assert sha256_file(artifact, chunk_size=3) == expected


def test_sha256_file_rejects_invalid_chunk_size(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"content")

    with pytest.raises(ValueError, match="positive"):
        sha256_file(artifact, chunk_size=0)


def test_model_config_matches_artifact_manifest() -> None:
    model = yaml.safe_load((ROOT / "configs/model/moondream2.yaml").read_text())
    manifest = yaml.safe_load(
        (ROOT / "configs/model/moondream2_artifact.yaml").read_text()
    )

    assert model["model_id"] == manifest["repository"]
    assert model["release_tag"] == manifest["release_tag"]
    assert model["revision"] == manifest["revision"]
    assert model["artifact_sha256"] == manifest["weights"]["sha256"]


def test_artifact_manifest_matches_dependency_lock() -> None:
    manifest = yaml.safe_load(
        (ROOT / "configs/model/moondream2_artifact.yaml").read_text()
    )
    lock_path = ROOT / manifest["dependency_lock"]["path"]

    assert sha256_file(lock_path) == manifest["dependency_lock"]["sha256"]


def test_artifact_component_counts_sum_to_total() -> None:
    manifest = yaml.safe_load(
        (ROOT / "configs/model/moondream2_artifact.yaml").read_text()
    )

    assert (
        sum(manifest["weights"]["component_parameter_counts"].values())
        == manifest["weights"]["parameter_count"]
    )
