from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from adaptive_vlm.config import load_experiment_config
from adaptive_vlm.models.moondream import (
    MoondreamAdapter,
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


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


def _component_adapter() -> tuple[MoondreamAdapter, Any, _FakeCuda]:
    adapter = object.__new__(MoondreamAdapter)
    core = SimpleNamespace(
        vision=object(),
        text=object(),
        region=object(),
        attn_mask=object(),
        tokenizer=object(),
        config=object(),
    )
    cuda = _FakeCuda()
    adapter._core = core
    adapter._runtime = SimpleNamespace(torch=SimpleNamespace(cuda=cuda))
    adapter._vision_device = "cuda:0"
    adapter._text_device = "cpu"
    adapter._component_role = "full"
    return adapter, core, cuda


def test_vision_role_releases_text_components() -> None:
    adapter, core, cuda = _component_adapter()

    adapter._retain_components("vision")

    assert adapter.component_role == "vision"
    assert hasattr(core, "vision")
    assert not hasattr(core, "text")
    assert not hasattr(core, "region")
    assert not hasattr(core, "attn_mask")
    assert adapter._active_devices() == ("cuda:0",)
    assert cuda.empty_cache_calls == 1


def test_text_role_releases_vision_components() -> None:
    adapter, core, cuda = _component_adapter()

    adapter._retain_components("text")

    assert adapter.component_role == "text"
    assert not hasattr(core, "vision")
    assert hasattr(core, "text")
    assert not hasattr(core, "region")
    assert hasattr(core, "attn_mask")
    assert adapter._active_devices() == ("cpu",)
    assert cuda.empty_cache_calls == 1


def test_component_role_prevents_wrong_stage_execution() -> None:
    adapter, _, _ = _component_adapter()
    adapter._retain_components("vision")

    with pytest.raises(RuntimeError, match="requires text"):
        adapter._require_component("text", "full")


def test_retain_only_is_irreversible() -> None:
    adapter, _, _ = _component_adapter()
    adapter.retain_only("vision")

    with pytest.raises(RuntimeError, match="only be pruned from a full"):
        adapter.retain_only("text")
