from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from adaptive_vlm.config import load_experiment_config
from adaptive_vlm.contracts import InferenceRequest
from adaptive_vlm.models.moondream.adapter import MoondreamAdapter
from adaptive_vlm.models.moondream.artifact import (
    _load_artifact_manifest,
    _validate_manifest,
    _validate_model_config,
    _verify_file,
)
from adaptive_vlm.models.moondream.residency import (
    _StageSwapper,
    _validate_weight_residency,
)

ROOT = Path(__file__).resolve().parents[2]
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
    adapter._swapper = None
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


class _Param:
    def __init__(self, data: str) -> None:
        self.data: Any = data


class _SwapHarness:
    def __init__(self, fail_on: str | None = None) -> None:
        self.vision = [_Param("v1"), _Param("v2")]
        self.text = [_Param("t1")]
        self.loads: list[str] = []
        self.syncs = 0
        self.contexts = 0
        self.fail_on = fail_on
        self.swapper = _StageSwapper(
            {"vision": self.vision, "text": self.text},
            load=self.load,
            synchronize=self.synchronize,
            context=self.context,
        )

    def load(self, master: str) -> str:
        if master == self.fail_on:
            raise RuntimeError("simulated device OOM")
        self.loads.append(master)
        return f"device:{master}"

    def synchronize(self) -> None:
        self.syncs += 1

    def context(self) -> Any:
        harness = self

        class _Context:
            def __enter__(self) -> None:
                harness.contexts += 1

            def __exit__(self, *_: object) -> None:
                return None

        return _Context()

    def data(self, stage: str) -> list[str]:
        return [p.data for p in getattr(self, stage)]


def test_swapper_keeps_exactly_one_stage_on_device() -> None:
    harness = _SwapHarness()
    assert harness.swapper.active_stage is None

    assert harness.swapper.activate("vision") >= 0
    assert harness.data("vision") == ["device:v1", "device:v2"]
    assert harness.data("text") == ["t1"]

    harness.swapper.activate("text")
    assert harness.swapper.active_stage == "text"
    assert harness.data("vision") == ["v1", "v2"]
    assert harness.data("text") == ["device:t1"]
    assert harness.loads == ["v1", "v2", "t1"]
    assert harness.syncs == 2
    assert harness.contexts == 2


def test_swapper_reactivation_is_free() -> None:
    harness = _SwapHarness()
    harness.swapper.activate("text")

    assert harness.swapper.activate("text") == 0.0
    assert harness.loads == ["t1"]
    assert harness.syncs == 1


def test_swapper_restores_masters_when_a_copy_fails() -> None:
    harness = _SwapHarness(fail_on="v2")
    harness.swapper.activate("text")

    with pytest.raises(RuntimeError, match="simulated device OOM"):
        harness.swapper.activate("vision")

    assert harness.swapper.active_stage is None
    assert harness.data("vision") == ["v1", "v2"]
    assert harness.data("text") == ["t1"]


def test_swapper_rejects_unknown_or_missing_stages() -> None:
    harness = _SwapHarness()
    with pytest.raises(ValueError, match="unknown swap stage"):
        harness.swapper.activate("region")
    with pytest.raises(ValueError, match="exactly"):
        _StageSwapper({"vision": []}, load=str, synchronize=lambda: None)


@pytest.mark.parametrize(
    ("components", "vision", "text", "message"),
    [
        ("vision", "cuda:0", "cuda:0", "full model"),
        ("full", "cuda:0", "cpu", "shared"),
        ("full", "cpu", "cpu", "CUDA"),
    ],
)
def test_swap_residency_validation(
    components: str, vision: str, text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_weight_residency("swap", components, vision, text)


def test_residency_validation_accepts_resident_and_rejects_unknown() -> None:
    _validate_weight_residency("resident", "vision", "cpu", "cuda:0")
    _validate_weight_residency("swap", "full", "cuda:0", "cuda:0")
    with pytest.raises(ValueError, match="unsupported weight residency"):
        _validate_weight_residency("partial", "full", "cuda:0", "cuda:0")


def test_from_pretrained_rejects_invalid_swap_before_loading(tmp_path: Path) -> None:
    config = load_experiment_config(MODEL_CONFIG).model

    with pytest.raises(ValueError, match="full model"):
        MoondreamAdapter.from_pretrained(
            config,
            cache_dir=tmp_path,
            artifact_manifest=ARTIFACT_MANIFEST,
            vision_device="cuda:0",
            text_device="cuda:0",
            components="vision",
            weight_residency="swap",
        )
    assert not any(tmp_path.iterdir())


def _swap_adapter() -> tuple[MoondreamAdapter, _SwapHarness]:
    adapter, _, _ = _component_adapter()
    harness = _SwapHarness()
    adapter._swapper = harness.swapper
    adapter._text_device = "cuda:0"
    return adapter, harness


def test_swap_metrics_only_appear_under_swap_residency() -> None:
    resident, _, _ = _component_adapter()
    assert resident.weight_residency == "resident"
    assert resident._swap_metrics("swap_to_text_ms", "text") == {}

    adapter, harness = _swap_adapter()
    metrics = adapter._swap_metrics("swap_to_vision_ms", "vision")
    assert adapter.weight_residency == "swap"
    assert metrics["weight_residency"] == "swap"
    assert isinstance(metrics["swap_to_vision_ms"], float)
    assert harness.swapper.active_stage == "vision"


def test_swap_adapter_refuses_monolithic_execution_and_pruning() -> None:
    adapter, _ = _swap_adapter()
    request = InferenceRequest("r1", ROOT / "tests/fixtures/moondream_smoke.ppm", "q")

    with pytest.raises(RuntimeError, match="both stages resident"):
        adapter.run_monolithic(request)
    with pytest.raises(RuntimeError, match="cannot be pruned"):
        adapter.retain_only("vision")
