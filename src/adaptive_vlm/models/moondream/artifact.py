"""Pinned Moondream2 artifact verification and lazy runtime imports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from adaptive_vlm.config import ModelConfig, Precision
from adaptive_vlm.model_artifact import sha256_file


@dataclass(frozen=True)
class _RuntimeModules:
    torch: Any
    image: Any
    auto_model: Any
    tokenizer: Any
    hf_hub_download: Any
    psutil: Any


def _validate_model_config(config: ModelConfig) -> None:
    if config.adapter != "moondream2":
        raise ValueError("MoondreamAdapter requires adapter 'moondream2'")
    if config.precision is not Precision.BF16:
        raise ValueError("the pinned Moondream baseline requires BF16")
    if config.preprocessing_backend != "pillow":
        raise ValueError("the pinned Moondream baseline requires Pillow")
    if config.do_sample or config.temperature != 0.0:
        raise ValueError("the pinned Moondream baseline requires greedy decoding")
    if config.variant is not None:
        raise ValueError("the baseline does not permit a LoRA variant")


def _load_artifact_manifest(path: str | Path) -> Mapping[str, Any]:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    parsed = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError("model artifact manifest must be a mapping")
    return cast(Mapping[str, Any], parsed)


def _validate_manifest(config: ModelConfig, manifest: Mapping[str, Any]) -> None:
    weights = manifest.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("artifact manifest is missing weights")
    expected = {
        "repository": config.model_id,
        "revision": config.revision,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"artifact manifest {key} does not match model config")
    if weights.get("sha256") != config.artifact_sha256:
        raise ValueError("artifact manifest weight hash does not match model config")


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def _load_runtime_modules() -> _RuntimeModules:
    try:
        import psutil  # type: ignore[import-untyped]
        import torch  # type: ignore[import-not-found]
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
        from tokenizers import Tokenizer  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Moondream dependencies are not installed; use the model lock file"
        ) from exc
    return _RuntimeModules(
        torch=torch,
        image=Image,
        auto_model=AutoModelForCausalLM,
        tokenizer=Tokenizer,
        hf_hub_download=hf_hub_download,
        psutil=psutil,
    )
