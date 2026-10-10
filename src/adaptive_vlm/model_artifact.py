"""Utilities for fingerprinting and inspecting model artifacts."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the file into RAM."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def classify_tensor_name(name: str) -> str:
    """Map Moondream tensor names to high-level components."""

    if name.startswith("model.vision.proj_mlp."):
        return "projector"
    if name.startswith("model.vision."):
        return "vision_encoder"
    if name.startswith("model.text."):
        return "language_model"
    if name.startswith("model.region."):
        return "region_model"
    return "other"


def inspect_safetensors(path: str | Path) -> dict[str, Any]:
    """Inspect safetensors metadata and hash the artifact.

    `safetensors` is imported lazily so the CPU-only project foundation does
    not require model dependencies.
    """

    from safetensors import safe_open  # type: ignore[import-not-found]

    artifact_path = Path(path).resolve(strict=True)
    tensor_count = 0
    parameter_count = 0
    dtype_counts: dict[str, int] = {}
    component_counts: dict[str, int] = {}

    with safe_open(artifact_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():  # noqa: SIM118 - safe_open is not a mapping
            tensor = handle.get_slice(name)
            count = math.prod(tensor.get_shape())
            dtype = str(tensor.get_dtype())
            component = classify_tensor_name(name)
            tensor_count += 1
            parameter_count += count
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
            component_counts[component] = component_counts.get(component, 0) + count

    return {
        "path": str(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
        "sha256": sha256_file(artifact_path),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "dtype_parameter_counts": dtype_counts,
        "component_parameter_counts": component_counts,
    }
