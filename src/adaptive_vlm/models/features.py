"""Contracts for the projected visual representation at the split boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from adaptive_vlm.contracts import MetricValue

MOONDREAM_VISUAL_TOKENS = 729
MOONDREAM_HIDDEN_SIZE = 2048


@dataclass(frozen=True)
class ProjectedVisionFeatures:
    """A validated tensor produced after Moondream's vision projector."""

    tensor: Any
    shape: tuple[int, int]
    dtype: str
    device: str
    nbytes: int

    @classmethod
    def from_tensor(cls, tensor: Any) -> ProjectedVisionFeatures:
        shape = tuple(int(dimension) for dimension in tensor.shape)
        expected = (MOONDREAM_VISUAL_TOKENS, MOONDREAM_HIDDEN_SIZE)
        if shape != expected:
            raise ValueError(
                f"projected features must have shape {expected}, got {shape}"
            )

        nbytes = int(tensor.numel()) * int(tensor.element_size())
        if nbytes <= 0:
            raise ValueError("projected features must occupy at least one byte")

        return cls(
            tensor=tensor,
            shape=(shape[0], shape[1]),
            dtype=str(tensor.dtype).removeprefix("torch."),
            device=str(tensor.device),
            nbytes=nbytes,
        )

    def metadata(self) -> dict[str, int | str]:
        """Return JSON-safe metadata without serializing the tensor payload."""

        return {
            "visual_tokens": self.shape[0],
            "hidden_size": self.shape[1],
            "dtype": self.dtype,
            "device": self.device,
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class VisionEncoding:
    """Projected features and measurements produced by the edge stage."""

    projected: ProjectedVisionFeatures
    metrics: Mapping[str, MetricValue]
