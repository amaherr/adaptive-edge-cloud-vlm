"""Typed manual state recorded with Phase 1 configuration decisions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adaptive_vlm.contracts import InferenceRequest


@dataclass(frozen=True)
class SystemState:
    source: str
    uplink_bps: float | None = None
    rtt_ms: float | None = None
    edge_memory_budget_mib: float | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("system state source is required")
        for value in (self.uplink_bps, self.rtt_ms, self.edge_memory_budget_mib):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("system state values must be finite and non-negative")
        if self.uplink_bps == 0:
            raise ValueError("uplink bandwidth must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskState:
    image_bytes: int
    prompt_characters: int
    prompt_tokens: int | None = None
    expected_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.image_bytes <= 0 or self.prompt_characters <= 0:
            raise ValueError("task image and prompt must be non-empty")
        for value in (self.prompt_tokens, self.expected_output_tokens):
            if value is not None and value < 0:
                raise ValueError("task token counts must be non-negative")

    @classmethod
    def from_request(
        cls, request: InferenceRequest, *, expected_output_tokens: int | None = None
    ) -> TaskState:
        return cls(
            image_bytes=Path(request.image_path).stat().st_size,
            prompt_characters=len(request.prompt),
            expected_output_tokens=expected_output_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
