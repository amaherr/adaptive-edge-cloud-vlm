"""Deterministic, dependency-free adapter for infrastructure tests."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from typing import Literal

from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.models.base import ModelAdapter, ModelMetadata
from adaptive_vlm.models.features import (
    MOONDREAM_HIDDEN_SIZE,
    MOONDREAM_VISUAL_TOKENS,
    ProjectedVisionFeatures,
    VisionEncoding,
)
from adaptive_vlm.timing import perf_counter_ns
from adaptive_vlm.transport.tensors import TensorInferenceRequest, TensorPayload


@dataclass(frozen=True)
class FakeProjectedTensor:
    """Projected-feature stand-in carrying the image digest in its first bytes."""

    image_digest: bytes
    shape: tuple[int, int] = (MOONDREAM_VISUAL_TOKENS, MOONDREAM_HIDDEN_SIZE)
    dtype: str = "bfloat16"
    device: str = "cpu"

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def element_size(self) -> int:
        return 2

    def to_payload(self) -> TensorPayload:
        size = self.numel() * self.element_size()
        return TensorPayload(
            shape=self.shape,
            dtype=self.dtype,
            byte_order=sys.byteorder,
            layout="contiguous",
            data=self.image_digest + bytes(size - len(self.image_digest)),
        )


def fake_tensor_payload(tensor: object) -> TensorPayload:
    """Serializer for :class:`FakeProjectedTensor` in place of ``from_torch``."""

    if not isinstance(tensor, FakeProjectedTensor):
        raise TypeError("fake serializer accepts only FakeProjectedTensor")
    return tensor.to_payload()


class FakeModelAdapter(ModelAdapter):
    """Generate repeatable output from image bytes and prompt text."""

    component_role = "full"
    _metadata = ModelMetadata(
        model_id="test/deterministic-vlm",
        revision="v1",
        fingerprint=hashlib.sha256(b"test/deterministic-vlm@v1").hexdigest(),
    )

    def __init__(self) -> None:
        self.active_stage: str | None = None
        self.stage_activations: list[str] = []

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def activate_stage(self, stage: Literal["vision", "text"]) -> float:
        """Record stage changes the way swap residency would, at no cost."""

        if stage not in ("vision", "text"):
            raise ValueError(f"unknown stage: {stage}")
        if self.active_stage != stage:
            self.stage_activations.append(stage)
            self.active_stage = stage
        return 0.0

    def generate(self, request: InferenceRequest) -> InferenceResult:
        if not request.image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {request.image_path}")

        started_ns = perf_counter_ns()
        digest = hashlib.sha256()
        digest.update(request.image_path.read_bytes())
        digest.update(b"\0")
        digest.update(request.prompt.encode("utf-8"))
        raw_digest = digest.digest()
        elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000
        self.active_stage = "text"

        return InferenceResult(
            request_id=request.request_id,
            output_text=f"fake-{digest.hexdigest()[:16]}",
            output_token_ids=tuple(raw_digest[:8]),
            metrics={
                "inference_ms": elapsed_ms,
                "input_bytes": request.image_path.stat().st_size,
                "output_tokens": 8,
            },
        )

    def encode_projected(self, request: InferenceRequest) -> VisionEncoding:
        """Edge half of a fake distributed request: hash the image bytes."""

        if not request.image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {request.image_path}")
        started_ns = perf_counter_ns()
        image_digest = hashlib.sha256(request.image_path.read_bytes()).digest()
        self.active_stage = "vision"
        return VisionEncoding(
            projected=ProjectedVisionFeatures.from_tensor(
                FakeProjectedTensor(image_digest)
            ),
            metrics={"edge_vision_ms": (perf_counter_ns() - started_ns) / 1_000_000},
        )

    def generate_from_payload(self, request: TensorInferenceRequest) -> InferenceResult:
        """Cloud half of a fake distributed request."""

        digest = hashlib.sha256(request.tensor.data[:32])
        digest.update(b"\0")
        digest.update(request.prompt.encode("utf-8"))
        raw_digest = digest.digest()
        return InferenceResult(
            request_id=request.request_id,
            output_text=f"fake-{digest.hexdigest()[:16]}",
            output_token_ids=tuple(raw_digest[:8]),
            metrics={"output_tokens": 8},
        )
