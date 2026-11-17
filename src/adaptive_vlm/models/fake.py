"""Deterministic, dependency-free adapter for infrastructure tests.

Besides its outputs, the fake reports the same metric names Moondream reports
(prompt tokens, image size, prefill, decode, end-to-end and edge vision
times), so a rehearsal produces profile rows with the full field set the
Predictor V2 dataset requires. Its "image size" and "prompt tokens" are
deterministic stand-ins (it decodes no pixels and has no tokenizer); rehearsal
task features use the same rules (``fake_image_geometry``,
``fake_prompt_tokens``), so features and measured fields agree exactly.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Literal

from adaptive_vlm.contracts import InferenceRequest, InferenceResult, MetricValue
from adaptive_vlm.models.base import ModelAdapter, ModelMetadata
from adaptive_vlm.models.features import (
    MOONDREAM_HIDDEN_SIZE,
    MOONDREAM_VISUAL_TOKENS,
    ProjectedVisionFeatures,
    VisionEncoding,
)
from adaptive_vlm.timing import perf_counter_ns
from adaptive_vlm.transport.tensors import TensorInferenceRequest, TensorPayload

FAKE_OUTPUT_TOKENS = 8
# Optional per-call delay for fault-injection rehearsals, so a kill can land
# inside a sub-block; unset (0) everywhere else.
DELAY_ENVIRONMENT = "ADAPTIVE_VLM_FAKE_DELAY_MS"


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
        # Clear the top exponent bit of every bf16 value so the digest reads
        # as finite numbers (|x| < 2), which the int8 encoding requires.
        high = 1 if sys.byteorder == "little" else 0
        digest = bytes(
            byte & 0xBF if index % 2 == high else byte
            for index, byte in enumerate(self.image_digest)
        )
        return TensorPayload(
            shape=self.shape,
            dtype=self.dtype,
            byte_order=sys.byteorder,
            layout="contiguous",
            data=digest + bytes(size - len(digest)),
        )


def fake_tensor_payload(tensor: object) -> TensorPayload:
    """Serializer for :class:`FakeProjectedTensor` in place of ``from_torch``."""

    if not isinstance(tensor, FakeProjectedTensor):
        raise TypeError("fake serializer accepts only FakeProjectedTensor")
    return tensor.to_payload()


def fake_image_geometry(image: bytes) -> tuple[int, int]:
    """Deterministic stand-in image size derived from the bytes (not pixels)."""

    digest = hashlib.sha256(image).digest()
    return 64 + digest[0] * 4, 64 + digest[1] * 4


def fake_prompt_tokens(prompt: str) -> int:
    """Deterministic stand-in prompt length."""

    return 5 + len(prompt.split())


def _model_metrics(prompt: str, elapsed_ms: float) -> dict[str, MetricValue]:
    return {
        "prompt_tokens": fake_prompt_tokens(prompt),
        "prompt_prefill_ms": elapsed_ms / 4,
        "decode_ms": elapsed_ms / 2,
        "output_tokens": FAKE_OUTPUT_TOKENS,
    }


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
        self.delay_s = float(os.environ.get(DELAY_ENVIRONMENT, "0")) / 1000
        if self.delay_s < 0:
            raise ValueError(f"{DELAY_ENVIRONMENT} must be non-negative")

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
        time.sleep(self.delay_s)
        image = request.image_path.read_bytes()
        digest = hashlib.sha256()
        digest.update(image)
        digest.update(b"\0")
        digest.update(request.prompt.encode("utf-8"))
        raw_digest = digest.digest()
        elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000
        self.active_stage = "text"
        width, height = fake_image_geometry(image)

        return InferenceResult(
            request_id=request.request_id,
            output_text=f"fake-{digest.hexdigest()[:16]}",
            output_token_ids=tuple(raw_digest[:FAKE_OUTPUT_TOKENS]),
            metrics={
                "inference_ms": elapsed_ms,
                "end_to_end_ms": elapsed_ms,
                "input_bytes": len(image),
                "image_width": width,
                "image_height": height,
                **_model_metrics(request.prompt, elapsed_ms),
            },
        )

    def encode_projected(self, request: InferenceRequest) -> VisionEncoding:
        """Edge half of a fake distributed request: hash the image bytes."""

        if not request.image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {request.image_path}")
        started_ns = perf_counter_ns()
        time.sleep(self.delay_s)
        image = request.image_path.read_bytes()
        image_digest = hashlib.sha256(image).digest()
        self.active_stage = "vision"
        elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000
        width, height = fake_image_geometry(image)
        return VisionEncoding(
            projected=ProjectedVisionFeatures.from_tensor(
                FakeProjectedTensor(image_digest)
            ),
            metrics={
                "edge_vision_ms": elapsed_ms,
                "edge_vision_end_to_end_ms": elapsed_ms,
                "image_width": width,
                "image_height": height,
            },
        )

    def prepare_prompt(self, prompt: str) -> tuple[int, ...]:
        """Prompt-first preparation: deterministic stand-in token IDs."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        return tuple(range(fake_prompt_tokens(prompt)))

    def generate_from_payload(
        self,
        request: TensorInferenceRequest,
        prepared_prompt_ids: object | None = None,
    ) -> InferenceResult:
        """Cloud half of a fake distributed request (prompt-first gives the same)."""

        if prepared_prompt_ids is not None and prepared_prompt_ids != (
            self.prepare_prompt(request.prompt)
        ):
            raise ValueError("prepared prompt differs from the request prompt")
        started_ns = perf_counter_ns()
        time.sleep(self.delay_s)
        digest = hashlib.sha256(request.tensor.data[:32])
        digest.update(b"\0")
        digest.update(request.prompt.encode("utf-8"))
        raw_digest = digest.digest()
        elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000
        return InferenceResult(
            request_id=request.request_id,
            output_text=f"fake-{digest.hexdigest()[:16]}",
            output_token_ids=tuple(raw_digest[:FAKE_OUTPUT_TOKENS]),
            metrics=_model_metrics(request.prompt, elapsed_ms),
        )
