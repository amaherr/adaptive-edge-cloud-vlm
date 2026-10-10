"""Deterministic, dependency-free adapter for infrastructure tests."""

from __future__ import annotations

import hashlib
from time import perf_counter_ns

from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.models.base import ModelAdapter, ModelMetadata


class FakeModelAdapter(ModelAdapter):
    """Generate repeatable output from image bytes and prompt text."""

    _metadata = ModelMetadata(
        model_id="test/deterministic-vlm",
        revision="v1",
        fingerprint=hashlib.sha256(b"test/deterministic-vlm@v1").hexdigest(),
    )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

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

