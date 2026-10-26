"""Local execution policy used by the fake and future all-edge adapters."""

from __future__ import annotations

from adaptive_vlm.config import ExecutionMode
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.models.base import ModelAdapter


class LocalExecutionPolicy(ExecutionPolicy):
    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.EDGE

    def infer(self, request: InferenceRequest) -> InferenceResult:
        return self._adapter.generate(request)

    def prompt_token_count(self, prompt: str) -> int:
        """Count the pinned query tokens before choosing an execution policy."""
        prepare = getattr(self._adapter, "prepare_prompt", None)
        if not callable(prepare):
            raise ValueError("resident adapter cannot count exact prompt tokens")
        return len(prepare(prompt))
