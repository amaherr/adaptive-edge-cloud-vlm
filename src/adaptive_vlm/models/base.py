"""Interfaces shared by real and fake VLM adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from adaptive_vlm.contracts import InferenceRequest, InferenceResult


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    revision: str
    fingerprint: str


class ModelAdapter(ABC):
    """Minimal model interface required by an execution policy."""

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Describe the exact model artifact used by this adapter."""

    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResult:
        """Generate a deterministic response for one request."""

