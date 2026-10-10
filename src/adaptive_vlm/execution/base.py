"""Execution-policy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptive_vlm.config import ExecutionMode
from adaptive_vlm.contracts import InferenceRequest, InferenceResult


class ExecutionPolicy(ABC):
    @property
    @abstractmethod
    def mode(self) -> ExecutionMode:
        """Return the execution mode represented by this policy."""

    @abstractmethod
    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute one inference request."""

