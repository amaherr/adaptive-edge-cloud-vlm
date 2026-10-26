"""Manual Phase 1 selection and single-policy dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.scheduler.configuration import CapabilityRegistry, ConfigurationId
from adaptive_vlm.scheduler.state import SystemState, TaskState


@dataclass(frozen=True)
class ConfigurationDecision:
    configuration_id: ConfigurationId
    reason: str
    model_fingerprint: str
    system_state: SystemState
    task_state: TaskState
    capabilities: tuple[dict[str, str | bool | None], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id.value,
            "reason": self.reason,
            "model_fingerprint": self.model_fingerprint,
            "system_state": self.system_state.to_dict(),
            "task_state": self.task_state.to_dict(),
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class DispatchResult:
    decision: ConfigurationDecision
    result: InferenceResult
    selection_overhead_ms: float = 0.0


class ConfigurationDispatcher:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    def infer(
        self,
        request: InferenceRequest,
        *,
        configuration_id: ConfigurationId,
        system_state: SystemState,
        task_state: TaskState,
    ) -> DispatchResult:
        started_ns = perf_counter_ns()
        policy = self.registry.policy_for(configuration_id)
        decision = ConfigurationDecision(
            configuration_id=configuration_id,
            reason="manual_phase1",
            model_fingerprint=self.registry.expected_model_fingerprint,
            system_state=system_state,
            task_state=task_state,
            capabilities=tuple(status.to_dict() for status in self.registry.statuses()),
        )
        overhead_ms = (perf_counter_ns() - started_ns) / 1_000_000
        return DispatchResult(
            decision=decision,
            result=policy.infer(request),
            selection_overhead_ms=overhead_ms,
        )
