"""Select and execute one adaptive request with an explicit safe retry bound."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from time import perf_counter_ns
from typing import Any

from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.scheduler.configuration import (
    CONFIGURATIONS,
    CapabilityRegistry,
    ConfigurationId,
)
from adaptive_vlm.scheduler.predictor import (
    PredictionState,
    PredictionTask,
    Predictor,
)
from adaptive_vlm.scheduler.selector import (
    CandidateSnapshot,
    SelectionConstraints,
    select,
)
from adaptive_vlm.transport.client import PreInferenceConnectionError


@dataclass(frozen=True)
class AdaptiveExecution:
    result: InferenceResult | None
    selected_configuration_id: ConfigurationId | None
    attempts: tuple[dict[str, Any], ...]
    error: Exception | None
    selection_overhead_ms: float
    retry_overhead_ms: float
    monitor_overhead_ms: float = 0.0


@dataclass(frozen=True)
class RegistrySnapshot:
    registry: CapabilityRegistry
    monitoring: dict[str, Any]


def execute_adaptive_request(
    *,
    request: InferenceRequest,
    task: PredictionTask,
    state: PredictionState,
    placement_hashes: Mapping[ConfigurationId, str],
    healthy: Mapping[ConfigurationId, bool],
    constraints: SelectionConstraints,
    predictor: Predictor,
    registry_provider: Callable[[], CapabilityRegistry | RegistrySnapshot],
    allow_safe_retry: bool = True,
) -> AdaptiveExecution:
    """Hold each choice stable through inference and retry only before send.

    A failed connection is the only policy failure known to precede inference.
    The failed configuration is excluded from the one fresh decision. Every
    selection, attempt, and elapsed time is returned even on failure.
    """
    expected = {spec.id for spec in CONFIGURATIONS}
    if set(placement_hashes) != expected or set(healthy) != expected:
        raise ValueError("adaptive state must cover every configuration")
    attempts: list[dict[str, Any]] = []
    excluded: set[ConfigurationId] = set()
    total_selection_ms = 0.0
    retry_overhead_ms = 0.0
    monitor_overhead_ms = 0.0
    for attempt_index in range(2 if allow_safe_retry else 1):
        selection_start = perf_counter_ns()
        registry_result = registry_provider()
        registry = (
            registry_result.registry
            if isinstance(registry_result, RegistrySnapshot)
            else registry_result
        )
        monitoring = (
            registry_result.monitoring
            if isinstance(registry_result, RegistrySnapshot)
            else None
        )
        if monitoring is not None:
            monitor_overhead_ms += float(monitoring["sampling_overhead_ms"])
        snapshots = tuple(
            CandidateSnapshot(
                spec.id,
                registry.status(spec),
                predictor.predict(
                    spec,
                    replace(state, placement_profile_sha256=placement_hashes[spec.id]),
                    task,
                ),
                healthy=healthy[spec.id] and spec.id not in excluded,
            )
            for spec in CONFIGURATIONS
        )
        decision = select(snapshots, constraints)
        selection_ms = (perf_counter_ns() - selection_start) / 1_000_000
        total_selection_ms += selection_ms
        if attempt_index > 0:
            retry_overhead_ms += selection_ms
        selected = decision.selected_configuration_id
        attempt: dict[str, Any] = {
            "attempt_index": attempt_index,
            "decision": decision.to_dict(),
            "selection_overhead_ms": selection_ms,
            "execution_ms": 0.0,
            "status": "not_started",
        }
        if monitoring is not None:
            attempt["monitoring"] = monitoring
        attempts.append(attempt)
        if selected is None:
            attempt["status"] = "no_feasible_candidate"
            return AdaptiveExecution(
                None,
                None,
                tuple(attempts),
                ValueError("no feasible candidate"),
                total_selection_ms,
                retry_overhead_ms,
                monitor_overhead_ms,
            )
        configuration_id = ConfigurationId(selected)
        execution_start = perf_counter_ns()
        try:
            result = registry.policy_for(configuration_id).infer(request)
            if result.request_id != request.request_id:
                raise ValueError("policy returned a different request_id")
        except Exception as exc:
            attempt["execution_ms"] = (perf_counter_ns() - execution_start) / 1_000_000
            attempt["status"] = "failed"
            attempt["error_type"] = type(exc).__name__
            attempt["error_message"] = str(exc)
            if (
                isinstance(exc, PreInferenceConnectionError)
                and attempt_index == 0
                and allow_safe_retry
            ):
                attempt["retry_reason"] = "connection_failed_before_send"
                excluded.add(configuration_id)
                retry_overhead_ms += float(attempt["execution_ms"])
                continue
            return AdaptiveExecution(
                None,
                configuration_id,
                tuple(attempts),
                exc,
                total_selection_ms,
                retry_overhead_ms,
                monitor_overhead_ms,
            )
        attempt["execution_ms"] = (perf_counter_ns() - execution_start) / 1_000_000
        attempt["status"] = "ok"
        return AdaptiveExecution(
            result,
            configuration_id,
            tuple(attempts),
            None,
            total_selection_ms,
            retry_overhead_ms,
            monitor_overhead_ms,
        )
    raise AssertionError("adaptive retry loop exhausted unexpectedly")
