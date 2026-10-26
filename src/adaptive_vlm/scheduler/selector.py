"""Pure, deterministic selection over a complete configuration snapshot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from adaptive_vlm.scheduler.configuration import (
    CONFIGURATIONS,
    CapabilityStatus,
    ConfigurationId,
)
from adaptive_vlm.scheduler.predictor import Prediction


@dataclass(frozen=True)
class CandidateSnapshot:
    configuration_id: ConfigurationId
    capability: CapabilityStatus
    prediction: Prediction | None
    healthy: bool = True
    quality_score: float | None = None

    def __post_init__(self) -> None:
        if self.capability.configuration_id != self.configuration_id:
            raise ValueError("capability configuration does not match candidate")
        if (
            self.prediction is not None
            and self.prediction.configuration_id != self.configuration_id.value
        ):
            raise ValueError("prediction configuration does not match candidate")


@dataclass(frozen=True)
class SelectionConstraints:
    sla_ms: float | None = None
    quality_floor: float | None = None
    edge_memory_budget_mib: float | None = None

    def __post_init__(self) -> None:
        if self.sla_ms is not None and (
            not math.isfinite(self.sla_ms) or self.sla_ms <= 0
        ):
            raise ValueError("SLA must be finite and positive")
        if self.quality_floor is not None and not math.isfinite(self.quality_floor):
            raise ValueError("quality floor must be finite")
        if self.edge_memory_budget_mib is not None and (
            not math.isfinite(self.edge_memory_budget_mib)
            or self.edge_memory_budget_mib < 0
        ):
            raise ValueError("memory budget must be finite and non-negative")


@dataclass(frozen=True)
class CandidateDecision:
    configuration_id: str
    eligible: bool
    exclusion_reason: str | None
    predicted_sla_met: bool | None
    capability: CapabilityStatus
    healthy: bool
    quality_score: float | None
    prediction: Prediction | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "eligible": self.eligible,
            "exclusion_reason": self.exclusion_reason,
            "predicted_sla_met": self.predicted_sla_met,
            "capability": self.capability.to_dict(),
            "healthy": self.healthy,
            "quality_score": self.quality_score,
            "prediction": self.prediction.to_dict() if self.prediction else None,
        }


@dataclass(frozen=True)
class SelectionDecision:
    selected_configuration_id: str | None
    status: str  # selected, sla_unmet, or no_feasible_candidate
    candidates: tuple[CandidateDecision, ...]
    constraints: SelectionConstraints

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "selected_configuration_id": self.selected_configuration_id,
            "status": self.status,
            "constraints": {
                "sla_ms": self.constraints.sla_ms,
                "quality_floor": self.constraints.quality_floor,
                "edge_memory_budget_mib": self.constraints.edge_memory_budget_mib,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _exclusion(
    candidate: CandidateSnapshot, constraints: SelectionConstraints
) -> str | None:
    if not candidate.capability.available:
        return candidate.capability.reason
    if not candidate.healthy:
        return "unhealthy"
    prediction = candidate.prediction
    if prediction is None:
        return "prediction_missing"
    if not prediction.valid:
        return prediction.reason or "prediction_invalid"
    if (
        prediction.total_ms is None
        or not math.isfinite(prediction.total_ms)
        or prediction.total_ms <= 0
    ):
        return "prediction_nonfinite_or_nonpositive"
    if constraints.quality_floor is not None:
        if candidate.quality_score is None or not math.isfinite(
            candidate.quality_score
        ):
            return "quality_unknown"
        if candidate.quality_score < constraints.quality_floor:
            return "quality_below_floor"
    if (
        constraints.edge_memory_budget_mib is not None
        and candidate.configuration_id != ConfigurationId.CLOUD_RAW_IMAGE
    ):
        required = prediction.edge_memory_requirement_mib
        if required is None or not math.isfinite(required) or required <= 0:
            return "edge_memory_requirement_unknown"
        if required > constraints.edge_memory_budget_mib:
            return "edge_memory_budget_exceeded"
    return None


def select(
    snapshots: tuple[CandidateSnapshot, ...],
    constraints: SelectionConstraints | None = None,
) -> SelectionDecision:
    """Filter unsafe/unknown candidates, then choose minimum predicted latency.

    The stable configuration order breaks exact ties. SLA is a preference among
    feasible candidates; when none meets it, the fastest safe candidate is logged
    with ``sla_unmet``. No inference or measured outcomes enter this function.
    """
    constraints = constraints or SelectionConstraints()
    expected = {spec.id for spec in CONFIGURATIONS}
    if (
        len(snapshots) != len(expected)
        or {item.configuration_id for item in snapshots} != expected
    ):
        raise ValueError("selection requires exactly one snapshot per configuration")
    by_id = {item.configuration_id: item for item in snapshots}
    decisions: list[CandidateDecision] = []
    for spec in CONFIGURATIONS:
        item = by_id[spec.id]
        reason = _exclusion(item, constraints)
        total = item.prediction.total_ms if item.prediction is not None else None
        decisions.append(
            CandidateDecision(
                configuration_id=spec.id.value,
                eligible=reason is None,
                exclusion_reason=reason,
                predicted_sla_met=(total <= constraints.sla_ms)
                if reason is None
                and constraints.sla_ms is not None
                and total is not None
                else None,
                capability=item.capability,
                healthy=item.healthy,
                quality_score=item.quality_score,
                prediction=item.prediction,
            )
        )
    feasible = [item for item in decisions if item.eligible]
    if not feasible:
        return SelectionDecision(
            None, "no_feasible_candidate", tuple(decisions), constraints
        )
    within_sla = [item for item in feasible if item.predicted_sla_met]
    pool = within_sla if within_sla else feasible
    selected = min(
        pool,
        key=lambda item: (
            float(item.prediction.total_ms)
            if item.prediction and item.prediction.total_ms is not None
            else math.inf
        ),
    )
    status = (
        "sla_unmet" if constraints.sla_ms is not None and not within_sla else "selected"
    )
    return SelectionDecision(
        selected.configuration_id, status, tuple(decisions), constraints
    )


def predicted_oracle(decision: SelectionDecision) -> str | None:
    """Independent minimum over the logged feasible predictions for audit."""
    eligible = [candidate for candidate in decision.candidates if candidate.eligible]
    if not eligible:
        return None
    order = {spec.id.value: index for index, spec in enumerate(CONFIGURATIONS)}
    eligible.sort(
        key=lambda item: (
            item.prediction.total_ms if item.prediction else math.inf,
            order[item.configuration_id],
        )
    )
    return eligible[0].configuration_id
