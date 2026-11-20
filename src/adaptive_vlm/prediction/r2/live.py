"""The live G5 adaptive policy: sense, predict with frozen r2, select, execute.

``AdaptiveRunner.infer`` is timed by the interleaved benchmark like any fixed
placement, so sensing, prediction and selection overhead are part of the
adaptive policy's observed latency. The chosen configuration runs through
the same policy object the fixed placement uses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.prediction.r2.g4 import decide
from adaptive_vlm.prediction.r2.models import Calibration
from adaptive_vlm.prediction.r2.units import UnitInputs
from adaptive_vlm.scheduler.configuration import (
    ALL_CONFIGURATIONS,
    SPEC_BY_ID,
    ConfigurationId,
)
from adaptive_vlm.scheduler.sensing import Sensor
from adaptive_vlm.timing import perf_counter_ns

ADAPTIVE = "adaptive"


class NoFeasibleConfiguration(RuntimeError):
    """The selector excluded every configuration; nothing was dispatched."""


@dataclass(frozen=True)
class AdaptiveRecord:
    """Identity of the frozen inputs, recorded in the adaptive plan."""

    calibration_sha256: str
    sensing_sha256: str
    network_probe_sha256: str
    task: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_sha256": self.calibration_sha256,
            "sensing_sha256": self.sensing_sha256,
            "network_probe_sha256": self.network_probe_sha256,
            "task": self.task,
            "candidates": [spec.id.value for spec in ALL_CONFIGURATIONS],
        }


class AdaptiveRunner:
    def __init__(
        self,
        *,
        calibration: Calibration,
        record: AdaptiveRecord,
        sensor: Sensor,
        policies: Mapping[str, ExecutionPolicy],
        features: Mapping[str, Mapping[str, Any]],
        prepare_stage: Callable[[str], float] | None = None,
    ) -> None:
        if calibration.get("model") != "r2_component":
            raise ValueError("the adaptive policy uses the frozen r2_component model")
        missing = {spec.id.value for spec in ALL_CONFIGURATIONS} - set(policies)
        if missing:
            raise ValueError(f"adaptive policy lacks configurations: {sorted(missing)}")
        self.calibration = calibration
        self.record = record
        self.sensor = sensor
        self.policies = dict(policies)
        self.features = features
        self.prepare_stage = prepare_stage

    def infer(self, request: InferenceRequest) -> InferenceResult:
        started = perf_counter_ns()
        sensed = self.sensor.sense()
        task = self.features[request.request_id]
        inputs = UnitInputs(
            request_id=request.request_id,
            cell_id="live",
            condition_id="live",
            task=self.record.task,
            edge_load=sensed.edge_load,
            cloud_load=sensed.cloud_load,
            width=int(task["image_width"]),
            height=int(task["image_height"]),
            image_bytes=int(task["image_bytes"]),
            uplink_bps=sensed.uplink_bps,
            rtt_ms=sensed.rtt_ms,
        )
        selection_started = perf_counter_ns()
        decision = decide(self.calibration, inputs)
        selection_ms = (perf_counter_ns() - selection_started) / 1_000_000
        chosen = decision.selected_configuration_id
        if chosen is None:
            raise NoFeasibleConfiguration(
                "the selector found no feasible configuration"
            )
        spec = SPEC_BY_ID[ConfigurationId(chosen)]
        prepare_ms = 0.0
        if self.prepare_stage is not None and spec.prepared_stage is not None:
            prepare_ms = float(self.prepare_stage(spec.prepared_stage))
        decided_ms = (perf_counter_ns() - started) / 1_000_000
        result = self.policies[chosen].infer(request)
        metrics: dict[str, Any] = dict(result.metrics)
        metrics.update(sensed.to_metrics())
        metrics.update(
            {
                "adaptive_selected_configuration_id": chosen,
                "adaptive_decision_status": decision.status,
                "adaptive_selection_ms": selection_ms,
                "adaptive_stage_prepare_ms": prepare_ms,
                "adaptive_overhead_ms": decided_ms,
                **{
                    f"adaptive_predicted_ms_{c.configuration_id}": (
                        c.prediction.total_ms if c.prediction else None
                    )
                    for c in decision.candidates
                },
            }
        )
        return InferenceResult(
            request_id=result.request_id,
            output_text=result.output_text,
            output_token_ids=result.output_token_ids,
            metrics=metrics,
        )
