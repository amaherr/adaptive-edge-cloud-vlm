"""Versioned physical latency predictor with separate research/live eligibility."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adaptive_vlm.prediction.v1.calibration import canonical_sha256
from adaptive_vlm.prediction.v1.predictor import Prediction, PredictionTask, transfer_ms
from adaptive_vlm.prediction.v2.dataset.parsing import number, strict_json
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS, ConfigurationSpec

VARIANTS = ("constant_median", "v1_recalibrated", "v2_a")


@dataclass(frozen=True)
class PredictionStateV2:
    host_pair_id: str
    model_contract_sha256: str
    placement_profile_sha256: str
    tokenizer_sha256: str
    model_artifact_sha256: str
    model_revision: str
    dataset_manifest_sha256: str
    network_bucket_id: str
    network_source: str
    precision: str = "bf16"
    uplink_bps: float | None = None
    downlink_bps: float | None = None
    rtt_ms: float | None = None
    probe_age_seconds: float | None = None
    probe_scope: str = "unknown"
    edge_memory_budget_mib: float | None = None
    cloud_memory_budget_mib: float | None = None
    server_healthy: bool | None = None
    correctness_approved: bool = False
    available_configurations: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionV2(Prediction):
    cloud_memory_requirement_mib: float | None = None
    selection_eligible: bool = False
    execution_feasible: bool = False
    diagnostic_estimate_available: bool = False
    warnings: tuple[str, ...] = ()
    support_distance: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationV2:
    schema_version: int
    calibration_id: str
    variant: str
    training_rows_sha256: str
    split_sha256: str
    policy: dict[str, Any]
    candidates: dict[str, dict[str, Any]]
    training_request_ids: tuple[str, ...]
    training_block_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        *,
        variant: str,
        training_rows_sha256: str,
        split_sha256: str,
        policy: dict[str, Any],
        candidates: dict[str, dict[str, Any]],
        request_ids: tuple[str, ...],
        block_ids: tuple[str, ...],
    ) -> CalibrationV2:
        raw = {
            "schema_version": 2,
            "variant": variant,
            "training_rows_sha256": training_rows_sha256,
            "split_sha256": split_sha256,
            "policy": policy,
            "candidates": candidates,
            "training_request_ids": list(request_ids),
            "training_block_ids": list(block_ids),
        }
        raw["calibration_id"] = canonical_sha256(raw)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CalibrationV2:
        if raw.get("schema_version") != 2 or raw.get("variant") not in VARIANTS:
            raise ValueError("unsupported Predictor V2 schema/variant")
        if canonical_sha256(
            {k: v for k, v in raw.items() if k != "calibration_id"}
        ) != raw.get("calibration_id"):
            raise ValueError("V2 calibration content hash mismatch")
        if set(raw["candidates"]) != {c.id.value for c in CONFIGURATIONS}:
            raise ValueError(
                "V2 calibration needs exactly the three registered policies"
            )
        for candidate in raw["candidates"].values():
            for name in (
                "expected_output_tokens",
                "decode_ms_per_token",
                "request_framing_bytes",
                "response_wire_bytes",
                "interval_half_width_ms",
                "constant_median_ms",
            ):
                number(candidate[name], name)
            if candidate["expected_output_tokens"] > 32:
                raise ValueError("output prior exceeds generation budget")
            for component in candidate["components"].values():
                number(component["intercept_ms"], "compute intercept")
                for name, slope in component["slopes"].items():
                    if name not in {"image_pixels", "prompt_tokens"}:
                        raise ValueError("unapproved compute feature")
                    number(slope, "compute slope")
                    number(component["scales"][name], "compute scale", positive=True)
            intercept = candidate["network"]["intercept_ms"]
            if (
                isinstance(intercept, bool)
                or not isinstance(intercept, (int, float))
                or not math.isfinite(intercept)
            ):
                raise ValueError("network intercept must be finite")
            if raw["variant"] != "v1_recalibrated":
                number(candidate["network"]["intercept_ms"], "transport intercept")
            number(candidate["network"]["rtt_coefficient"], "RTT sensitivity")
        return cls(
            schema_version=2,
            calibration_id=raw["calibration_id"],
            variant=raw["variant"],
            training_rows_sha256=raw["training_rows_sha256"],
            split_sha256=raw["split_sha256"],
            policy=raw["policy"],
            candidates=raw["candidates"],
            training_request_ids=tuple(raw["training_request_ids"]),
            training_block_ids=tuple(raw["training_block_ids"]),
        )

    @classmethod
    def load(cls, path: Path) -> CalibrationV2:
        return cls.from_dict(strict_json(path.read_bytes()))


def state_from_snapshot(
    snapshot: dict[str, Any], policy: dict[str, Any]
) -> PredictionStateV2:
    """Offline campaign preflight inventory, never inferred from trial success."""
    cid = snapshot["configuration_id"]
    return PredictionStateV2(
        host_pair_id=policy["host_pair_id"],
        model_contract_sha256=policy["model_contract_sha256"],
        dataset_manifest_sha256=policy["dataset_manifest_sha256"],
        placement_profile_sha256=policy["placement_profile_sha256"][cid],
        tokenizer_sha256=policy["tokenizer_sha256"],
        model_artifact_sha256=policy["model_artifact_sha256"],
        model_revision=policy["model_revision"],
        precision="bf16",
        edge_memory_budget_mib=policy["edge_dedicated_memory_mib"],
        cloud_memory_budget_mib=policy["cloud_dedicated_memory_mib"],
        server_healthy=True,
        correctness_approved=True,
        available_configurations=tuple(c.id.value for c in CONFIGURATIONS),
        **snapshot["network"],
    )


def component_value(component: dict[str, Any], task: PredictionTask) -> float:
    values = {
        "image_pixels": task.image_width * task.image_height,
        "prompt_tokens": task.prompt_tokens,
    }
    result = float(component["intercept_ms"])
    for name, coefficient in component["slopes"].items():
        value = values[name]
        if value is None:
            raise ValueError("required pre-execution feature is unknown")
        result += coefficient * value / component["scales"][name]
    return result


class PredictorV2:
    def __init__(self, calibration: CalibrationV2) -> None:
        self.calibration = calibration

    def _invalid(
        self, cid: str, reason: str, coverage: str = "unsupported"
    ) -> PredictionV2:
        return PredictionV2(
            configuration_id=cid,
            valid=False,
            reason=reason,
            coverage=coverage,
            calibration_id=self.calibration.calibration_id,
            estimated_output_tokens=None,
            output_estimate_source=None,
            components_ms={},
            total_ms=None,
            lower_ms=None,
            upper_ms=None,
            request_payload_bytes=None,
            request_wire_bytes=None,
            response_wire_bytes=None,
            edge_memory_requirement_mib=None,
        )

    def predict(
        self, spec: ConfigurationSpec, state: PredictionStateV2, task: PredictionTask
    ) -> PredictionV2:
        cid = spec.id.value
        fit = self.calibration.candidates.get(cid)
        policy = self.calibration.policy
        if fit is None:
            return self._invalid(cid, "configuration_not_calibrated")
        identities = (
            "host_pair_id",
            "model_contract_sha256",
            "dataset_manifest_sha256",
            "tokenizer_sha256",
            "model_artifact_sha256",
            "model_revision",
        )
        for name in identities:
            if getattr(state, name) != policy[name]:
                return self._invalid(cid, name + "_mismatch")
        if (
            state.precision != "bf16"
            or state.placement_profile_sha256 != fit["placement_profile_sha256"]
        ):
            return self._invalid(cid, "precision_or_placement_mismatch")
        if cid not in state.available_configurations:
            return self._invalid(cid, "configuration_unavailable")
        if not state.correctness_approved:
            return self._invalid(cid, "correctness_gate_unapproved")
        if spec.mode.value != "edge" and state.server_healthy is not True:
            return self._invalid(cid, "cloud_unhealthy_or_unknown")
        if (
            task.task_bucket != policy["supported_task_bucket"]
            or task.requested_max_new_tokens != 32
        ):
            return self._invalid(cid, "task_or_generation_mismatch")
        if task.expected_output_tokens is not None:
            return self._invalid(cid, "unapproved_output_length_estimator")
        distance: dict[str, float] = {}
        for name, limits in policy["task_envelope"].items():
            value = getattr(task, name)
            try:
                value = number(value, name, positive=True)
            except ValueError:
                return self._invalid(cid, name + "_unknown_or_invalid")
            if not limits[0] <= value <= limits[1]:
                return self._invalid(cid, "task_outside_operational_envelope")
            low, high = fit["task_support"][name]
            distance[name] = max(low - value, value - high, 0.0)
        remote = spec.mode.value != "edge"
        up = down = rtt = 0.0
        if remote:
            if state.network_source != policy["network_source"]:
                return self._invalid(cid, "network_source_mismatch")
            if state.probe_scope not in {"live", "controlled_block_preflight"}:
                return self._invalid(cid, "network_probe_scope_unknown")
            try:
                age = number(state.probe_age_seconds, "probe_age")
                up = number(state.uplink_bps, "uplink", positive=True)
                down = number(state.downlink_bps, "downlink", positive=True)
                rtt = number(state.rtt_ms, "rtt")
            except ValueError:
                return self._invalid(cid, "network_state_missing_or_invalid")
            ceiling = (
                policy["live_probe_max_age_seconds"]
                if state.probe_scope == "live"
                else policy["max_controlled_block_probe_age_seconds"]
            )
            if age > ceiling:
                return self._invalid(cid, "network_probe_stale")
            for name, value in (
                ("uplink_bps", up),
                ("downlink_bps", down),
                ("rtt_ms", rtt),
            ):
                limits = policy["network_envelope"][name]
                if not limits[0] <= value <= limits[1]:
                    return self._invalid(cid, "network_outside_operational_envelope")
                low, high = fit["network"]["support"][name]
                distance[name] = max(low - value, value - high, 0.0)
        extrapolated = any(value > 0 for value in distance.values())
        coverage = "physics_extrapolation" if extrapolated else "interpolation"
        if self.calibration.variant == "v1_recalibrated" and extrapolated:
            return self._invalid(cid, "v1_strict_support_exceeded", coverage)
        payload = (
            float(task.image_bytes)
            if spec.mode.value == "cloud"
            else 2985984.0
            if spec.mode.value == "distributed"
            else 0.0
        )
        request_wire = payload + fit["request_framing_bytes"] if remote else 0.0
        response_wire = fit["response_wire_bytes"] if remote else 0.0
        if self.calibration.variant == "constant_median":
            components = {
                "constant_training_median_ms": float(fit["constant_median_ms"])
            }
        else:
            components = {
                name: component_value(component, task)
                for name, component in fit["components"].items()
            }
            components["decode_ms"] = (
                fit["decode_ms_per_token"] * fit["expected_output_tokens"]
            )
            components["transport_ms"] = (
                (
                    fit["network"]["intercept_ms"]
                    + fit["network"]["rtt_coefficient"] * rtt
                    + transfer_ms(request_wire, up)
                    + transfer_ms(response_wire, down)
                )
                if remote
                else 0.0
            )
        if any(not math.isfinite(v) or v < 0 for v in components.values()):
            return self._invalid(cid, "invalid_component_prediction", coverage)
        total = sum(components.values())
        if total <= 0:
            return self._invalid(cid, "nonpositive_prediction", coverage)
        memory_reason = None
        for node, needed in (("edge", spec.mode.value != "cloud"), ("cloud", remote)):
            if not needed:
                continue
            requirement = fit[node + "_memory_requirement_mib"]
            budget = getattr(state, node + "_memory_budget_mib")
            if requirement is None:
                memory_reason = node + "_memory_requirement_unknown"
                break
            try:
                available = number(budget, node + "_budget", positive=True)
            except ValueError:
                memory_reason = node + "_memory_budget_unknown"
                break
            if available > policy[node + "_dedicated_memory_mib"]:
                memory_reason = node + "_memory_inventory_mismatch"
                break
            if available < requirement:
                memory_reason = node + "_memory_budget_exceeded"
                break
        feasible = memory_reason is None
        warnings = []
        if extrapolated:
            warnings.append("outside_measured_training_support")
        offline = state.probe_scope != "live"
        if offline:
            warnings.append("offline_campaign_preflight_state_not_live")
        if memory_reason:
            warnings.append("latency_is_diagnostic_only_not_execution_eligibility")
        eligible = (
            feasible
            and not offline
            and (not extrapolated or policy["automatic_extrapolation_enabled"])
        )
        width = fit["interval_half_width_ms"] * (
            policy["extrapolation_interval_multiplier"] if extrapolated else 1
        )
        return PredictionV2(
            configuration_id=cid,
            valid=feasible,
            reason=memory_reason,
            coverage=coverage,
            calibration_id=self.calibration.calibration_id,
            estimated_output_tokens=float(fit["expected_output_tokens"]),
            output_estimate_source="training_median",
            components_ms=components,
            total_ms=total,
            lower_ms=max(0.0, total - width),
            upper_ms=total + width,
            request_payload_bytes=payload,
            request_wire_bytes=request_wire,
            response_wire_bytes=response_wire,
            edge_memory_requirement_mib=fit["edge_memory_requirement_mib"],
            cloud_memory_requirement_mib=fit["cloud_memory_requirement_mib"],
            selection_eligible=eligible,
            execution_feasible=feasible,
            diagnostic_estimate_available=True,
            warnings=tuple(warnings),
            support_distance=distance,
        )
