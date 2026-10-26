"""Deterministic component latency prediction from a frozen profile fit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adaptive_vlm.scheduler.configuration import ConfigurationSpec


def transfer_ms(byte_count: float, goodput_bps: float) -> float:
    """Convert bytes and measured bits/second to transfer milliseconds."""

    if not math.isfinite(byte_count) or byte_count < 0:
        raise ValueError("byte_count must be finite and non-negative")
    if not math.isfinite(goodput_bps) or goodput_bps <= 0:
        raise ValueError("goodput_bps must be finite and positive")
    return 8_000 * byte_count / goodput_bps


@dataclass(frozen=True)
class PredictionState:
    host_pair_id: str
    model_contract_sha256: str
    placement_profile_sha256: str
    network_bucket_id: str
    network_source: str
    uplink_bps: float | None = None
    downlink_bps: float | None = None
    rtt_ms: float | None = None
    edge_memory_budget_mib: float | None = None

    def __post_init__(self) -> None:
        if not all(
            (
                self.host_pair_id,
                self.model_contract_sha256,
                self.placement_profile_sha256,
                self.network_bucket_id,
                self.network_source,
            )
        ):
            raise ValueError("prediction state identifiers must be non-empty")
        for name in ("uplink_bps", "downlink_bps", "rtt_ms", "edge_memory_budget_mib"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.uplink_bps == 0 or self.downlink_bps == 0:
            raise ValueError("bandwidth must be positive when provided")


@dataclass(frozen=True)
class PredictionTask:
    task_bucket: str
    image_bytes: int
    image_width: int
    image_height: int
    prompt_tokens: int | None
    requested_max_new_tokens: int
    expected_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.task_bucket.strip():
            raise ValueError("task bucket must be non-empty")
        if (
            min(
                self.image_bytes,
                self.image_width,
                self.image_height,
                self.requested_max_new_tokens,
            )
            <= 0
        ):
            raise ValueError("task sizes and generation budget must be positive")
        if self.prompt_tokens is not None and self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive when supplied")
        if self.expected_output_tokens is not None and not (
            0 <= self.expected_output_tokens <= self.requested_max_new_tokens
        ):
            raise ValueError("expected output tokens must fit the generation budget")


@dataclass(frozen=True)
class NetworkFit:
    kind: str  # none, loopback, or controlled
    source: str | None
    bucket_id: str | None
    transport_ms: float
    intercept_ms: float
    rtt_coefficient: float
    uplink_min_bps: float | None
    uplink_max_bps: float | None
    downlink_min_bps: float | None
    downlink_max_bps: float | None
    rtt_min_ms: float | None
    rtt_max_ms: float | None

    def __post_init__(self) -> None:
        if self.kind not in {"none", "loopback", "controlled"}:
            raise ValueError("unsupported network fit kind")
        for value in (self.transport_ms, self.intercept_ms, self.rtt_coefficient):
            if not math.isfinite(value):
                raise ValueError("network fit coefficients must be finite")
        if self.transport_ms < 0 or self.rtt_coefficient < 0:
            raise ValueError("network fit has a negative duration or RTT coefficient")
        if self.kind == "loopback" and (not self.source or not self.bucket_id):
            raise ValueError("loopback fit requires source and bucket")
        if self.kind == "controlled":
            ranges = (
                (self.uplink_min_bps, self.uplink_max_bps),
                (self.downlink_min_bps, self.downlink_max_bps),
                (self.rtt_min_ms, self.rtt_max_ms),
            )
            if any(
                low is None
                or high is None
                or not math.isfinite(low)
                or not math.isfinite(high)
                or low < 0
                or low > high
                for low, high in ranges
            ):
                raise ValueError("controlled network fit has invalid coverage bounds")
            if self.uplink_min_bps == 0 or self.downlink_min_bps == 0:
                raise ValueError("controlled network bandwidth must be positive")


@dataclass(frozen=True)
class CandidateFit:
    configuration_id: str
    placement_profile_sha256: str
    sample_count: int
    task_bucket: str
    image_bytes_min: int
    image_bytes_max: int
    image_width_min: int
    image_width_max: int
    image_height_min: int
    image_height_max: int
    prompt_tokens_min: int
    prompt_tokens_max: int
    requested_max_new_tokens: int
    expected_output_tokens: float
    edge_compute_ms: float
    client_preprocess_ms: float
    serialization_ms: float
    server_fixed_ms: float
    decode_ms_per_token: float
    client_residual_ms: float
    request_payload_bytes: float
    request_framing_bytes: float
    response_wire_bytes: float
    edge_memory_requirement_mib: float | None
    constant_median_ms: float
    interval_half_width_ms: float
    network: NetworkFit

    def __post_init__(self) -> None:
        if not self.configuration_id or not self.placement_profile_sha256:
            raise ValueError("candidate fit identifiers must be non-empty")
        if self.sample_count < 1 or self.requested_max_new_tokens < 1:
            raise ValueError("candidate fit needs samples and a generation budget")
        ranges = (
            (self.image_bytes_min, self.image_bytes_max),
            (self.image_width_min, self.image_width_max),
            (self.image_height_min, self.image_height_max),
            (self.prompt_tokens_min, self.prompt_tokens_max),
        )
        if any(low <= 0 or high < low for low, high in ranges):
            raise ValueError("candidate fit has invalid task coverage bounds")
        durations = (
            self.edge_compute_ms,
            self.client_preprocess_ms,
            self.serialization_ms,
            self.server_fixed_ms,
            self.decode_ms_per_token,
            self.client_residual_ms,
            self.constant_median_ms,
            self.interval_half_width_ms,
        )
        if any(not math.isfinite(value) for value in durations):
            raise ValueError("candidate fit durations must be finite")
        nonnegative = (
            self.edge_compute_ms,
            self.client_preprocess_ms,
            self.serialization_ms,
            self.server_fixed_ms,
            self.decode_ms_per_token,
            self.constant_median_ms,
            self.interval_half_width_ms,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("candidate fit has a negative compute duration")
        sizes = (
            self.request_payload_bytes,
            self.request_framing_bytes,
            self.response_wire_bytes,
        )
        if any(not math.isfinite(value) or value < 0 for value in sizes):
            raise ValueError("candidate fit byte sizes must be finite and non-negative")
        if not 0 <= self.expected_output_tokens <= self.requested_max_new_tokens:
            raise ValueError("fitted output estimate exceeds generation budget")
        if self.edge_memory_requirement_mib is not None and (
            not math.isfinite(self.edge_memory_requirement_mib)
            or self.edge_memory_requirement_mib <= 0
        ):
            raise ValueError("edge memory requirement must be positive when known")


@dataclass(frozen=True)
class Calibration:
    schema_version: int
    calibration_id: str
    source_profile_sha256: str
    split_specification_sha256: str
    model_contract_sha256: str
    dataset_manifest_sha256: str
    host_pair_id: str
    training_request_ids: tuple[str, ...]
    holdout_request_ids: tuple[str, ...]
    holdout_network_bucket_ids: tuple[str, ...]
    candidates: dict[str, CandidateFit]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Calibration:
        if value.get("schema_version") != 1:
            raise ValueError("unsupported calibration schema")
        expected_id = hashlib.sha256(
            json.dumps(
                {key: item for key, item in value.items() if key != "calibration_id"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if value.get("calibration_id") != expected_id:
            raise ValueError("calibration content hash differs from identifier")
        candidates = {
            name: CandidateFit(**{**raw, "network": NetworkFit(**raw["network"])})
            for name, raw in value["candidates"].items()
        }
        return cls(
            schema_version=1,
            calibration_id=value["calibration_id"],
            source_profile_sha256=value["source_profile_sha256"],
            split_specification_sha256=value["split_specification_sha256"],
            model_contract_sha256=value["model_contract_sha256"],
            dataset_manifest_sha256=value["dataset_manifest_sha256"],
            host_pair_id=value["host_pair_id"],
            training_request_ids=tuple(value["training_request_ids"]),
            holdout_request_ids=tuple(value["holdout_request_ids"]),
            holdout_network_bucket_ids=tuple(value["holdout_network_bucket_ids"]),
            candidates=candidates,
        )

    @classmethod
    def load(cls, path: Path) -> Calibration:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("calibration must be a JSON object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class Prediction:
    configuration_id: str
    valid: bool
    reason: str | None
    coverage: str
    calibration_id: str
    estimated_output_tokens: float | None
    output_estimate_source: str | None
    components_ms: dict[str, float]
    total_ms: float | None
    lower_ms: float | None
    upper_ms: float | None
    request_payload_bytes: float | None
    request_wire_bytes: float | None
    response_wire_bytes: float | None
    edge_memory_requirement_mib: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Predictor:
    def __init__(self, calibration: Calibration) -> None:
        if calibration.schema_version != 1:
            raise ValueError("unsupported calibration schema")
        self.calibration = calibration

    def _invalid(self, configuration_id: str, reason: str) -> Prediction:
        return Prediction(
            configuration_id=configuration_id,
            valid=False,
            reason=reason,
            coverage="out_of_coverage",
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
        self, spec: ConfigurationSpec, state: PredictionState, task: PredictionTask
    ) -> Prediction:
        identifier = spec.id.value
        fit = self.calibration.candidates.get(identifier)
        if fit is None:
            return self._invalid(identifier, "configuration_not_calibrated")
        if state.host_pair_id != self.calibration.host_pair_id:
            return self._invalid(identifier, "host_pair_mismatch")
        if state.model_contract_sha256 != self.calibration.model_contract_sha256:
            return self._invalid(identifier, "model_contract_mismatch")
        if state.placement_profile_sha256 != fit.placement_profile_sha256:
            return self._invalid(identifier, "placement_profile_mismatch")
        if task.requested_max_new_tokens != fit.requested_max_new_tokens:
            return self._invalid(identifier, "generation_contract_mismatch")
        if task.task_bucket != fit.task_bucket:
            return self._invalid(identifier, "task_bucket_mismatch")
        if task.prompt_tokens is None:
            return self._invalid(identifier, "prompt_tokens_unknown")
        bounds = (
            (task.image_bytes, fit.image_bytes_min, fit.image_bytes_max),
            (task.image_width, fit.image_width_min, fit.image_width_max),
            (task.image_height, fit.image_height_min, fit.image_height_max),
            (task.prompt_tokens, fit.prompt_tokens_min, fit.prompt_tokens_max),
        )
        if any(value < low or value > high for value, low, high in bounds):
            return self._invalid(identifier, "task_out_of_coverage")
        if state.edge_memory_budget_mib is not None and spec.mode.value != "cloud":
            required = fit.edge_memory_requirement_mib
            if required is None:
                return self._invalid(identifier, "edge_memory_requirement_unknown")
            if required > state.edge_memory_budget_mib:
                return self._invalid(identifier, "edge_memory_budget_exceeded")

        output_tokens = (
            float(task.expected_output_tokens)
            if task.expected_output_tokens is not None
            else fit.expected_output_tokens
        )
        output_source = (
            "task_estimate"
            if task.expected_output_tokens is not None
            else "training_median"
        )
        request_payload = (
            float(task.image_bytes)
            if identifier == "cloud/raw_image"
            else fit.request_payload_bytes
        )
        request_wire = (
            request_payload + fit.request_framing_bytes
            if spec.mode.value != "edge"
            else 0.0
        )
        response_wire = fit.response_wire_bytes if spec.mode.value != "edge" else 0.0
        network = fit.network
        if network.kind == "none":
            transport = 0.0
        elif network.kind == "loopback":
            if (
                state.network_bucket_id != network.bucket_id
                or state.network_source != network.source
                or any(
                    value is not None
                    for value in (state.uplink_bps, state.downlink_bps, state.rtt_ms)
                )
            ):
                return self._invalid(identifier, "network_out_of_coverage")
            transport = network.transport_ms
        elif network.kind == "controlled":
            if network.source is not None and state.network_source != network.source:
                return self._invalid(identifier, "network_source_mismatch")
            values = (state.uplink_bps, state.downlink_bps, state.rtt_ms)
            if any(value is None for value in values):
                return self._invalid(identifier, "network_state_missing")
            up, down, rtt = values
            assert up is not None and down is not None and rtt is not None
            if not (
                network.uplink_min_bps is not None
                and network.uplink_max_bps is not None
                and network.downlink_min_bps is not None
                and network.downlink_max_bps is not None
                and network.rtt_min_ms is not None
                and network.rtt_max_ms is not None
                and network.uplink_min_bps <= up <= network.uplink_max_bps
                and network.downlink_min_bps <= down <= network.downlink_max_bps
                and network.rtt_min_ms <= rtt <= network.rtt_max_ms
            ):
                return self._invalid(identifier, "network_out_of_coverage")
            transport = (
                network.intercept_ms
                + network.rtt_coefficient * rtt
                + transfer_ms(request_wire, up)
                + transfer_ms(response_wire, down)
            )
            if transport < 0:
                return self._invalid(identifier, "negative_transport_prediction")
        else:
            return self._invalid(identifier, "unsupported_network_fit")
        components = {
            "edge_compute_ms": fit.edge_compute_ms,
            "client_preprocess_ms": fit.client_preprocess_ms,
            "serialization_ms": fit.serialization_ms,
            "server_fixed_ms": fit.server_fixed_ms,
            "decode_ms": fit.decode_ms_per_token * output_tokens,
            "transport_ms": transport,
            "client_residual_ms": fit.client_residual_ms,
        }
        total = sum(components.values())
        if not math.isfinite(total) or total <= 0:
            return self._invalid(identifier, "invalid_latency_prediction")
        return Prediction(
            configuration_id=identifier,
            valid=True,
            reason=None,
            coverage="in_distribution",
            calibration_id=self.calibration.calibration_id,
            estimated_output_tokens=output_tokens,
            output_estimate_source=output_source,
            components_ms=components,
            total_ms=total,
            lower_ms=max(0.0, total - fit.interval_half_width_ms),
            upper_ms=total + fit.interval_half_width_ms,
            request_payload_bytes=request_payload,
            request_wire_bytes=request_wire,
            response_wire_bytes=response_wire,
            edge_memory_requirement_mib=fit.edge_memory_requirement_mib,
        )
