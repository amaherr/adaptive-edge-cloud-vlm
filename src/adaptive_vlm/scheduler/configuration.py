"""Stable configuration IDs and current executor availability."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from adaptive_vlm.config import ExecutionMode
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.remote import RemoteCapabilities


class ConfigurationId(StrEnum):
    EDGE_FULL = "edge/full"
    CLOUD_RAW_IMAGE = "cloud/raw_image"
    DISTRIBUTED_PROJECTED = "distributed/projected/synchronous"


@dataclass(frozen=True)
class ConfigurationSpec:
    id: ConfigurationId
    mode: ExecutionMode
    split_id: str | None
    pipeline: str | None
    remote_path: str | None


CONFIGURATIONS = (
    ConfigurationSpec(ConfigurationId.EDGE_FULL, ExecutionMode.EDGE, None, None, None),
    ConfigurationSpec(
        ConfigurationId.CLOUD_RAW_IMAGE,
        ExecutionMode.CLOUD,
        None,
        None,
        "raw_image",
    ),
    ConfigurationSpec(
        ConfigurationId.DISTRIBUTED_PROJECTED,
        ExecutionMode.DISTRIBUTED,
        "projected",
        "synchronous",
        "projected_synchronous",
    ),
)


@dataclass(frozen=True)
class CapabilityStatus:
    configuration_id: ConfigurationId
    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available == (self.reason is not None):
            raise ValueError(
                "available status must have no reason; unavailable needs one"
            )

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "configuration_id": self.configuration_id.value,
            "available": self.available,
            "reason": self.reason,
        }


class CapabilityRegistry:
    """Validate loaded policies against one server and optional memory budget."""

    def __init__(
        self,
        *,
        policies: Mapping[ConfigurationId, ExecutionPolicy],
        expected_model_fingerprint: str,
        remote: RemoteCapabilities | None,
        remote_error: str | None = None,
        edge_resident_memory_mib: float | None = None,
        edge_required_peak_mib: float | None = None,
        edge_memory_budget_mib: float | None = None,
    ) -> None:
        if len(expected_model_fingerprint) != 64:
            raise ValueError("model fingerprint must be a SHA-256 digest")
        if remote is not None and remote_error is not None:
            raise ValueError("remote result and error are mutually exclusive")
        for value in (
            edge_resident_memory_mib,
            edge_required_peak_mib,
            edge_memory_budget_mib,
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("memory values must be finite and non-negative")
        self.policies = dict(policies)
        self.expected_model_fingerprint = expected_model_fingerprint
        self.remote = remote
        self.remote_error = remote_error
        self.edge_resident_memory_mib = edge_resident_memory_mib
        self.edge_required_peak_mib = edge_required_peak_mib
        self.edge_memory_budget_mib = edge_memory_budget_mib

    def status(self, spec: ConfigurationSpec) -> CapabilityStatus:
        policy = self.policies.get(spec.id)
        if policy is None:
            return CapabilityStatus(spec.id, False, "policy_not_loaded")
        if policy.mode is not spec.mode:
            return CapabilityStatus(spec.id, False, "policy_mode_mismatch")
        if self.edge_memory_budget_mib is not None:
            if self.edge_required_peak_mib is None:
                return CapabilityStatus(
                    spec.id, False, "edge_memory_requirement_unknown"
                )
            if self.edge_required_peak_mib > self.edge_memory_budget_mib:
                return CapabilityStatus(spec.id, False, "edge_memory_budget_exceeded")
        if spec.remote_path is not None:
            if self.remote is None:
                return CapabilityStatus(
                    spec.id, False,
                    f"remote_unavailable: {self.remote_error or 'not_checked'}",
                )
            if self.remote.model_fingerprint != self.expected_model_fingerprint:
                return CapabilityStatus(spec.id, False, "remote_model_mismatch")
            if not self.remote.ready:
                return CapabilityStatus(spec.id, False, "remote_not_ready")
            if spec.remote_path not in self.remote.paths:
                return CapabilityStatus(spec.id, False, "remote_path_missing")
        return CapabilityStatus(spec.id, True)

    def statuses(self) -> tuple[CapabilityStatus, ...]:
        return tuple(self.status(spec) for spec in CONFIGURATIONS)

    def policy_for(self, configuration_id: ConfigurationId) -> ExecutionPolicy:
        spec = next(spec for spec in CONFIGURATIONS if spec.id is configuration_id)
        status = self.status(spec)
        if not status.available:
            raise ValueError(f"{configuration_id.value} unavailable: {status.reason}")
        return self.policies[configuration_id]
