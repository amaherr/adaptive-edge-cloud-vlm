"""Bounded application health probing and diagnostic passive send estimates."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import perf_counter_ns
from typing import Any

from adaptive_vlm.config import TransportConfig
from adaptive_vlm.execution.remote import (
    HealthStatus,
    RemoteCapabilities,
    check_remote_capabilities,
    check_remote_health,
)
from adaptive_vlm.models.base import ModelMetadata

HealthProbe = Callable[[TransportConfig, ModelMetadata], HealthStatus]
CapabilityProbe = Callable[[TransportConfig, ModelMetadata], RemoteCapabilities]


@dataclass(frozen=True)
class MonitorConfig:
    sampling_interval_ms: float = 0.0
    max_age_ms: float = 5_000.0
    probe_timeout_ms: float = 2_000.0
    passive_alpha: float = 0.25
    passive_min_bytes: int = 4_096
    passive_min_bps: float = 100_000.0
    passive_max_bps: float = 10_000_000_000.0

    def __post_init__(self) -> None:
        values = (
            self.sampling_interval_ms,
            self.max_age_ms,
            self.probe_timeout_ms,
            self.passive_min_bps,
            self.passive_max_bps,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("monitor bounds must be finite and non-negative")
        if self.max_age_ms <= 0 or self.probe_timeout_ms <= 0:
            raise ValueError("monitor age and timeout must be positive")
        if not 0 < self.passive_alpha <= 1:
            raise ValueError("passive smoothing alpha must be in (0, 1]")
        if self.passive_min_bytes < 1 or self.passive_min_bps <= 0:
            raise ValueError("passive sample minimum must be positive")
        if self.passive_max_bps < self.passive_min_bps:
            raise ValueError("passive rate bounds are reversed")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sampling_interval_ms": self.sampling_interval_ms,
            "max_age_ms": self.max_age_ms,
            "probe_timeout_ms": self.probe_timeout_ms,
            "passive_alpha": self.passive_alpha,
            "passive_min_bytes": self.passive_min_bytes,
            "passive_min_bps": self.passive_min_bps,
            "passive_max_bps": self.passive_max_bps,
        }


@dataclass(frozen=True)
class MonitorSample:
    remote: RemoteCapabilities | None
    sampled_at_utc: str
    age_ms: float
    sampling_overhead_ms: float
    health_probe_rtt_ms: float | None
    capability_probe_ms: float | None
    health_ready: bool | None
    error: str | None
    cached: bool
    passive_uplink_application_bps: float | None
    passive_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "live_application_health",
            "sampled_at_utc": self.sampled_at_utc,
            "age_ms": self.age_ms,
            "sampling_overhead_ms": self.sampling_overhead_ms,
            "health_probe_rtt_ms": self.health_probe_rtt_ms,
            "health_ready": self.health_ready,
            "capability_probe_ms": self.capability_probe_ms,
            "remote_ready": self.remote.ready if self.remote else False,
            "remote_paths": sorted(self.remote.paths) if self.remote else [],
            "error": self.error,
            "cached": self.cached,
            "passive_uplink_application_bps": self.passive_uplink_application_bps,
            "passive_sample_count": self.passive_sample_count,
            "passive_source": "socket_send_duration_diagnostic_only",
            "downlink_goodput_bps": None,
            "downlink_null_reason": "receive_duration_includes_server_compute",
        }


class LiveMonitor:
    """Fail closed for remote eligibility; retain passive rates as diagnostics."""

    def __init__(
        self,
        transport: TransportConfig,
        model: ModelMetadata,
        config: MonitorConfig,
        *,
        health_probe: HealthProbe = check_remote_health,
        capability_probe: CapabilityProbe = check_remote_capabilities,
        clock_ns: Callable[[], int] = perf_counter_ns,
        utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.transport = transport
        self.model = model
        self.config = config
        self.health_probe = health_probe
        self.capability_probe = capability_probe
        self.clock_ns = clock_ns
        self.utc_now = utc_now
        self._last_sample: MonitorSample | None = None
        self._last_ns: int | None = None
        self._passive_uplink_bps: float | None = None
        self._passive_count = 0

    def sample(self) -> MonitorSample:
        now = self.clock_ns()
        if self._last_sample is not None and self._last_ns is not None:
            age_ms = max(0.0, (now - self._last_ns) / 1_000_000)
            if age_ms <= min(self.config.sampling_interval_ms, self.config.max_age_ms):
                return replace(
                    self._last_sample,
                    age_ms=age_ms,
                    sampling_overhead_ms=0.0,
                    cached=True,
                    passive_uplink_application_bps=self._passive_uplink_bps,
                    passive_sample_count=self._passive_count,
                )
        started = self.clock_ns()
        bounded = replace(
            self.transport,
            timeout_seconds=min(
                self.transport.timeout_seconds,
                self.config.probe_timeout_ms / 1_000,
            ),
        )
        remote = None
        health_ready: bool | None = None
        health_ms: float | None = None
        capability_ms: float | None = None
        error = None
        try:
            health_started = self.clock_ns()
            health = self.health_probe(bounded, self.model)
            health_ms = (self.clock_ns() - health_started) / 1_000_000
            health_ready = health.ready
            if health.model_fingerprint != self.model.fingerprint:
                raise ValueError("health model fingerprint mismatch")
            if health.ready:
                capability_started = self.clock_ns()
                remote = self.capability_probe(bounded, self.model)
                capability_ms = (self.clock_ns() - capability_started) / 1_000_000
                if (
                    not remote.ready
                    or remote.model_fingerprint != self.model.fingerprint
                ):
                    raise ValueError("remote capabilities unavailable or mismatched")
            else:
                error = "remote_not_ready"
        except (OSError, RuntimeError, ValueError) as exc:
            remote = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = (self.clock_ns() - started) / 1_000_000
        sampled_ns = self.clock_ns()
        sample = MonitorSample(
            remote=remote,
            sampled_at_utc=self.utc_now().isoformat().replace("+00:00", "Z"),
            age_ms=0.0,
            sampling_overhead_ms=elapsed,
            health_probe_rtt_ms=health_ms,
            capability_probe_ms=capability_ms,
            health_ready=health_ready,
            error=error,
            cached=False,
            passive_uplink_application_bps=self._passive_uplink_bps,
            passive_sample_count=self._passive_count,
        )
        self._last_sample = sample
        self._last_ns = sampled_ns
        return sample

    def observe_send(self, metrics: Mapping[str, Any]) -> bool:
        """Update a bounded EMA of application socket-send rate after inference."""
        byte_count = metrics.get("request_wire_bytes")
        send_ms = metrics.get("network_send_ms")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < self.config.passive_min_bytes
            or isinstance(send_ms, bool)
            or not isinstance(send_ms, (int, float))
            or not math.isfinite(send_ms)
            or send_ms <= 0
        ):
            return False
        raw_bps = 8_000 * byte_count / send_ms
        bounded = min(
            self.config.passive_max_bps,
            max(self.config.passive_min_bps, raw_bps),
        )
        self._passive_uplink_bps = (
            bounded
            if self._passive_uplink_bps is None
            else self.config.passive_alpha * bounded
            + (1 - self.config.passive_alpha) * self._passive_uplink_bps
        )
        self._passive_count += 1
        return True
