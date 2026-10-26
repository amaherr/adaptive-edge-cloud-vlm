from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_vlm.benchmark.network_trace import replay_network_trace
from adaptive_vlm.config import TransportConfig
from adaptive_vlm.execution.remote import HealthStatus, RemoteCapabilities
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.scheduler.monitor import LiveMonitor, MonitorConfig
from adaptive_vlm.scheduler.predictor import Calibration

ROOT = Path(__file__).resolve().parents[1]


def test_controlled_trace_crossover_noise_and_fail_closed() -> None:
    calibration = Calibration.load(
        ROOT / "configs/network_traces/phase6_synthetic_calibration.json"
    )
    trace = json.loads(
        (ROOT / "configs/network_traces/phase6_synthetic_trace.json").read_text()
    )
    report, rows = replay_network_trace(calibration, trace)
    assert report["synthetic_trace_gate_passed"]
    assert report["switches"] == 3
    assert report["noisy_segment_switches_after_warmup"] == 0
    assert rows[0]["decision"]["selected_configuration_id"] == "edge/full"
    assert rows[1]["decision"]["selected_configuration_id"] == (
        "distributed/projected/synchronous"
    )
    assert rows[-3]["decision"]["selected_configuration_id"] == "edge/full"
    assert rows[-1]["decision"]["status"] == "no_feasible_candidate"
    assert rows[-1]["decision"]["candidates"][0]["exclusion_reason"] == (
        "edge_memory_budget_exceeded"
    )


def test_trace_expectation_miss_is_reported() -> None:
    calibration = Calibration.load(
        ROOT / "configs/network_traces/phase6_synthetic_calibration.json"
    )
    trace = json.loads(
        (ROOT / "configs/network_traces/phase6_synthetic_trace.json").read_text()
    )
    trace["samples"][1]["expected_configuration_id"] = "edge/full"
    report, _ = replay_network_trace(calibration, trace)
    assert not report["synthetic_trace_gate_passed"]
    assert report["expected_choice_mismatches"] == ["fast"]


def test_live_probe_caching_timeout_and_failure() -> None:
    ticks = [0]
    calls = [0]
    seen_timeouts: list[float] = []
    fingerprint = "a" * 64

    def clock_ns() -> int:
        return ticks[0]

    def health(transport: TransportConfig, _model: ModelMetadata) -> HealthStatus:
        seen_timeouts.append(transport.timeout_seconds)
        calls[0] += 1
        ticks[0] += 5_000_000
        if calls[0] == 2:
            raise TimeoutError("probe expired")
        return HealthStatus(True, fingerprint)

    def capabilities(
        _transport: TransportConfig, _model: ModelMetadata
    ) -> RemoteCapabilities:
        ticks[0] += 3_000_000
        return RemoteCapabilities(True, fingerprint, frozenset({"raw_image"}))

    monitor = LiveMonitor(
        TransportConfig("127.0.0.1", 5050, 120.0, 1),
        ModelMetadata("model", "v1", fingerprint),
        MonitorConfig(sampling_interval_ms=100, max_age_ms=150, probe_timeout_ms=20),
        health_probe=health,
        capability_probe=capabilities,
        clock_ns=clock_ns,
    )
    first = monitor.sample()
    assert first.remote is not None
    assert first.health_probe_rtt_ms == 5
    assert first.capability_probe_ms == 3
    assert first.sampling_overhead_ms == 8
    assert first.to_dict()["downlink_goodput_bps"] is None
    ticks[0] += 50_000_000
    cached = monitor.sample()
    assert cached.cached and cached.age_ms == 50
    assert cached.sampling_overhead_ms == 0
    assert calls[0] == 1
    ticks[0] += 120_000_000
    failed = monitor.sample()
    assert failed.remote is None
    assert failed.error is not None and "TimeoutError" in failed.error
    assert seen_timeouts == [0.02, 0.02]


def test_passive_send_estimate_is_bounded_and_not_downlink_goodput() -> None:
    monitor = LiveMonitor(
        TransportConfig("127.0.0.1", 5050, 120.0, 1),
        ModelMetadata("model", "v1", "a" * 64),
        MonitorConfig(
            passive_alpha=0.5,
            passive_min_bytes=4_096,
            passive_min_bps=1_000_000,
            passive_max_bps=20_000_000,
        ),
        health_probe=lambda _transport, _model: HealthStatus(False, "a" * 64),
    )
    assert not monitor.observe_send({"request_wire_bytes": 100, "network_send_ms": 1})
    assert monitor.observe_send({"request_wire_bytes": 8_000, "network_send_ms": 4})
    assert monitor.observe_send({"request_wire_bytes": 8_000, "network_send_ms": 8})
    sample = monitor.sample()
    assert sample.passive_uplink_application_bps == pytest.approx(12_000_000)
    assert sample.passive_sample_count == 2
    assert sample.to_dict()["downlink_null_reason"] == (
        "receive_duration_includes_server_compute"
    )
    assert not monitor.observe_send(
        {"request_wire_bytes": 8_000, "network_send_ms": float("nan")}
    )
