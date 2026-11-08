from __future__ import annotations

import pytest

from adaptive_vlm import timing
from adaptive_vlm.measurement import measure_call


def test_raw_elapsed_time_ignores_backwards_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_RAW_CLOCK_ID", 4)
    ticks = iter((1_000_000_000, 1_025_000_000))
    wall = iter((1000.0, 997.3))

    def raw(clock_id: int) -> int:
        assert clock_id == 4
        return next(ticks)

    monkeypatch.setattr(timing.time, "clock_gettime_ns", raw)
    monkeypatch.setattr(timing.time, "time", lambda: next(wall))

    def operation() -> float:
        started = timing.time.time()
        return timing.time.time() - started

    result = measure_call(operation, clock=timing.perf_counter_ns)
    assert result.elapsed_ms == 25.0
    assert result.value < -2.6
    assert timing.clock_metadata()["uses_raw_clock"] is True


def test_non_raw_fallback_and_seconds_share_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_RAW_CLOCK_ID", None)
    monkeypatch.setattr(timing.time, "perf_counter_ns", lambda: 1_234_500_000)
    assert timing.perf_counter_ns() == 1_234_500_000
    assert timing.perf_counter() == 1.2345
    assert timing.monotonic() == 1.2345
    assert timing.clock_metadata()["name"] == "perf_counter"


def test_environment_records_duration_clock() -> None:
    from adaptive_vlm.environment import collect_environment

    assert collect_environment()["duration_clock"] == timing.clock_metadata()
