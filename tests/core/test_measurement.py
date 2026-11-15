from __future__ import annotations

from collections.abc import Iterator

from adaptive_vlm.measurement import measure_call


def test_measure_call_returns_value_and_elapsed_time() -> None:
    times = iter((1_000_000, 3_500_000))

    measured = measure_call(lambda: "result", clock=lambda: next(times))

    assert measured.value == "result"
    assert measured.elapsed_ms == 2.5


def test_measure_call_synchronizes_both_boundaries() -> None:
    events: list[str] = []
    times: Iterator[int] = iter((0, 1_000_000))

    measured = measure_call(
        lambda: events.append("operation"),
        synchronize=lambda: events.append("synchronize"),
        clock=lambda: next(times),
    )

    assert measured.value is None
    assert measured.elapsed_ms == 1.0
    assert events == ["synchronize", "operation", "synchronize"]
