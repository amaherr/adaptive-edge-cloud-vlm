"""Host-local elapsed time, insulated from Linux NTP frequency corrections.

UTC timestamps remain useful for provenance, but are never duration clocks.
CLOCK_MONOTONIC_RAW is selected once so an interval cannot mix clock domains.
Other platforms retain Python's performance-counter implementation.
"""

from __future__ import annotations

import sys
import time

_RAW_CLOCK_ID: int | None = (
    getattr(time, "CLOCK_MONOTONIC_RAW", None)
    if sys.platform.startswith("linux")
    else None
)


def perf_counter_ns() -> int:
    """Return elapsed-time ticks in nanoseconds, with no UTC/epoch meaning."""
    if _RAW_CLOCK_ID is not None:
        return time.clock_gettime_ns(_RAW_CLOCK_ID)
    return time.perf_counter_ns()


def perf_counter() -> float:
    """Return seconds from the same clock used by perf_counter_ns."""
    return perf_counter_ns() / 1_000_000_000


def monotonic() -> float:
    """Use the same unadjusted clock for application age/deadline checks."""
    return perf_counter()


def clock_metadata() -> dict[str, str | bool]:
    """Identify the clock without implying cross-host timestamp comparability."""
    return {
        "name": "CLOCK_MONOTONIC_RAW" if _RAW_CLOCK_ID is not None else "perf_counter",
        "uses_raw_clock": _RAW_CLOCK_ID is not None,
        "scope": "host_local_elapsed_time",
    }
