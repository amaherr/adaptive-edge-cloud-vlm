"""Small measurement primitives shared by real execution policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TimedValue(Generic[T]):
    value: T
    elapsed_ms: float


def measure_call(
    operation: Callable[[], T],
    *,
    synchronize: Callable[[], None] | None = None,
    clock: Callable[[], int] = perf_counter_ns,
) -> TimedValue[T]:
    """Measure an operation, synchronizing asynchronous devices at both edges."""

    if synchronize is not None:
        synchronize()
    started_ns = clock()
    value = operation()
    if synchronize is not None:
        synchronize()
    elapsed_ms = (clock() - started_ns) / 1_000_000
    return TimedValue(value=value, elapsed_ms=elapsed_ms)
