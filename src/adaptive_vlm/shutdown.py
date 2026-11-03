"""Finish the active inference and durable local append after SIGTERM."""

from collections.abc import Iterator
from contextlib import contextmanager
from signal import SIGTERM, signal
from threading import Event


@contextmanager
def graceful_sigterm() -> Iterator[Event]:
    """Restore the previous handler after a controlled fixed-policy run."""
    stop = Event()
    previous = signal(SIGTERM, lambda _signal, _frame: stop.set())
    try:
        yield stop
    finally:
        signal(SIGTERM, previous)
