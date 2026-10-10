"""Human-readable CLI progress that does not contaminate JSON stdout."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import TextIO


@dataclass(frozen=True)
class ProgressReporter:
    verbose: bool = False
    stream: TextIO | None = None

    def info(self, message: str) -> None:
        print(
            f"[adaptive-vlm] {message}",
            file=self.stream or sys.stderr,
            flush=True,
        )

    def detail(self, message: str) -> None:
        if self.verbose:
            self.info(message)

    @contextmanager
    def phase(self, message: str) -> Iterator[None]:
        self.info(f"{message}...")
        started = perf_counter()
        try:
            yield
        except Exception:
            elapsed = perf_counter() - started
            self.info(f"{message} failed after {elapsed:.1f}s")
            raise
        elapsed = perf_counter() - started
        self.info(f"{message} completed in {elapsed:.1f}s")
