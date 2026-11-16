"""Process lifetime helpers for unattended runs (Linux; no-ops elsewhere).

An unattended campaign must never leave a GPU load generator or a benchmark
running after the process that owns it has died: an orphaned load generator
would silently load every later "no load" sub-block.
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

_PR_SET_PDEATHSIG = 1


def exit_with_parent(sig: int = signal.SIGTERM) -> bool:
    """Ask the kernel to send ``sig`` to this process when its parent dies.

    If the parent already died before the request (re-parented to init),
    exit immediately. Returns False where ``prctl`` is unavailable.
    """

    if not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0) != 0:
        return False
    if os.getppid() == 1:
        os.kill(os.getpid(), sig)
    return True


def terminate_on_sigterm() -> None:
    """Turn SIGTERM into KeyboardInterrupt so ``finally`` blocks run."""

    def handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, handler)


def find_processes(marker: str, proc: Path = Path("/proc")) -> list[int]:
    """PIDs (other than this one) whose command line contains ``marker``."""

    found = []
    for entry in proc.glob("[0-9]*"):
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        pid = int(entry.name)
        if pid != os.getpid() and marker.encode() in command:
            found.append(pid)
    return sorted(found)


@contextmanager
def exclusive_lock(path: Path, purpose: str) -> Iterator[TextIO]:
    """Hold a non-blocking exclusive ``flock`` for the duration of the block."""

    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another process holds the {purpose} lock") from None
        yield handle
