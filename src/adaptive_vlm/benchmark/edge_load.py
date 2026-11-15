"""Edge-load levels and the supervised load-generator process (protocol section 6).

A level fixes the competitor's GPU working set and its busy/idle cycle. The
controller starts the worker (``edge_load_worker``) in its own session, waits
until its heartbeat shows the working set allocated, checks it before every
request, and stops it at the end, verifying that it exited.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adaptive_vlm.contracts import MetricValue

NO_LOAD = "L0"
WORKER_MODULE = "adaptive_vlm.benchmark.edge_load_worker"


class LoadGeneratorFailure(RuntimeError):
    """The edge load generator is not running as configured."""


@dataclass(frozen=True)
class LoadLevel:
    name: str
    working_set_mib: int
    busy_ms: float
    idle_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_levels(path: Path) -> dict[str, LoadLevel]:
    """Read ``{"schema_version": 1, "levels": {"L1": {...}, ...}}``."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("levels"), dict
    ):
        raise ValueError("edge-load levels need schema_version 1 and a levels map")
    levels: dict[str, LoadLevel] = {}
    for name, values in payload["levels"].items():
        if name == NO_LOAD:
            raise ValueError("L0 means no load generator and must not be configured")
        level = LoadLevel(
            name=name,
            working_set_mib=int(values["working_set_mib"]),
            busy_ms=float(values["busy_ms"]),
            idle_ms=float(values["idle_ms"]),
        )
        if level.working_set_mib < 1 or level.busy_ms < 0 or level.idle_ms < 0:
            raise ValueError(f"invalid edge-load level {name}")
        levels[name] = level
    return levels


class EdgeLoadGenerator:
    """Start, verify, snapshot and stop one load-generator process."""

    def __init__(
        self,
        level: LoadLevel,
        *,
        directory: Path,
        device: str = "cuda:0",
        command_prefix: Sequence[str] | None = None,
        start_timeout_s: float = 60.0,
        stale_after_s: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.level = level
        self._directory = directory
        self.heartbeat_path = directory / "heartbeat.json"
        self._command = [
            *(command_prefix or [sys.executable, "-m", WORKER_MODULE]),
            "--working-set-mib",
            str(level.working_set_mib),
            "--busy-ms",
            str(level.busy_ms),
            "--idle-ms",
            str(level.idle_ms),
            "--heartbeat",
            str(self.heartbeat_path),
            "--device",
            device,
            "--level",
            level.name,
        ]
        self._start_timeout = start_timeout_s
        self._stale_after = stale_after_s
        self._clock = clock
        self._process: subprocess.Popen[bytes] | None = None

    def heartbeat(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.heartbeat_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def start(self) -> dict[str, Any]:
        self._directory.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.unlink(missing_ok=True)
        with (self._directory / "worker.log").open("ab") as log:
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + self._start_timeout
        while time.monotonic() < deadline:
            beat = self.heartbeat()
            if (
                beat is not None
                and beat.get("status") == "running"
                and float(beat.get("allocated_mib", 0))
                >= 0.95 * self.level.working_set_mib
            ):
                return beat
            if self._process.poll() is not None:
                break
            time.sleep(0.2)
        self.stop()
        raise LoadGeneratorFailure(
            f"edge load {self.level.name} did not start; see "
            f"{self._directory / 'worker.log'}"
        )

    def check(self) -> None:
        """Raise unless the worker process is alive with a fresh heartbeat."""

        if self._process is None or self._process.poll() is not None:
            raise LoadGeneratorFailure(f"edge load {self.level.name} is not running")
        beat = self.heartbeat()
        if beat is None or beat.get("status") != "running":
            raise LoadGeneratorFailure(f"edge load {self.level.name} has no heartbeat")
        age = self._clock() - float(beat["updated_unix_s"])
        if age > self._stale_after:
            raise LoadGeneratorFailure(
                f"edge load {self.level.name} heartbeat is {age:.1f}s old"
            )

    def snapshot(self) -> dict[str, MetricValue]:
        beat = self.heartbeat() or {}
        alive = self._process is not None and self._process.poll() is None
        updated = beat.get("updated_unix_s")
        return {
            "load_level": self.level.name,
            "load_alive": alive,
            "load_heartbeat_age_s": (
                self._clock() - float(updated) if updated is not None else None
            ),
            "load_duty": beat.get("duty"),
            "load_kernels": beat.get("kernels"),
            "load_allocated_mib": beat.get("allocated_mib"),
        }

    def stop(self) -> dict[str, Any] | None:
        process = self._process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        if process is not None and process.poll() is None:
            raise LoadGeneratorFailure(f"edge load {self.level.name} did not stop")
        self._process = None
        final = self.heartbeat()
        if process is not None and final is not None:
            with (self._directory / "sessions.jsonl").open(
                "a", encoding="utf-8"
            ) as log:
                log.write(json.dumps(final, sort_keys=True) + "\n")
        return final

    def __enter__(self) -> EdgeLoadGenerator:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def level_conditions(level: LoadLevel | None) -> Mapping[str, Any]:
    """Plan entry recording the load condition a sub-block ran under."""

    return {
        "edge_load_level": NO_LOAD if level is None else level.name,
        "edge_load": None if level is None else level.to_dict(),
    }
