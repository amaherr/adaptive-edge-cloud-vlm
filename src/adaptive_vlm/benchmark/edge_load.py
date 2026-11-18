"""Edge-load levels and the supervised load generators (protocol section 6).

A level fixes the competitor's GPU working set and its busy/idle cycle. A
levels file picks one of two generators for all its levels:

- ``process`` (G2 v2, the default): the worker (``edge_load_worker``) runs in
  its own session. The controller waits until its heartbeat shows the working
  set allocated, checks it before every request, and stops it at the end,
  verifying that it exited.
- ``in_process`` (amendment A2a-3, AWS edge): the same busy/idle cycle runs in
  a thread of the benchmark process, on its own CUDA stream, sharing the
  model's CUDA context and caching allocator. A second CUDA context does not
  fit beside the resident model on the 6 GiB vGPU.

Both expose ``start``, ``check``, ``snapshot`` and ``stop`` with the same
snapshot fields, so records and validity gates do not depend on the choice.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from adaptive_vlm.contracts import MetricValue

NO_LOAD = "L0"
WORKER_MODULE = "adaptive_vlm.benchmark.edge_load_worker"
PROCESS = "process"
IN_PROCESS = "in_process"
GENERATORS = (PROCESS, IN_PROCESS)
_MIB = 1024 * 1024


class LoadGeneratorFailure(RuntimeError):
    """The edge load generator is not running as configured."""


@dataclass(frozen=True)
class LoadLevel:
    name: str
    working_set_mib: int
    busy_ms: float
    idle_ms: float
    generator: str = PROCESS
    # In-process only: split each pass over the working set into kernels of
    # this size, so the model's kernels interleave with the load (A2a-3).
    kernel_mib: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_levels(path: Path, *, no_load: str = NO_LOAD) -> dict[str, LoadLevel]:
    """Read ``{"schema_version": 1, "generator": ..., "levels": {"L1": {...}}}``.

    ``generator`` is ``process`` (default) or ``in_process``. Cloud-load
    levels use the same format with ``no_load="C0"``; the cloud worker is
    always a separate process.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("levels"), dict
    ):
        raise ValueError("edge-load levels need schema_version 1 and a levels map")
    generator = payload.get("generator", PROCESS)
    if generator not in GENERATORS:
        raise ValueError(f"load generator must be one of {GENERATORS}")
    if generator != PROCESS and no_load != NO_LOAD:
        raise ValueError("only edge load can run inside the benchmark process")
    levels: dict[str, LoadLevel] = {}
    for name, values in payload["levels"].items():
        if name == no_load:
            raise ValueError(
                f"{no_load} means no load generator and must not be configured"
            )
        level = LoadLevel(
            name=name,
            working_set_mib=int(values["working_set_mib"]),
            busy_ms=float(values["busy_ms"]),
            idle_ms=float(values["idle_ms"]),
            generator=generator,
            kernel_mib=(int(values["kernel_mib"]) if "kernel_mib" in values else None),
        )
        if level.working_set_mib < 1 or level.busy_ms < 0 or level.idle_ms < 0:
            raise ValueError(f"invalid edge-load level {name}")
        if level.kernel_mib is not None and (
            generator != IN_PROCESS
            or not 1 <= level.kernel_mib <= level.working_set_mib
        ):
            raise ValueError(
                f"edge-load level {name}: kernel_mib needs the in-process "
                "generator and must be between 1 and the working set"
            )
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


class Workload(Protocol):
    """GPU work for the in-process generator: one kernel per ``step``."""

    allocated_mib: float

    def step(self) -> None: ...

    def release(self) -> None: ...


class TorchWorkload:
    """In-place additions over a float32 working set on a dedicated stream.

    The buffer is allocated through the model's caching allocator, so it adds
    only its own size. ``step`` is one pass over the working set: one addition
    per ``kernel_mib`` slice (one for the whole buffer by default) on the side
    stream, then a wait for that stream alone, which releases the GIL. The
    model's stream is never synchronized by the load. Small slices let the
    GPU run the model's kernels between them; one whole-buffer kernel on a
    fractional vGPU lasts about 20 ms per 512 MiB and starves decoding.
    """

    def __init__(self, level: LoadLevel, device: str) -> None:
        torch: Any = importlib.import_module("torch")  # Model environment only.
        self._torch = torch
        self._stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self._stream):
            self._buffer: Any = torch.zeros(
                level.working_set_mib * _MIB // 4, dtype=torch.float32, device=device
            )
            self._buffer.add_(1.0)  # Touch once so the working set is resident.
        self._stream.synchronize()
        self.allocated_mib = self._buffer.numel() * 4 / _MIB
        self._slices: list[Any] = (
            list(self._buffer.split(level.kernel_mib * _MIB // 4))
            if level.kernel_mib is not None
            else [self._buffer]
        )

    def step(self) -> None:
        with self._torch.cuda.stream(self._stream):
            for part in self._slices:
                part.add_(1.0)
        self._stream.synchronize()

    def release(self) -> None:
        self._stream.synchronize()
        self._slices = []
        self._buffer = None


class InProcessEdgeLoad:
    """Run a level's busy/idle cycle in a thread of this process (A2a-3).

    The thread repeats what ``edge_load_worker`` does: run passes over the
    working set for ``busy_ms`` (each followed by a wait on its own stream;
    a pass is several kernels if the level sets ``kernel_mib``), then sleep
    ``idle_ms``. Its state is the heartbeat, refreshed every ``heartbeat_s``.
    A thread that dies, or a heartbeat older than ``stale_after_s``, fails
    ``check`` with :class:`LoadGeneratorFailure`, like a dead worker process.
    """

    def __init__(
        self,
        level: LoadLevel,
        *,
        directory: Path,
        device: str = "cuda:0",
        workload: Callable[[LoadLevel, str], Workload] = TorchWorkload,
        start_timeout_s: float = 60.0,
        stale_after_s: float = 5.0,
        heartbeat_s: float = 1.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.level = level
        self._directory = directory
        self._device = device
        self._workload = workload
        self._start_timeout = start_timeout_s
        self._stale_after = stale_after_s
        self._heartbeat_s = heartbeat_s
        self._clock = clock
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {}

    def heartbeat(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._state) if self._state else None

    def _publish(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def _run(self) -> None:
        level = self.level
        try:
            work = self._workload(level, self._device)
        except BaseException as exc:
            self._publish(status="failed", error=f"{type(exc).__name__}: {exc}")
            self._ready.set()
            return
        started = self._clock()
        busy_s = 0.0
        kernels = 0
        self._publish(
            status="running",
            generator=IN_PROCESS,
            pid=os.getpid(),
            level=level.name,
            working_set_mib=level.working_set_mib,
            busy_ms=level.busy_ms,
            idle_ms=level.idle_ms,
            kernel_mib=level.kernel_mib,
            allocated_mib=work.allocated_mib,
            started_unix_s=started,
            updated_unix_s=started,
            kernels=0,
            busy_s=0.0,
            duty=0.0,
        )
        self._ready.set()
        next_beat = 0.0
        status = "stopped"
        error: str | None = None
        try:
            while not self._stopping.is_set():
                cycle = time.perf_counter()
                if level.busy_ms > 0:
                    while (time.perf_counter() - cycle) * 1000 < level.busy_ms:
                        work.step()
                        kernels += 1
                    busy_s += time.perf_counter() - cycle
                if level.idle_ms > 0:
                    self._stopping.wait(level.idle_ms / 1000)
                now = self._clock()
                if now >= next_beat:
                    wall = now - started
                    self._publish(
                        updated_unix_s=now,
                        kernels=kernels,
                        busy_s=busy_s,
                        wall_s=wall,
                        duty=busy_s / wall if wall > 0 else 0.0,
                    )
                    next_beat = now + self._heartbeat_s
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                work.release()
            except BaseException as exc:
                if error is None:
                    status, error = "failed", f"{type(exc).__name__}: {exc}"
            now = self._clock()
            final: dict[str, Any] = {
                "status": status,
                "updated_unix_s": now,
                "kernels": kernels,
                "busy_s": busy_s,
                "wall_s": now - started,
                "duty": busy_s / (now - started) if now > started else 0.0,
            }
            if error is not None:
                final["error"] = error
            self._publish(**final)

    def start(self) -> dict[str, Any]:
        self._directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._state = {}
        self._stopping.clear()
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"edge-load-{self.level.name}", daemon=True
        )
        self._thread.start()
        self._ready.wait(self._start_timeout)
        beat = self.heartbeat()
        if (
            beat is not None
            and beat.get("status") == "running"
            and float(beat.get("allocated_mib", 0)) >= 0.95 * self.level.working_set_mib
        ):
            return beat
        reason = (beat or {}).get("error", "no heartbeat")
        self.stop()
        raise LoadGeneratorFailure(
            f"edge load {self.level.name} did not start: {reason}"
        )

    def _alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def check(self) -> None:
        """Raise unless the load thread is alive with a fresh heartbeat."""

        beat = self.heartbeat()
        if not self._alive() or beat is None or beat.get("status") != "running":
            reason = (beat or {}).get("error")
            raise LoadGeneratorFailure(
                f"edge load {self.level.name} is not running"
                + (f": {reason}" if reason else "")
            )
        age = self._clock() - float(beat["updated_unix_s"])
        if age > self._stale_after:
            raise LoadGeneratorFailure(
                f"edge load {self.level.name} heartbeat is {age:.1f}s old"
            )

    def snapshot(self) -> dict[str, MetricValue]:
        beat = self.heartbeat() or {}
        updated = beat.get("updated_unix_s")
        return {
            "load_level": self.level.name,
            "load_alive": self._alive() and beat.get("status") == "running",
            "load_heartbeat_age_s": (
                self._clock() - float(updated) if updated is not None else None
            ),
            "load_duty": beat.get("duty"),
            "load_kernels": beat.get("kernels"),
            "load_allocated_mib": beat.get("allocated_mib"),
        }

    def stop(self) -> dict[str, Any] | None:
        thread = self._thread
        self._stopping.set()
        if thread is not None:
            thread.join(timeout=15)
            if thread.is_alive():
                raise LoadGeneratorFailure(f"edge load {self.level.name} did not stop")
        self._thread = None
        final = self.heartbeat()
        if thread is not None and final is not None:
            with (self._directory / "sessions.jsonl").open(
                "a", encoding="utf-8"
            ) as log:
                log.write(json.dumps(final, sort_keys=True) + "\n")
        return final

    def __enter__(self) -> InProcessEdgeLoad:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def make_generator(
    level: LoadLevel, *, directory: Path, device: str = "cuda:0"
) -> EdgeLoadGenerator | InProcessEdgeLoad:
    """The generator the level's file asks for."""

    if level.generator == IN_PROCESS:
        return InProcessEdgeLoad(level, directory=directory, device=device)
    return EdgeLoadGenerator(level, directory=directory, device=device)


def level_conditions(level: LoadLevel | None) -> Mapping[str, Any]:
    """Plan entry recording the load condition a sub-block ran under."""

    return {
        "edge_load_level": NO_LOAD if level is None else level.name,
        "edge_load": None if level is None else level.to_dict(),
    }
