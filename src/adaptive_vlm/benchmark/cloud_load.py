"""Cloud GPU load (C1): control commands run on the cloud, plus the server probe.

The C1 competitor is the edge-load worker (``edge_load_worker``) running on
the cloud GPU (amendment A2a rule 6). The supervisor sends one short control
command per sub-block through its cloud runner (SSM into the inference
container on AWS, SSH on the physical pair, a local shell in rehearsals)::

    python -m adaptive_vlm.benchmark.cloud_load start --level C1 \\
        --working-set-mib 512 --busy-ms 10 --idle-ms 10
    python -m adaptive_vlm.benchmark.cloud_load status | stop | reap

Each prints one JSON object. ``start`` first stops any leftover worker, then
launches a detached worker in its own session and waits until its heartbeat
shows the working set allocated. ``stop`` (and ``reap``, which tolerates
there being nothing to stop) sends SIGTERM, then SIGKILL after a grace period,
and fails unless no worker remains.

The inference server reads the same heartbeat file (``--load-heartbeat``) and
reports ``load_level``, ``load_alive``, ``load_heartbeat_age_s`` and
``load_duty`` with every response as ``server_state_*``, which the sub-block
gate checks (rule 7).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from adaptive_vlm.benchmark.edge_load import WORKER_MODULE
from adaptive_vlm.contracts import MetricValue
from adaptive_vlm.process import find_processes

NO_CLOUD_LOAD = "C0"
DEFAULT_HEARTBEAT = Path("/tmp/adaptive-vlm-cloud-load/heartbeat.json")
STOP_GRACE_S = 15.0


def _marker(heartbeat: Path) -> str:
    """Only a cloud-load worker carries this exact argument pair."""

    return f"--heartbeat {heartbeat} "


def read_heartbeat(heartbeat: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(heartbeat.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def heartbeat_source(
    heartbeat: Path, clock: Callable[[], float] = time.time
) -> Callable[[], dict[str, MetricValue]]:
    """Snapshot function for the inference server's resource-state sampler."""

    def snapshot() -> dict[str, MetricValue]:
        beat = read_heartbeat(heartbeat)
        if beat is None:
            return {
                "load_level": NO_CLOUD_LOAD,
                "load_alive": False,
                "load_heartbeat_age_s": None,
                "load_duty": None,
            }
        updated = beat.get("updated_unix_s")
        alive = beat.get("status") == "running" and _pid_alive(beat.get("pid"))
        return {
            "load_level": str(beat.get("level") or NO_CLOUD_LOAD),
            "load_alive": alive,
            "load_heartbeat_age_s": (
                clock() - float(updated) if isinstance(updated, (int, float)) else None
            ),
            "load_duty": beat.get("duty"),
        }

    return snapshot


def status(heartbeat: Path) -> dict[str, Any]:
    pids = find_processes(_marker(heartbeat))
    beat = read_heartbeat(heartbeat)
    updated = (beat or {}).get("updated_unix_s")
    return {
        "alive": bool(pids) and (beat or {}).get("status") == "running",
        "pids": pids,
        "heartbeat": beat,
        "heartbeat_age_s": (
            time.time() - float(updated) if isinstance(updated, (int, float)) else None
        ),
    }


def stop(heartbeat: Path, *, grace_s: float = STOP_GRACE_S) -> dict[str, Any]:
    marker = _marker(heartbeat)
    pids = find_processes(marker)
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and find_processes(marker):
        time.sleep(0.2)
    for pid in find_processes(marker):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and find_processes(marker):
        time.sleep(0.1)
    return {
        "stopped": pids,
        "remaining": find_processes(marker),
        "final_heartbeat": read_heartbeat(heartbeat),
    }


def start(
    heartbeat: Path,
    *,
    level: str,
    working_set_mib: int,
    busy_ms: float,
    idle_ms: float,
    device: str,
    timeout_s: float,
    worker: Sequence[str] | None = None,
) -> dict[str, Any]:
    if level == NO_CLOUD_LOAD:
        raise ValueError("C0 means no cloud load")
    leftovers = stop(heartbeat)
    if leftovers["remaining"]:
        raise RuntimeError(f"leftover cloud load would not stop: {leftovers}")
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.unlink(missing_ok=True)
    command = [
        *(worker or [sys.executable, "-m", WORKER_MODULE]),
        "--working-set-mib",
        str(working_set_mib),
        "--busy-ms",
        str(busy_ms),
        "--idle-ms",
        str(idle_ms),
        "--device",
        device,
        "--level",
        level,
        "--heartbeat",
        str(heartbeat),
        "--detached",
    ]
    with (heartbeat.parent / "worker.log").open("ab") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"cloud load worker exited {process.returncode}")
        beat = read_heartbeat(heartbeat)
        if (
            beat is not None
            and beat.get("status") == "running"
            and float(beat.get("allocated_mib", 0)) >= 0.9 * working_set_mib
        ):
            return {"started": True, "pid": process.pid, "heartbeat": beat}
        time.sleep(0.2)
    stop(heartbeat)
    raise RuntimeError(f"cloud load {level} did not start within {timeout_s:.0f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop", "reap"))
    parser.add_argument("--heartbeat-file", type=Path, default=DEFAULT_HEARTBEAT)
    parser.add_argument("--level")
    parser.add_argument("--working-set-mib", type=int)
    parser.add_argument("--busy-ms", type=float)
    parser.add_argument("--idle-ms", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        if args.action == "start":
            if None in (args.level, args.working_set_mib, args.busy_ms, args.idle_ms):
                parser.error("start needs --level and the level's parameters")
            result = start(
                args.heartbeat_file,
                level=args.level,
                working_set_mib=args.working_set_mib,
                busy_ms=args.busy_ms,
                idle_ms=args.idle_ms,
                device=args.device,
                timeout_s=args.timeout_s,
            )
        elif args.action == "status":
            result = status(args.heartbeat_file)
        else:
            result = stop(args.heartbeat_file)
            if result["remaining"]:
                print(json.dumps(result, sort_keys=True), file=sys.stderr)
                return 1
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
