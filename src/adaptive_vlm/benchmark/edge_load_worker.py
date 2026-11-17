"""Controlled edge GPU load: hold a working set and keep the GPU busy a set share.

Run as ``python -m adaptive_vlm.benchmark.edge_load_worker`` in the model
environment (needs torch). The worker allocates ``--working-set-mib`` on the
GPU, then repeats: launch in-place additions over the whole buffer for
``--busy-ms``, synchronize, sleep ``--idle-ms``. Every ``--heartbeat-s`` it
atomically rewrites a JSON heartbeat with its achieved duty cycle, so a
supervisor can verify it is alive and loading the GPU as configured. SIGTERM
or SIGINT stops it cleanly with a final ``"status": "stopped"`` heartbeat.

The same worker is the cloud competitor (C1, amendment A2a rule 6). Started on
the cloud by ``adaptive_vlm.benchmark.cloud_load``, it runs ``--detached``:
it must outlive the short-lived control command that launched it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

from adaptive_vlm.process import exit_with_parent

_MIB = 1024 * 1024


def write_heartbeat(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-set-mib", type=int, required=True)
    parser.add_argument("--busy-ms", type=float, required=True)
    parser.add_argument("--idle-ms", type=float, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--heartbeat-s", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--level", default="")
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Keep running after the launching process exits (cloud load)",
    )
    args = parser.parse_args(argv)
    if not args.detached:
        # Never outlive the benchmark that started us; SIGTERM stops us cleanly.
        exit_with_parent()
    if args.working_set_mib < 1 or args.busy_ms < 0 or args.idle_ms < 0:
        parser.error("working set must be positive; busy and idle non-negative")

    torch: Any = importlib.import_module("torch")  # Model environment only.
    if "microsoft" in platform.release().lower():
        # WSL's NVML device count crashes CUDA initialization in a second
        # process on the same GPU; fall back to the CUDA runtime's count.
        torch.cuda._device_count_nvml = lambda: -1

    stopping = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    buffer = torch.zeros(
        args.working_set_mib * _MIB // 4, dtype=torch.float32, device=args.device
    )
    torch.cuda.synchronize(args.device)
    started = time.time()
    state: dict[str, Any] = {
        "status": "running",
        "pid": os.getpid(),
        "level": args.level,
        "working_set_mib": args.working_set_mib,
        "busy_ms": args.busy_ms,
        "idle_ms": args.idle_ms,
        "allocated_mib": torch.cuda.memory_allocated(args.device) / _MIB,
        "started_unix_s": started,
        "kernels": 0,
        "busy_s": 0.0,
    }
    busy_s = 0.0
    kernels = 0
    next_beat = 0.0
    while not stopping:
        cycle = time.perf_counter()
        if args.busy_ms > 0:
            while (time.perf_counter() - cycle) * 1000 < args.busy_ms:
                buffer.add_(1.0)
                torch.cuda.synchronize(args.device)
                kernels += 1
            busy_s += time.perf_counter() - cycle
        elif kernels == 0:
            buffer.add_(1.0)  # Touch once so the working set is resident.
            torch.cuda.synchronize(args.device)
            kernels = 1
        if args.idle_ms > 0:
            time.sleep(args.idle_ms / 1000)
        now = time.time()
        if now >= next_beat:
            wall = now - started
            state.update(
                updated_unix_s=now,
                kernels=kernels,
                busy_s=busy_s,
                wall_s=wall,
                duty=busy_s / wall if wall > 0 else 0.0,
            )
            write_heartbeat(args.heartbeat, state)
            next_beat = now + args.heartbeat_s
    now = time.time()
    state.update(
        status="stopped",
        updated_unix_s=now,
        kernels=kernels,
        busy_s=busy_s,
        wall_s=now - started,
        duty=busy_s / (now - started) if now > started else 0.0,
    )
    write_heartbeat(args.heartbeat, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
