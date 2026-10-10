"""Read-only collection of hardware, software, GPU, and Git metadata."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def collect_environment(workdir: str | Path | None = None) -> dict[str, Any]:
    """Collect non-secret environment metadata for a run manifest."""

    working_directory = Path(workdir or Path.cwd()).resolve()
    return {
        "collected_at_utc": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "host": {
            "hostname": socket.gethostname(),
            "operating_system": platform.platform(),
            "machine": platform.machine(),
            "cpu": _cpu_name(),
            "logical_cpu_count": os.cpu_count(),
            "ram_gib": _ram_gib(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "gpu": _nvidia_gpu(),
        "git": _git_state(working_directory),
    }


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    processor = platform.processor().strip()
    if processor:
        return processor
    return "unknown"


def _ram_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return round(kib / 1024**2, 3)
    return None


def _nvidia_gpu() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = _run(command)
    if completed is None:
        return {"available": False, "reason": "nvidia-smi is not installed"}
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        reason = diagnostic.splitlines()
        return {
            "available": False,
            "reason": reason[0] if reason else "nvidia-smi failed",
        }

    devices: list[dict[str, Any]] = []
    for index, line in enumerate(completed.stdout.strip().splitlines()):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        name, memory_mib, driver_version = fields
        devices.append(
            {
                "index": index,
                "name": name,
                "memory_total_mib": _parse_int(memory_mib),
                "driver_version": driver_version,
            }
        )
    return {"available": bool(devices), "devices": devices}


def _git_state(workdir: Path) -> dict[str, Any]:
    commit_result = _run(["git", "rev-parse", "HEAD"], cwd=workdir)
    status_result = _run(["git", "status", "--porcelain"], cwd=workdir)
    if (
        commit_result is None
        or status_result is None
        or commit_result.returncode != 0
        or status_result.returncode != 0
    ):
        return {"available": False, "commit": None, "dirty": None}
    return {
        "available": True,
        "commit": commit_result.stdout.strip(),
        "dirty": bool(status_result.stdout.strip()),
    }


def _run(
    command: Sequence[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None

