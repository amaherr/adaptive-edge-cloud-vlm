"""Read-only collection of hardware, software, GPU, and Git metadata."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.timing import clock_metadata


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
        "duration_clock": clock_metadata(),
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


BUILD_RECORD = Path("/usr/local/share/adaptive-vlm-build.json")


def image_identity(record: Path | None = None) -> dict[str, Any] | None:
    """Commit and image of the running container, if this is the campaign image.

    The image is built from a clean commit (``scripts/cloud/build_cloud_image.py``)
    and records it at build time; ``APP_IMAGE`` holds the digest it runs as.
    """

    path = record or BUILD_RECORD
    if not path.is_file():
        return None
    build = json.loads(path.read_text(encoding="utf-8"))
    return {
        "commit": build.get("git_commit"),
        "image": os.environ.get("APP_IMAGE"),
        "base_image": build.get("base_image"),
        "dependency_lock_sha256": build.get("dependency_lock_sha256"),
    }


def _git_state(workdir: Path) -> dict[str, Any]:
    image = image_identity()
    if image is not None and not (workdir / ".git").exists():
        # Inside the campaign image there is no checkout; the image is the
        # source identity and was built from a clean commit.
        return {
            "available": True,
            "commit": image["commit"],
            "dirty": False,
            "source": "image",
            "image": image["image"],
        }
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


# The swap-residency peak measured on the RTX 3050 edge was 3,164 MiB reserved;
# require that plus a margin free before an unattended run starts.
SWAP_RESIDENCY_MIN_FREE_VRAM_MIB = 3_400


def disable_core_dumps() -> bool:
    """Stop this process and its children from writing core dumps.

    WSL pipes core dumps to its own crash collector, which ignores a limit of
    0 and wrote a 5.4 GB dump of a crashed benchmark to the Windows system
    drive. The kernel skips piped core dumps when the limit is exactly 1.
    Returns False where resource limits are unavailable (Windows).
    """

    try:
        import resource
    except ImportError:
        return False
    resource.setrlimit(resource.RLIMIT_CORE, (1, 1))
    return True


def gpu_free_memory_mib(index: int = 0) -> int | None:
    """Free memory on one NVIDIA GPU as reported by nvidia-smi, if available."""

    result = _run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0:
        return None
    return _parse_int(result.stdout.strip())
