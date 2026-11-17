"""Unattended G2 v2 campaign supervisor (protocol sections 5-8).

The supervisor runs preflight checks, measures the unshaped network baseline
once, then walks the grid's seeded sub-block order: shape both hosts, run the
sub-block (``campaign.subblock``), and move an invalid sub-block aside to rerun
it whole. Children are watched for stalled journals, disk and host-memory
pressure. Shaping is always removed at the end, even after a failure.

``mode: "loopback"`` rehearses the same flow on one machine: no shaping, no
probes, no SSH, and optional relaxations that are recorded in the plan and
refused in ``physical`` mode.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.edge_load import load_levels
from adaptive_vlm.campaign.checkpoint import Heartbeats, S3Checkpoint
from adaptive_vlm.campaign.grid import PRIMARY_TASK, Grid, check_split, load_grid
from adaptive_vlm.campaign.network import (
    ClassifiedNetwork,
    LoopbackNetwork,
    PhysicalNetwork,
    is_default_qdisc,
)
from adaptive_vlm.campaign.remote import Runner, runner_from_settings
from adaptive_vlm.campaign.subblock import (
    Gates,
    check_completed,
    run_subblock,
)
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.environment import (
    SWAP_RESIDENCY_MIN_FREE_VRAM_MIB,
    gpu_free_memory_mib,
    image_identity,
)
from adaptive_vlm.execution.cloud import CloudExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.process import find_processes
from adaptive_vlm.timing import monotonic

MODES = ("physical", "loopback")
_PATHS = (
    "grid",
    "output_root",
    "artifact_manifest",
    "cache_dir",
    "config",
    "split",
    "edge_load_levels",
    "same_device_reference_evidence",
)
_EXECUTABLES = ("python", "test_python", "adaptive_vlm")
PLATFORMS = {
    # platform: (spill metric, source identity)
    "physical-wsl": ("edge_windows_gpu_dedicated_mib", "git"),
    "aws-linux": ("edge_gpu_memory_used_mib", "image"),
}
_DEFAULTS: dict[str, Any] = {
    "platform": "physical-wsl",
    "edge_device": "cuda:0",
    "edge_residency": "swap",
    "rate_fallback": False,
    "max_restarts_per_subblock": 3,
    "max_subblock_attempts": 2,
    "max_failure_rate": 0.01,
    "edge_dedicated_limit_mib": 3960,
    "spill_margin_mib": 128,
    "require_spill_evidence": True,
    "run_test_suite": True,
    "correctness_smoke": True,
    "allow_unfrozen_protocol": False,
    "allow_dirty_tree": False,
    "min_host_available_ram_mib": 0,
    "min_runtime_host_available_ram_mib": 0,
    "require_cloud_wsl_keepalive": False,
    "smoke_requests": 6,
}
_LOOPBACK_ONLY = ("allow_dirty_tree",)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{now()} {message}", flush=True)


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(command: list[str], *, timeout: float = 60) -> str:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{command[0]} exited {exc.returncode}: {exc.stderr.strip()}"
        ) from exc
    return result.stdout.strip()


def load_settings(path: Path, root: Path) -> dict[str, Any]:
    """Read campaign settings; resolve paths against the repository root."""

    settings = {**_DEFAULTS, **json.loads(path.read_text(encoding="utf-8"))}
    if settings.get("mode") not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    required = [
        "grid",
        "output_root",
        "python",
        "test_python",
        "adaptive_vlm",
        "artifact_manifest",
        "cache_dir",
        "stall_seconds",
        "min_free_gb",
    ]
    if settings["mode"] == "physical":
        legacy = str(settings.get("cloud_ssh") or "")
        if not settings.get("cloud_access") and (not legacy or "REPLACE" in legacy):
            raise ValueError(f"set cloud_access (ssm or ssh) or cloud_ssh in {path}")
        required += [
            "cloud_repo",
            "cloud_python",
            "cloud_interface",
            "same_device_reference_evidence",
        ]
        for name in _LOOPBACK_ONLY:
            if settings[name]:
                raise ValueError(f"{name} is allowed only in loopback rehearsals")
    for name in required:
        value = settings.get(name)
        if value in (None, "") or (isinstance(value, str) and "REPLACE" in value):
            raise ValueError(f"set {name} in {path}")
    for name in _PATHS:
        value = settings.get(name)
        if isinstance(value, dict):  # Per-task paths, e.g. reference evidence.
            settings[name] = {k: str((root / v).resolve()) for k, v in value.items()}
        elif value:
            settings[name] = str((root / value).resolve())
    for name in _EXECUTABLES:
        # Resolving a venv interpreter symlink would drop its packages.
        settings[name] = os.path.abspath(root / settings[name])
    if settings["platform"] not in PLATFORMS:
        raise ValueError(f"platform must be one of {sorted(PLATFORMS)}")
    if settings["edge_residency"] not in ("resident", "swap"):
        raise ValueError(f"set edge_residency (resident or swap) in {path}")
    for name in ("edge_dedicated_limit_mib", "spill_margin_mib"):
        try:
            settings[name] = float(settings[name])
        except (TypeError, ValueError):
            raise ValueError(f"set {name} to a number in {path}") from None
    if settings["mode"] == "physical" and PLATFORMS[settings["platform"]][1] == "image":
        for name in ("cloud_container",):
            if not settings.get(name):
                raise ValueError(f"set {name} in {path}")
    if float(settings["stall_seconds"]) < 180:
        raise ValueError("stall_seconds must be at least 180")
    return settings


def host_available_ram_mib() -> float:
    """Windows host free RAM under WSL; Linux MemAvailable otherwise."""

    release = Path("/proc/sys/kernel/osrelease").read_text().lower()
    if "microsoft" in release:
        powershell = shutil.which("powershell.exe") or (
            "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        )
        return float(
            checked(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "[math]::Round((Get-CimInstance Win32_OperatingSystem)."
                    "FreePhysicalMemory / 1024, 2)",
                ],
                timeout=20,
            )
        )
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def progress(output: Path, target: int | None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    warmups = 0
    for path in (output / "subblocks").glob("*/run/*/requests.jsonl"):
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue
            row = json.loads(line)
            if row["warmup"]:
                warmups += 1
            else:
                counts[row["status"]] += 1
    measured = sum(counts.values())
    valid = sum(
        1
        for path in (output / "subblocks").glob("*/validation_report.json")
        if json.loads(path.read_text()).get("valid")
    )
    return {
        "measured_attempts": measured,
        "target_attempts": target,
        "percent": round(100 * measured / target, 2) if target else None,
        "warmup_attempts": warmups,
        "status_counts": dict(counts),
        "valid_subblocks": valid,
    }


class CampaignSupervisor:
    def __init__(
        self,
        settings: dict[str, Any],
        *,
        root: Path,
        network: Any = None,
        ssh: Callable[[list[str]], str] | None = None,
    ) -> None:
        self.settings = settings
        self.root = root
        self.output = Path(settings["output_root"])
        self.grid: Grid = load_grid(
            Path(settings["grid"]), rate_fallback=bool(settings["rate_fallback"])
        )
        # Per-task experiment config and split; settings may override the
        # primary task's (rehearsals), never an extra task's.
        self.task_paths: dict[str, tuple[Path, Path]] = {
            name: (task.experiment_config, task.split)
            for name, task in self.grid.tasks.items()
        }
        primary_config, primary_split = self.task_paths[PRIMARY_TASK]
        self.task_paths[PRIMARY_TASK] = (
            Path(settings["config"]) if settings.get("config") else primary_config,
            Path(settings["split"]) if settings.get("split") else primary_split,
        )
        self.config_path, self.split_path = self.task_paths[PRIMARY_TASK]
        self.levels_path = (
            Path(settings["edge_load_levels"])
            if settings.get("edge_load_levels")
            else self.grid.resolve("edge_load_levels_path")
        )
        self.physical = settings["mode"] == "physical"
        # The cloud runner is SSH on the physical pair and SSM on AWS.
        remote: Runner | None = (
            runner_from_settings(settings) if self.physical else None
        )
        self._ssh = ssh or (
            (lambda command: remote(command).strip()) if remote is not None else None
        )
        self.network = network or self._network(settings, root, remote)
        self.child: subprocess.Popen[Any] | None = None
        self.stage = "initializing"
        self.block: str | None = None
        self.shaped = (self.output / "shaping_active.json").exists()
        self.target: int | None = None
        self.baseline_rtt_ms: float | None = None
        checkpoint = settings.get("checkpoint") or {}
        self.checkpoint = (
            S3Checkpoint(
                str(checkpoint["s3_uri"]),
                region=checkpoint.get("region"),
                profile=checkpoint.get("profile"),
            )
            if checkpoint.get("s3_uri")
            else None
        )
        beats = settings.get("heartbeat") or {}
        self.heartbeats = Heartbeats(
            edge_path=beats.get("edge_path"),
            cloud_path=beats.get("cloud_path"),
            cloud_runner=remote,
            cloud_every_s=float(beats.get("cloud_every_s", 300)),
        )
        self.warnings: list[str] = []

    def _network(self, settings: dict[str, Any], root: Path, remote: Any) -> Any:
        """Whole-interface netem (physical pair) or classified netem (AWS)."""

        if remote is None:
            return LoopbackNetwork()
        common = {
            "edge_interface": self.grid.payload["edge_interface"],
            "cloud_interface": settings["cloud_interface"],
            "cloud_host": self.grid.payload["cloud_host"],
            "iperf_port": int(self.grid.payload["iperf_port"]),
            "echo_port": int(self.grid.payload["echo_port"]),
            "limit_packets": int(self.grid.payload["netem_limit_packets"]),
            "echo_script": root / "scripts/network/tcp_rtt_probe.py",
            "remote": remote,
        }
        if settings.get("shaping", "root") == "classified":
            return ClassifiedNetwork(
                edge_address=settings["edge_address"],
                privilege=settings.get("tc_privilege", ["sudo", "-n"]),
                **common,
            )
        return PhysicalNetwork(**common)

    # -- status and child processes -------------------------------------

    def status(self, state: str, **extra: Any) -> None:
        warning = self.heartbeats.beat()
        if warning:
            self.warnings.append(f"{now()} {warning}")
            log(warning)
        payload = {
            "updated_at_utc": now(),
            "pid": os.getpid(),
            "state": state,
            "stage": self.stage,
            "subblock": self.block,
            **progress(self.output, self.target),
            "warnings": self.warnings[-20:],
            **extra,
        }
        write_json_atomic(self.output / "status.json", payload)
        log(
            f"{state} {self.stage} subblock={self.block} "
            f"{payload['measured_attempts']}/{self.target} "
            f"valid_subblocks={payload['valid_subblocks']}"
        )

    def check_host_memory(self, *, startup: bool) -> None:
        key = (
            "min_host_available_ram_mib"
            if startup
            else "min_runtime_host_available_ram_mib"
        )
        minimum = float(self.settings[key])
        if minimum > 0 and (available := host_available_ram_mib()) < minimum:
            raise RuntimeError(
                f"host RAM pressure: {available:.0f} MiB available, "
                f"{minimum:.0f} needed"
            )

    def _committed_rows(self, run_dir: Path | None) -> int:
        if run_dir is None:
            return 0
        return sum(
            path.read_bytes().count(b"\n") for path in run_dir.glob("*/requests.jsonl")
        )

    def run_child(self, command: list[str], run_dir: Path | None = None) -> int:
        """Run one child with a stall/disk/memory watchdog; return its exit code."""

        self.child = subprocess.Popen(command, start_new_session=True)
        last_count = self._committed_rows(run_dir)
        last_change = monotonic()
        try:
            while True:
                try:
                    return self.child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.check_host_memory(startup=False)
                    self.status("running")
                    minimum = float(self.settings["min_free_gb"]) * 1024**3
                    if shutil.disk_usage(self.output).free < minimum:
                        raise RuntimeError(
                            "edge disk space fell below reserve"
                        ) from None
                    if run_dir is None:
                        continue
                    count = self._committed_rows(run_dir)
                    if count != last_count:
                        last_count, last_change = count, monotonic()
                    elif monotonic() - last_change > float(
                        self.settings["stall_seconds"]
                    ):
                        raise RuntimeError(
                            "no committed attempt within the stall timeout"
                        ) from None
        finally:
            self.stop_child()

    def reap_strays(self) -> list[int]:
        """Stop benchmarks or load workers left by a killed supervisor.

        Every campaign child names a path under ``subblocks/`` on its command
        line, so this matches only this campaign's processes.
        """

        import time

        pids = find_processes(str(self.output / "subblocks"))
        if not pids:
            return []
        for pid in pids:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and any(
            Path(f"/proc/{pid}").exists() for pid in pids
        ):
            time.sleep(0.2)
        for pid in pids:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        with (self.output / "stray_processes.jsonl").open("a") as handle:
            handle.write(json.dumps({"time": now(), "pids": pids}) + "\n")
        log(f"stopped stray campaign processes: {pids}")
        return pids

    def stop_child(self) -> None:
        child, self.child = self.child, None
        if child is None:
            return
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            child.wait(timeout=20)
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> dict[str, Any]:
        s = self.settings
        self.stage = "preflight"
        self.output.mkdir(parents=True, exist_ok=True)
        self.check_host_memory(startup=True)
        self.status("preflight")
        commit, dirty = self._source_identity()
        if dirty and not s["allow_dirty_tree"]:
            raise RuntimeError("commit or stash source changes before the campaign")
        protocol = self.grid.resolve("protocol_path")
        pinned = self.grid.payload.get("protocol_sha256")
        if pinned != sha256(protocol) and not s["allow_unfrozen_protocol"]:
            raise RuntimeError(
                "the grid does not pin the current protocol hash; freeze the "
                "protocol first (or set allow_unfrozen_protocol for a rehearsal)"
            )
        tasks: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
        for name, (config_path, split_path) in self.task_paths.items():
            task_config = load_experiment_config(config_path)
            task_manifest = load_frozen_manifest(
                task_config.dataset.manifest_path,
                expected_revision=task_config.dataset.revision,
            )
            task_split = json.loads(split_path.read_text(encoding="utf-8"))
            if task_manifest.sha256 != task_split["dataset_manifest_sha256"] or (
                model_contract_fingerprint(task_config.model)
                != task_split["model_contract_sha256"]
            ):
                raise ValueError(f"{name} experiment config differs from its split")
            check_split(self.grid, task_split, name)
            tasks[name] = (task_config, task_manifest, task_split)
        config = tasks[PRIMARY_TASK][0]
        if any(
            model_contract_fingerprint(c.model)
            != model_contract_fingerprint(config.model)
            or c.transport != config.transport
            for c, _, _ in tasks.values()
        ):
            raise ValueError("all tasks must share the model contract and cloud server")
        levels = load_levels(self.levels_path)
        missing = self.grid.all_load_levels() - set(levels)
        if missing:
            raise ValueError(f"edge-load levels missing: {sorted(missing)}")
        expected = self.grid.expected_attempts({n: t[2] for n, t in tasks.items()})
        self.target = expected["measured"]
        policy = CloudExecutionPolicy(
            transport=config.transport,
            model=ModelMetadata(
                config.model.model_id,
                config.model.revision,
                model_contract_fingerprint(config.model),
            ),
        )
        if not policy.health().ready:
            raise RuntimeError("cloud inference server is not ready")
        minimum = float(s["min_free_gb"]) * 1024**3
        if shutil.disk_usage(self.output).free < minimum:
            raise RuntimeError("insufficient free disk space on edge")
        if s["edge_residency"] == "swap" and config.model.adapter != "fake":
            free = gpu_free_memory_mib()
            if free is None or free < SWAP_RESIDENCY_MIN_FREE_VRAM_MIB:
                raise RuntimeError(
                    f"edge GPU has {free} MiB free; swap residency needs "
                    f"{SWAP_RESIDENCY_MIN_FREE_VRAM_MIB} MiB"
                )
        hosts: dict[str, Any] = {}
        if self.physical:
            hosts = self._physical_preflight(commit, config, tasks)
        if s["run_test_suite"]:
            self._run_test_suite()
        self.baseline_rtt_ms = self._network_baseline()
        contract = {
            "schema_version": 1,
            "mode": s["mode"],
            "git_sha": commit,
            "git_dirty": bool(dirty),
            "settings": s,
            "grid_sha256": self.grid.sha256,
            "protocol_sha256": sha256(protocol),
            "protocol_pinned": pinned == sha256(protocol),
            "tasks": {
                name: {
                    "config_sha256": sha256(config_path),
                    "split_sha256": sha256(split_path),
                    "dataset_manifest_sha256": tasks[name][1].sha256,
                    "task_bucket": self.grid.tasks[name].task_bucket,
                }
                for name, (config_path, split_path) in self.task_paths.items()
            },
            "edge_load_levels_sha256": sha256(self.levels_path),
            "expected_attempts": expected,
            "subblock_order": [b.subblock_id for b in self.grid.execution_order()],
        }
        plan = self.output / "campaign_plan.json"
        if plan.exists() and json.loads(plan.read_text()) != json.loads(
            json.dumps(contract)
        ):
            raise ValueError("campaign contract changed; refusing to mix results")
        write_json_atomic(plan, contract)
        write_json_atomic(self.output / "host_preflight.json", {"time": now(), **hosts})
        return contract

    def reference_for(self, task: str) -> Path:
        """The same-device reference index for one task's manifest."""

        value = self.settings.get("same_device_reference_evidence")
        if isinstance(value, dict):
            if task not in value:
                raise ValueError(f"no same-device reference evidence for task {task}")
            return Path(value[task])
        if task != PRIMARY_TASK or not value:
            raise ValueError(
                f"same_device_reference_evidence needs a path per task ({task})"
            )
        return Path(value)

    def _source_identity(self) -> tuple[str, str]:
        """(commit, dirty description): git checkout, or the campaign image."""

        if PLATFORMS[self.settings["platform"]][1] == "git":
            return (
                checked(["git", "rev-parse", "HEAD"]),
                checked(["git", "status", "--porcelain"]),
            )
        identity = image_identity()
        image = os.environ.get("APP_IMAGE", "")
        if identity is None or not identity.get("commit"):
            raise RuntimeError("not running in the campaign image (no build record)")
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
            raise RuntimeError("APP_IMAGE must be the repository@sha256 digest")
        return str(identity["commit"]), ""

    def _physical_preflight(
        self,
        commit: str,
        config: Any,
        tasks: dict[str, tuple[Any, Any, dict[str, Any]]],
    ) -> dict[str, Any]:
        from adaptive_vlm.benchmark.heterogeneous_correctness import (
            PROTOCOL,
            verify_reference_evidence,
        )

        s, ssh = self.settings, self._ssh
        assert ssh is not None
        image_mode = PLATFORMS[s["platform"]][1] == "image"
        required = (
            ("tc", "iperf3", "nvidia-smi", "aws")
            if image_mode
            else ("ssh", "tc", "sudo", "iperf3", "nvidia-smi")
        )
        for command in required:
            if shutil.which(command) is None:
                raise RuntimeError(f"missing edge command: {command}")
        if image_mode:
            cloud_image = ssh(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.Config.Image}}",
                    s["cloud_container"],
                ]
            )
            if cloud_image != os.environ.get("APP_IMAGE"):
                raise RuntimeError("cloud must run the same image digest as the edge")
        elif ssh(["git", "-C", s["cloud_repo"], "rev-parse", "HEAD"]) != commit or ssh(
            ["git", "-C", s["cloud_repo"], "status", "--porcelain"]
        ):
            raise RuntimeError("cloud must use the same clean source commit as edge")
        checked(
            [
                *s.get("tc_privilege", ["sudo", "-n"]),
                "tc",
                "qdisc",
                "show",
                "dev",
                self.grid.payload["edge_interface"],
            ]
        )
        ssh(
            [
                *s.get("tc_privilege", ["sudo", "-n"]),
                "tc",
                "qdisc",
                "show",
                "dev",
                s["cloud_interface"],
            ]
        )
        if config.transport.host != self.grid.payload["cloud_host"]:
            raise ValueError("experiment config and grid name different cloud hosts")
        for port in (self.grid.payload["iperf_port"], self.grid.payload["echo_port"]):
            with socket.create_connection((self.grid.payload["cloud_host"], port), 10):
                pass
        remote_free = int(
            ssh(
                [
                    *shlex.split(s["cloud_python"]),
                    "-c",
                    "import shutil; print(shutil.disk_usage("
                    f"{s['cloud_repo']!r}).free)",
                ]
            )
        )
        if remote_free < float(s["min_free_gb"]) * 1024**3:
            raise RuntimeError("insufficient free disk space on cloud")
        if self.grid.payload.get("correctness_protocol") != PROTOCOL:
            raise ValueError(
                "G2 v2 requires the heterogeneous-GPU correctness protocol"
            )
        code = (
            "import json; from adaptive_vlm.benchmark.heterogeneous_correctness "
            "import runtime_versions; print(json.dumps(runtime_versions()))"
        )
        runtimes = {
            "edge": json.loads(checked([s["python"], "-c", code])),
            "cloud": json.loads(ssh([*shlex.split(s["cloud_python"]), "-c", code])),
        }
        references = {}
        for name, (_task_config, task_manifest, task_split) in tasks.items():
            reference = self.reference_for(name)
            verify_reference_evidence(
                reference,
                sha256(reference),
                task_manifest,
                task_split["model_contract_sha256"],
                expected_devices={
                    host: config.to_dict()[host]["gpu"] for host in ("edge", "cloud")
                },
            )
            if json.loads(reference.read_text())["runtime_versions"] != runtimes:
                raise RuntimeError(
                    f"runtime versions differ from the {name} reference evidence"
                )
            references[name] = sha256(reference)
        if (
            s["require_cloud_wsl_keepalive"]
            and "microsoft" in ssh(["uname", "-r"]).lower()
        ):
            sleepers = ssh(["pgrep", "-a", "-x", "sleep"])
            if not any(
                line.endswith("sleep infinity") for line in sleepers.splitlines()
            ):
                raise RuntimeError("keep the cloud WSL foreground sleep infinity open")
        return {
            "edge_gpu": checked(["nvidia-smi"]),
            "cloud_gpu": ssh(["nvidia-smi"]),
            "cloud_free_bytes": remote_free,
            "source_commit": commit,
            "reference_evidence_sha256": references,
        }

    def _run_test_suite(self) -> None:
        junit = self.output / "tests.xml"
        for arguments in (
            ["ruff", "check", "."],
            ["mypy", "--cache-dir", str(self.output / "mypy_cache")],
            ["pytest", "-q", "-rs", f"--junitxml={junit}"],
        ):
            if self.run_child([self.settings["test_python"], "-m", *arguments]):
                raise RuntimeError(f"preflight {arguments[0]} failed")
        if any(True for _ in ElementTree.parse(junit).iter("skipped")):
            raise RuntimeError("preflight requires tests to pass without skips")

    def _network_baseline(self) -> float | None:
        """Unshaped echo RTT, measured once and reused on every resume."""

        path = self.output / "network_baseline.json"
        if path.exists():
            saved = json.loads(path.read_text()).get("echo_median_rtt_ms")
            return None if saved is None else float(saved)
        if self.shaped:
            errors = self.network.clear()
            if errors:
                raise RuntimeError(f"could not clear leftover shaping: {errors}")
            (self.output / "shaping_active.json").unlink(missing_ok=True)
            self.shaped = False
        for name, state in self.network.qdiscs().items():
            if not is_default_qdisc(state):
                raise RuntimeError(f"{name} has a foreign qdisc: {state.raw}")
        measurement = self.network.measure()
        write_json_atomic(path, measurement)
        rtt = measurement.get("echo_median_rtt_ms")
        return None if rtt is None else float(rtt)

    # -- execution ----------------------------------------------------------

    def command_prefix(self, task: str = PRIMARY_TASK) -> list[str]:
        s = self.settings
        config_path, split_path = self.task_paths[task]
        return [
            s["adaptive_vlm"],
            "interleaved-benchmark",
            "--config",
            str(config_path),
            "--split",
            str(split_path),
            "--artifact-manifest",
            s["artifact_manifest"],
            "--cache-dir",
            s["cache_dir"],
            "--offline",
            "--edge-device",
            s["edge_device"],
            "--edge-residency",
            s["edge_residency"],
            "--resource-state",
            "--edge-load-config",
            str(self.levels_path),
            "--max-consecutive-failures",
            "3",
        ]

    def shape(self, rate_mbps: float, delay_ms: float) -> None:
        write_json_atomic(
            self.output / "shaping_active.json",
            {"time": now(), "rate_mbps": rate_mbps, "delay_ms": delay_ms},
        )
        owned, self.shaped = self.shaped, True  # Clean up even if one peer fails.
        self.network.apply(rate_mbps=rate_mbps, delay_ms=delay_ms, owned=owned)

    def cleanup(self) -> list[str]:
        if not self.shaped:
            return []
        errors = list(self.network.clear())
        if not errors:
            (self.output / "shaping_active.json").unlink(missing_ok=True)
            self.shaped = False
        return errors

    def execute(self, *, preflight_only: bool) -> None:
        terminal = "failed"
        failure: BaseException | None = None
        s = self.settings
        gates = Gates(
            max_failure_rate=float(s["max_failure_rate"]),
            edge_dedicated_limit_mib=float(s["edge_dedicated_limit_mib"]),
            spill_margin_mib=float(s["spill_margin_mib"]),
            require_spill_evidence=bool(s["require_spill_evidence"]),
            spill_metric=PLATFORMS[s["platform"]][0],
        )
        try:
            self.preflight()
            if s["correctness_smoke"]:
                self.smoke()
            if preflight_only:
                terminal = "preflight_passed"
                return
            order = self.grid.execution_order()
            for index, block in enumerate(order, 1):
                self.block = block.subblock_id
                directory = self.output / "subblocks" / block.subblock_id
                report = check_completed(directory)
                if report is not None and report["valid"]:
                    continue
                for attempt in range(1, int(s["max_subblock_attempts"]) + 1):
                    self.reap_strays()
                    if report is not None and not report["valid"]:
                        self._archive(directory, report)
                    self.stage = f"sub-block {index}/{len(order)} attempt {attempt}"
                    self.status("running")
                    condition = block.condition
                    self.shape(condition.rate_mbps, condition.egress_delay_ms)
                    report = run_subblock(
                        grid=self.grid,
                        block=block,
                        output=directory,
                        network=self.network,
                        baseline_rtt_ms=self.baseline_rtt_ms,
                        command_prefix=self.command_prefix(block.task),
                        run_child=self.run_child,
                        gates=gates,
                        max_restarts=int(s["max_restarts_per_subblock"]),
                        log=log,
                    )
                    if report["valid"]:
                        if self.checkpoint is not None:
                            error = self.checkpoint.push_subblock(
                                self.output, block.subblock_id
                            )
                            if error:
                                self.warnings.append(f"{now()} checkpoint {error}")
                                log(f"checkpoint failed (data safe on EBS): {error}")
                        break
                    log(f"{block.subblock_id} invalid: {report['reasons']}")
                else:
                    raise RuntimeError(
                        f"{block.subblock_id} stayed invalid after "
                        f"{s['max_subblock_attempts']} attempts"
                    )
            if progress(self.output, self.target)["measured_attempts"] != self.target:
                raise RuntimeError("measured attempts differ from the plan")
            terminal = "completed"
        except BaseException as exc:
            failure = exc
            raise
        finally:
            self.stop_child()
            errors = self.cleanup()
            self.status(
                "failed" if errors else terminal,
                error=str(failure) if failure else None,
                cleanup_errors=errors,
            )
            if errors and failure is None:
                raise RuntimeError(f"network cleanup failed: {errors}")

    def _archive(self, directory: Path, report: dict[str, Any]) -> None:
        archive = (
            self.output
            / "invalid"
            / f"{directory.name}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(archive)
        log(f"moved invalid {directory.name} to {archive}: {report['reasons']}")

    def smoke(self) -> None:
        """Three-placement GPU correctness smoke on a few training requests.

        Uses the campaign's own config (so it reaches the right cloud on either
        platform) and a subset of the primary task's training IDs, which the
        same-device reference covers. Validation and evaluation IDs are never
        touched; these preflight rows are not campaign data.
        """

        from adaptive_vlm.benchmark.compare import compare_runs
        from adaptive_vlm.benchmark.heterogeneous_correctness import (
            PROTOCOL,
            evaluate_rows,
            verify_reference_evidence,
        )

        s = self.settings
        self.stage = "three-placement correctness smoke"
        self.status("preflight")
        output = (
            self.output / "preflights" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        )
        output.mkdir(parents=True)
        experiment = load_experiment_config(self.config_path)
        experiment_manifest = load_frozen_manifest(
            experiment.dataset.manifest_path,
            expected_revision=experiment.dataset.revision,
        )
        split = json.loads(self.split_path.read_text(encoding="utf-8"))
        chosen = set(sorted(split["training_request_ids"])[: int(s["smoke_requests"])])
        source = {
            row["request_id"]: row
            for row in (
                json.loads(line)
                for line in experiment.dataset.manifest_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            )
        }
        rows = []
        for item in experiment_manifest.items:
            if item.request_id in chosen:
                row = dict(source[item.request_id])
                row["image_path"] = str(item.image_path.resolve())
                rows.append(row)
        subset = output / "smoke_manifest.jsonl"
        subset.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
            encoding="utf-8",
        )
        manifest = load_frozen_manifest(
            subset, expected_revision=experiment.dataset.revision
        )
        modes = ("edge", "cloud", "distributed")
        for mode in modes:
            command = [
                s["adaptive_vlm"],
                "benchmark",
                "--config",
                str(self.config_path),
                "--manifest",
                str(subset),
                "--mode",
                mode,
                "--run-dir",
                str(output / mode),
                "--artifact-manifest",
                s["artifact_manifest"],
                "--cache-dir",
                s["cache_dir"],
                "--offline",
                "--warmups",
                "1",
                "--repetitions",
                "2",
                "--max-consecutive-failures",
                "1",
                "--vision-device",
                "cpu" if mode == "cloud" else s["edge_device"],
                "--text-device",
                s["edge_device"] if mode == "edge" else "cpu",
            ]
            if mode == "edge":
                command += ["--edge-residency", s["edge_residency"]]
            if self.run_child(command):
                raise RuntimeError(f"correctness smoke {mode} failed")
        compare_runs([output / m for m in modes], output=output / "comparison.json")
        reference = self.reference_for(PRIMARY_TASK)
        proof = verify_reference_evidence(
            reference,
            sha256(reference),
            experiment_manifest,
            model_contract_fingerprint(experiment.model),
        )
        assessment = evaluate_rows(
            {
                mode: [
                    json.loads(line)
                    for line in (output / mode / "requests.jsonl")
                    .read_text()
                    .splitlines()
                ]
                for mode in modes
            },
            manifest,
            expected_outputs=proof["_expected_outputs"],
        )
        write_json_atomic(output / "heterogeneous_correctness.json", assessment)
        if (
            self.grid.payload.get("correctness_protocol") != PROTOCOL
            or not assessment["passed"]
        ):
            raise RuntimeError(f"GPU correctness smoke failed: {output}")
        log(f"GPU correctness smoke passed: {output}")
