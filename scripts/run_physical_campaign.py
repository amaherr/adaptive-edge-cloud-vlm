"""Supervise the physical 8,100-attempt campaign from the edge Linux host.

start detaches; run stays in the foreground. Both repeat preflights before work.
No automatic retry changes a recorded failed attempt into a successful attempt.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from adaptive_vlm.artifacts import write_json_atomic, write_jsonl_atomic
from adaptive_vlm.benchmark.compare import compare_runs
from adaptive_vlm.benchmark.heterogeneous import (
    PROTOCOL,
    evaluate_rows,
    verify_reference_evidence,
)
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.cloud import CloudExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.timing import monotonic

ROOT = Path(__file__).resolve().parents[1]
MODES = ("edge", "cloud", "distributed")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(message: str) -> None:
    print(f"{now()} {message}", flush=True)


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


def ssh(settings: dict[str, Any], command: list[str]) -> str:
    return checked(
        [
            "ssh",
            "-oBatchMode=yes",
            "-oConnectTimeout=10",
            "-oServerAliveInterval=15",
            "-oServerAliveCountMax=2",
            settings["cloud_ssh"],
            shlex.join(command),
        ]
    )


def load_settings(path: Path) -> dict[str, Any]:
    settings = read(path)
    for name in ("cloud_ssh", "cloud_repo", "cloud_python", "cloud_interface"):
        value = settings.get(name, "")
        if not value or "REPLACE" in value or value.startswith("-"):
            raise ValueError(f"set {name} in {path}")
    for name in (
        "grid",
        "split",
        "config",
        "smoke_config",
        "output_root",
        "python",
        "test_python",
        "adaptive_vlm",
        "artifact_manifest",
        "cache_dir",
    ):
        path = ROOT / settings[name]
        # Following a venv Python symlink selects the base interpreter and loses
        # the environment's packages. Keep executable paths inside their venv.
        settings[name] = (
            os.path.abspath(path)
            if name in ("python", "test_python", "adaptive_vlm")
            else str(path.resolve())
        )
    if float(settings["stall_seconds"]) < 180:
        raise ValueError("stall_seconds must be at least 180")
    output = Path(settings["output_root"])
    # Keep experiment outputs out of the tracked source tree.
    checked(["git", "check-ignore", str(output / "probe.json")])
    return settings


def progress(output: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    warmups = 0
    last_commit = None
    for path in output.glob("N*/*/requests.jsonl"):
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue  # Only newline-committed rows count.
            row = json.loads(line)
            if row["warmup"]:
                warmups += 1
            else:
                counts[row["status"]] += 1
        modified = path.stat().st_mtime
        last_commit = max(last_commit or modified, modified)
    measured = sum(counts.values())
    return {
        "measured_attempts": measured,
        "target_attempts": 8100,
        "percent": round(100 * measured / 8100, 2),
        "warmup_attempts": warmups,
        "status_counts": dict(counts),
        "last_journal_write_unix": last_commit,
    }


def host_available_ram_mib() -> float:
    """Read Windows host RAM on WSL, or Linux MemAvailable on native Linux."""
    release = Path("/proc/sys/kernel/osrelease").read_text().lower()
    if "microsoft" in release:
        powershell = shutil.which("powershell.exe") or (
            "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        )
        return float(
            json.loads(
                checked(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        "[math]::Round((Get-CimInstance Win32_OperatingSystem)."
                        "FreePhysicalMemory / 1024, 2) | ConvertTo-Json -Compress",
                    ],
                    timeout=20,
                )
            )
        )
    values = Path("/proc/meminfo").read_text().splitlines()
    return (
        float(
            next(line.split()[1] for line in values if line.startswith("MemAvailable:"))
        )
        / 1024
    )


def validate_block(output: Path) -> dict[str, Any]:
    report = read(output / "validation_report.json")
    for name, key in (
        ("network_pre.json", "network_pre_sha256"),
        ("network_post.json", "network_post_sha256"),
        ("profile/profile.jsonl", "profile_sha256"),
        ("edge_gpu_samples.csv", "edge_gpu_samples_sha256"),
    ):
        if sha(output / name) != report[key]:
            raise ValueError(f"changed completed block artifact: {output / name}")
    summary = read(output / "profile/summary.json")
    if (
        summary != report["profile_summary"]
        or not summary["controlled_network_gate_passed"]
    ):
        raise ValueError(f"block failed the controlled profile gate: {output}")
    if summary["row_count"] != 903 or summary["warmup_rows"] != 3:
        raise ValueError(f"block has incomplete profile counts: {output}")
    # Bind raw journals and plans to the final report, not just derived profiles.
    for relative, expected in report.get("artifact_hashes", {}).items():
        if sha(output / relative) != expected:
            raise ValueError(f"changed completed block input: {relative}")
    if not report.get("artifact_hashes"):
        raise ValueError(f"missing completed-block input hashes: {output}")
    if summary.get("correctness_protocol") == PROTOCOL:
        spec = read(output / "profile_specification.json")
        plan = read(output / "edge/plan.json")
        dataset = plan["config"]["dataset"]
        frozen = load_frozen_manifest(
            dataset["manifest_path"], expected_revision=dataset["revision"]
        )
        verify_reference_evidence(
            (output / spec["same_device_reference_evidence_path"]).resolve(),
            spec["same_device_reference_evidence_sha256"],
            frozen,
            plan["model_contract_sha256"],
        )
    return report


def edge_smoke(config_path: Path, output: Path, requests: int) -> None:
    """Exercise real local inference and checkpoints without any cloud access."""
    if requests < 1:
        raise ValueError("--requests must be positive")
    config = load_experiment_config(config_path)
    source = config.dataset.manifest_path
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if requests > len(rows):
        raise ValueError(f"--requests exceeds manifest size ({len(rows)})")
    selected = rows[:requests]
    for row in selected:
        row["image_path"] = str((source.parent / row["image_path"]).resolve())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "smoke.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "another edge smoke owns this output directory"
            ) from None
        subset = output / "requests_manifest.jsonl"
        if subset.exists():
            saved = [json.loads(line) for line in subset.read_text().splitlines()]
            if saved != selected:
                raise ValueError("smoke subset changed; choose a new --smoke-dir")
        else:
            write_jsonl_atomic(subset, selected)
        manifest = load_frozen_manifest(
            subset, expected_revision=config.dataset.revision
        )
        run_dir = output / "run"
        command = [
            sys.executable,
            "-u",
            "-m",
            "adaptive_vlm.cli",
            "benchmark",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest.path),
            "--mode",
            "edge",
            "--run-dir",
            str(run_dir),
            "--artifact-manifest",
            str(ROOT / "configs/model/moondream2_artifact.yaml"),
            "--cache-dir",
            str(ROOT / ".model-cache"),
            "--offline",
            "--warmups",
            "1",
            "--repetitions",
            "1",
            "--vision-device",
            "cuda:0",
            "--text-device",
            "cuda:0",
            "--max-consecutive-failures",
            "1",
        ]
        if run_dir.exists():
            command.append("--resume")
        log(f"Edge-only smoke: 1 warmup + {requests} measured requests; {output}")
        # Tee live request/model progress to the terminal and a persistent log.
        with (
            (output / "smoke.log").open("a") as logfile,
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                # Retain the writer lock if the launcher itself is killed.
                pass_fds=(lock.fileno(),),
            ) as child,
        ):
            try:
                assert child.stdout is not None
                for line in child.stdout:
                    stamped = f"{now()} {line}"
                    print(stamped, end="", flush=True)
                    logfile.write(stamped)
                    logfile.flush()
                if child.wait():
                    raise RuntimeError(f"edge smoke failed; see {output / 'smoke.log'}")
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        summary = read(run_dir / "summary.json")
        raw = [
            json.loads(line)
            for line in (run_dir / "requests.jsonl").read_text().splitlines()
        ]
        if summary["measured_attempts"] != requests or any(
            row["status"] != "ok" for row in raw
        ):
            raise RuntimeError("edge smoke contains failed or missing attempts")
        log(
            f"Edge smoke passed: {requests}/{requests} measured attempts; "
            "checkpoints saved"
        )


class Supervisor:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.output = Path(settings["output_root"])
        self.child: subprocess.Popen[Any] | None = None
        self.stage = "initializing"
        self.block: str | None = None
        self.shaped = False
        self.reference_outputs: Any = None
        self.available_host_ram_mib: float | None = None

    def status(self, state: str, **extra: Any) -> None:
        payload = {
            "updated_at_utc": now(),
            "pid": os.getpid(),
            "state": state,
            "stage": self.stage,
            "block": self.block,
            "available_host_ram_mib": self.available_host_ram_mib,
            **progress(self.output),
            **extra,
        }
        write_json_atomic(self.output / "status.json", payload)
        log(
            f"{state} {self.stage} block={self.block} "
            f"{payload['measured_attempts']}/8100 ({payload['percent']}%) "
            f"statuses={payload['status_counts']}"
        )

    def check_host_memory(self, *, startup: bool = False) -> None:
        key = (
            "min_host_available_ram_mib"
            if startup
            else "min_runtime_host_available_ram_mib"
        )
        minimum = float(self.settings.get(key, 0))
        if minimum <= 0:
            return
        self.available_host_ram_mib = host_available_ram_mib()
        if self.available_host_ram_mib < minimum:
            raise RuntimeError(
                f"host RAM pressure: available {self.available_host_ram_mib:.0f} MiB, "
                f"required {minimum:.0f} MiB; free memory before resume"
            )

    def wait_for_memory_release(self) -> None:
        # WDDM can release a terminated model process's memory asynchronously.
        for attempt in range(7):
            try:
                self.check_host_memory(startup=True)
                return
            except RuntimeError:
                if attempt == 6:
                    raise
                log("Waiting for model memory release before campaign startup")
                time.sleep(5)

    def run_child(self, command: list[str], *, stall: bool = False) -> None:
        self.child = subprocess.Popen(command, start_new_session=True)
        last_count = progress(self.output)["measured_attempts"]
        last_change = monotonic()
        started = last_change
        try:
            while True:
                try:
                    code = self.child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    self.check_host_memory()
                    self.status("running")
                    if not stall and monotonic() - started > 3600:
                        raise RuntimeError(
                            "preflight child exceeded one hour"
                        ) from None
                    minimum = float(self.settings.get("min_free_gb", 0)) * 1024**3
                    if shutil.disk_usage(self.output).free < minimum:
                        raise RuntimeError(
                            "edge disk space fell below reserve"
                        ) from None
                    count = progress(self.output)["measured_attempts"]
                    if count != last_count:
                        last_count, last_change = count, monotonic()
                    if stall and monotonic() - last_change > float(
                        self.settings["stall_seconds"]
                    ):
                        raise RuntimeError(
                            "no committed progress within stall timeout"
                        ) from None
            if code:
                raise RuntimeError(f"child exited {code}: {shlex.join(command)}")
        finally:
            self.stop_child()

    def stop_child(self) -> None:
        if self.child is not None:
            # The block may have benchmark/GPU-monitor descendants.
            child = self.child
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                child.wait(timeout=15)
            # The group leader can exit while GPU descendants remain alive.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            self.child = None

    def preflight(self) -> dict[str, Any]:
        s = self.settings
        self.check_host_memory(startup=True)
        self.stage = "software and host preflight"
        self.status("preflight")
        for command in ("ssh", "tc", "sudo", "iperf3", "nvidia-smi"):
            if shutil.which(command) is None:
                raise RuntimeError(f"missing edge command: {command}")
        if checked(["git", "status", "--porcelain"]):
            raise RuntimeError("commit/stash source changes before physical profiling")
        commit = checked(["git", "rev-parse", "HEAD"])
        remote_commit = ssh(s, ["git", "-C", s["cloud_repo"], "rev-parse", "HEAD"])
        remote_dirty = ssh(
            s,
            [
                "git",
                "-C",
                s["cloud_repo"],
                "status",
                "--porcelain",
            ],
        )
        if remote_commit != commit or remote_dirty:
            raise RuntimeError("cloud must use the same clean source commit as edge")
        checked(
            [
                "sudo",
                "-n",
                "tc",
                "qdisc",
                "show",
                "dev",
                read(Path(s["grid"]))["edge_interface"],
            ]
        )
        ssh(s, ["sudo", "-n", "tc", "qdisc", "show", "dev", s["cloud_interface"]])
        edge_gpu = checked(["nvidia-smi"])
        cloud_gpu = ssh(s, ["nvidia-smi"])
        minimum = float(s["min_free_gb"]) * 1024**3
        if shutil.disk_usage(self.output).free < minimum:
            raise RuntimeError("insufficient free disk space on edge")
        remote_free = int(
            ssh(
                s,
                [
                    s["cloud_python"],
                    "-c",
                    "import shutil; print(shutil.disk_usage("
                    + repr(s["cloud_repo"])
                    + ").free)",
                ],
            )
        )
        if remote_free < minimum:
            raise RuntimeError("insufficient free disk space on cloud")
        grid = read(Path(s["grid"]))
        config = load_experiment_config(s["config"])
        smoke = load_experiment_config(s["smoke_config"])
        if grid["cloud_host"] != config.transport.host or (
            smoke.transport != config.transport or smoke.model != config.model
        ):
            raise ValueError("grid, experiment and smoke must use matching cloud/model")
        manifest = load_frozen_manifest(
            config.dataset.manifest_path, expected_revision=config.dataset.revision
        )
        split = read(Path(s["split"]))
        if (
            len(manifest.items) != 100
            or manifest.sha256 != split["dataset_manifest_sha256"]
            or model_contract_fingerprint(config.model)
            != split["model_contract_sha256"]
        ):
            raise ValueError("campaign requires the frozen 100-request/model contract")
        if set(grid["blocks"]) != {f"N{i}" for i in range(1, 10)}:
            raise ValueError("campaign requires all nine blocks")
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
        for port in (grid["iperf_port"], grid["echo_port"]):
            import socket

            with socket.create_connection((grid["cloud_host"], port), timeout=10):
                pass
        protocol = grid.get("correctness_protocol", "strict-v1")
        reference_sha = None
        if protocol == PROTOCOL:
            reference_path = Path(grid["same_device_reference_evidence"]).resolve(
                strict=True
            )
            reference_sha = sha(reference_path)
            proof = verify_reference_evidence(
                reference_path,
                reference_sha,
                manifest,
                split["model_contract_sha256"],
                expected_devices={
                    host: config.to_dict()[host]["gpu"] for host in ("edge", "cloud")
                },
            )
            self.reference_outputs = proof["_expected_outputs"]
            pinned_runtime = read(reference_path)["runtime_versions"]
            code = (
                "import json; from adaptive_vlm.benchmark.heterogeneous "
                "import runtime_versions; print(json.dumps(runtime_versions()))"
            )
            edge_runtime = json.loads(checked([s["python"], "-c", code]))
            cloud_runtime = json.loads(ssh(s, [s["cloud_python"], "-c", code]))
            if (
                edge_runtime != pinned_runtime["edge"]
                or cloud_runtime != pinned_runtime["cloud"]
            ):
                raise RuntimeError(
                    "runtime versions differ from same-device reference evidence"
                )
        elif protocol != "strict-v1":
            raise ValueError("unsupported correctness protocol")
        if s.get("require_cloud_wsl_keepalive"):
            release = ssh(s, ["uname", "-r"]).lower()
            if "microsoft" in release:
                sleepers = ssh(s, ["pgrep", "-a", "-x", "sleep"])
                if not any(
                    line.endswith("sleep infinity") for line in sleepers.splitlines()
                ):
                    raise RuntimeError(
                        "keep the cloud WSL foreground sleep infinity open"
                    )
        # Execute in separate processes: CPU tools need not exist in model venv.
        junit = self.output / "tests.xml"
        for arguments in (
            ["ruff", "check", "."],
            ["mypy", "--cache-dir", str(self.output / "mypy_cache")],
            ["pytest", "-q", "-rs", f"--junitxml={junit}"],
        ):
            self.run_child([s["test_python"], "-m", *arguments])
        if any(True for _ in ElementTree.parse(junit).iter("skipped")):
            raise RuntimeError("preflight requires tests to pass without skips")
        contract = {
            "schema_version": 1,
            "same_device_reference_evidence_sha256": reference_sha,
            "git_sha": commit,
            "settings": s,
            "dataset_manifest_sha256": manifest.sha256,
            "input_sha256": {
                key: sha(Path(s[key]))
                for key in (
                    "grid",
                    "split",
                    "config",
                    "smoke_config",
                    "artifact_manifest",
                )
            },
        }
        plan = self.output / "campaign_plan.json"
        if plan.exists() and read(plan) != contract:
            raise ValueError("campaign contract changed; refusing to mix results")
        write_json_atomic(plan, contract)
        write_json_atomic(
            self.output / "host_preflight.json",
            {
                "time": now(),
                "edge_gpu": edge_gpu,
                "cloud_gpu": cloud_gpu,
                "cloud_free_bytes": remote_free,
                "source_commit": commit,
            },
        )
        return contract

    def smoke(self) -> None:
        s = self.settings
        protocol = read(Path(s["grid"])).get("correctness_protocol", "strict-v1")
        self.stage = "three-placement GPU correctness smoke"
        self.status("preflight")
        output = (
            self.output / "preflights" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        )
        output.mkdir(parents=True)
        for mode in MODES:
            self.run_child(
                [
                    s["adaptive_vlm"],
                    "benchmark",
                    "--config",
                    s["smoke_config"],
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
                    "2" if protocol == PROTOCOL else "1",
                    "--max-consecutive-failures",
                    "1",
                    "--vision-device",
                    "cpu" if mode == "cloud" else "cuda:0",
                    "--text-device",
                    "cuda:0" if mode == "edge" else "cpu",
                ]
            )
        report = compare_runs(
            [output / mode for mode in MODES], output=output / "comparison.json"
        )
        if protocol == PROTOCOL:
            if self.reference_outputs is None:
                raise RuntimeError("missing verified same-device reference proof")
            config = load_experiment_config(s["smoke_config"])
            manifest = load_frozen_manifest(
                config.dataset.manifest_path, expected_revision=config.dataset.revision
            )
            assessment = evaluate_rows(
                {
                    mode: [
                        json.loads(line)
                        for line in (output / mode / "requests.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    for mode in MODES
                },
                manifest,
                expected_outputs=self.reference_outputs,
            )
            write_json_atomic(output / "heterogeneous_correctness.json", assessment)
            if not assessment["passed"]:
                raise RuntimeError(f"GPU v2 correctness smoke failed: {output}")
        elif not report["complete_and_equal"]:
            raise RuntimeError(f"GPU correctness smoke failed: {output}")
        log(f"GPU correctness smoke passed: {output}")

    def shape(self, block: dict[str, Any], grid: dict[str, Any]) -> None:
        s = self.settings
        command = [
            "sudo",
            "-n",
            "tc",
            "qdisc",
            "replace",
            "dev",
            s["cloud_interface"],
            "root",
            "netem",
            "delay",
            f"{block['egress_delay_ms']}ms",
            "rate",
            f"{block['rate_mbps']}mbit",
        ]
        cloud_before = ssh(s, ["tc", "qdisc", "show", "dev", s["cloud_interface"]])
        edge_before = checked(["tc", "qdisc", "show", "dev", grid["edge_interface"]])
        for snapshot in (cloud_before, edge_before):
            first = snapshot.splitlines()[0]
            default = any(f"qdisc {kind} " in first for kind in ("mq", "noqueue"))
            owned_netem = (
                "netem" in first and (self.output / "shaping_active.json").exists()
            )
            if not default and not owned_netem:
                raise RuntimeError(
                    "unexpected existing qdisc; inspect before replacing"
                )
        self.shaped = True  # Cleanup even if only one peer succeeds.
        write_json_atomic(
            self.output / "shaping_active.json",
            {
                "block": self.block,
                "time": now(),
            },
        )
        ssh(s, command)
        command[6] = grid["edge_interface"]
        checked(command)
        write_json_atomic(
            self.output / self.block / "cloud_qdisc.json",
            {
                "time": now(),
                "before": cloud_before,
                "after": ssh(s, ["tc", "qdisc", "show", "dev", s["cloud_interface"]]),
                "edge_before": edge_before,
            },
        )

    def cleanup(self, grid: dict[str, Any]) -> list[str]:
        errors = []
        if self.shaped:
            for remote, interface in (
                (False, grid["edge_interface"]),
                (True, self.settings["cloud_interface"]),
            ):
                try:
                    show = ["tc", "qdisc", "show", "dev", interface]
                    call = (lambda cmd: ssh(self.settings, cmd)) if remote else checked
                    if "netem" in call(show):
                        call(
                            [
                                "sudo",
                                "-n",
                                "tc",
                                "qdisc",
                                "del",
                                "dev",
                                interface,
                                "root",
                            ]
                        )
                    if "netem" in call(show):
                        raise RuntimeError("netem remains after cleanup")
                except Exception as exc:
                    errors.append(f"{'cloud' if remote else 'edge'}: {exc}")
        if self.shaped and not errors:
            (self.output / "shaping_active.json").unlink(missing_ok=True)
        return errors

    def execute(self, *, preflight_only: bool) -> None:
        grid = read(Path(self.settings["grid"]))
        terminal = "failed"
        failure: BaseException | None = None
        try:
            self.preflight()
            # A previous hard kill may have left this campaign's netem active.
            self.shaped = (self.output / "shaping_active.json").exists()
            self.smoke()
            self.wait_for_memory_release()
            if preflight_only:
                terminal = "preflight_passed"
                return
            for index in range(1, 10):
                self.block = f"N{index}"
                block_dir = self.output / self.block
                if (block_dir / "validation_report.json").exists():
                    validate_block(block_dir)
                    log(f"{self.block}: validated checkpoint, skipping")
                    continue
                block_dir.mkdir(exist_ok=True)
                self.stage = "apply network block"
                self.status("running")
                self.shape(grid["blocks"][self.block], grid)
                self.stage = "measuring and profiling"
                s = self.settings
                self.run_child(
                    [
                        s["python"],
                        "-u",
                        "scripts/run_physical_profile_block.py",
                        "--block",
                        self.block,
                        "--output-root",
                        s["output_root"],
                        "--grid",
                        s["grid"],
                        "--split",
                        s["split"],
                        "--config",
                        s["config"],
                        "--adaptive-vlm",
                        s["adaptive_vlm"],
                        "--artifact-manifest",
                        s["artifact_manifest"],
                        "--cache-dir",
                        s["cache_dir"],
                    ],
                    stall=True,
                )
                validate_block(block_dir)
            if progress(self.output)["measured_attempts"] != 8100:
                raise RuntimeError("campaign count differs from 8,100")
            terminal = "completed"
        except BaseException as exc:
            failure = exc
            raise
        finally:
            self.stop_child()
            errors = self.cleanup(grid)
            self.status(
                "failed" if errors else terminal,
                error=str(failure) if failure else None,
                cleanup_errors=errors,
            )
            if errors and failure is None:
                raise RuntimeError(f"network cleanup failed: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("start", "run", "preflight", "status", "stop", "edge-smoke")
    )
    parser.add_argument("--settings", type=Path)
    parser.add_argument(
        "--requests",
        type=int,
        default=3,
        help="edge-smoke: number of measured requests (default: 3)",
    )
    parser.add_argument(
        "--smoke-dir", type=Path, default=Path("artifacts/unattended/edge-smoke")
    )
    parser.add_argument(
        "--edge-config",
        type=Path,
        default=Path("configs/experiments/physical_vqa_v1.yaml"),
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "edge-smoke":
        edge_smoke(args.edge_config.resolve(), args.smoke_dir.resolve(), args.requests)
        return
    if args.settings is None:
        parser.error("--settings is required except for edge-smoke")
    s = load_settings(args.settings.resolve())
    output = Path(s["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        status_path = output / "status.json"
        status = read(status_path) if status_path.exists() else {}
        # A persisted heartbeat alone must never imply the process is alive.
        with (output / "campaign.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                alive = False
            except BlockingIOError:
                alive = True
        print(
            json.dumps(
                {**status, **progress(output), "supervisor_running": alive}, indent=2
            )
        )
        return
    if args.action == "start":
        # Parent handshake: wait for the child to acquire the one-writer lock.
        with (output / "campaign.log").open("a") as logfile:
            child = subprocess.Popen(
                [
                    s["python"],
                    "-u",
                    str(Path(__file__).resolve()),
                    "run",
                    "--settings",
                    str(args.settings.resolve()),
                ],
                stdin=subprocess.DEVNULL,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        for _ in range(50):
            if child.poll() is not None:
                raise RuntimeError(f"launch failed; see {output / 'campaign.log'}")
            if (output / "supervisor.pid").exists() and (
                (output / "supervisor.pid").read_text().strip() == str(child.pid)
            ):
                log(f"started pid={child.pid}; tail -F {output / 'campaign.log'}")
                return
            time.sleep(0.1)
        raise RuntimeError("launch handshake timed out; inspect campaign.log")
    with (output / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if args.action == "stop":
                pid = int((output / "supervisor.pid").read_text())
                os.kill(pid, signal.SIGTERM)
                log(f"stop requested for pid={pid}; monitor cleanup in campaign.log")
                return
            raise RuntimeError(
                "another campaign supervisor holds the writer lock"
            ) from None
        if args.action == "stop":
            log("no supervisor is running")
            return
        (output / "supervisor.pid").write_text(str(os.getpid()) + "\n")

        def interrupted(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt(f"received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        supervisor = Supervisor(s)
        supervisor.execute(preflight_only=args.action == "preflight")


if __name__ == "__main__":
    main()
