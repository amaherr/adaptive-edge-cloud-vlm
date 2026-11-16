"""W8 local rehearsal: the full G2 v2 supervisor with injected faults.

Runs the loopback rehearsal campaign (fake model, real GPU load worker) and:

1. hard-kills the supervisor (SIGKILL) part-way through the first sub-block,
   then checks its benchmark and load-worker children exit on their own;
2. restarts the campaign, which resumes the interrupted sub-block;
3. kills the edge-load worker (SIGKILL) during the loaded sub-block, which
   must end that attempt with exit code 76 and be resumed automatically;
4. waits for completion and verifies every sub-block's journals (no gaps, no
   duplicates, one record per planned attempt), the restart log, the absence
   of stray processes, and that the derived Predictor V2 and oracle tables
   build without manual edits.

The report is written to ``<output_root>-faults-report.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from adaptive_vlm.campaign.supervisor import load_settings
from adaptive_vlm.models.fake import DELAY_ENVIRONMENT
from adaptive_vlm.process import find_processes

ROOT = Path(__file__).resolve().parents[2]


def wait(condition: Any, timeout: float, what: str, interval: float = 0.2) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(interval)
    raise TimeoutError(f"timed out waiting for {what}")


def committed(run_dir: Path) -> int:
    return sum(
        path.read_bytes().count(b"\n") for path in run_dir.glob("*/requests.jsonl")
    )


def supervisor_running(output: Path) -> bool:
    import fcntl

    with (output / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def campaign(action: str, settings: Path, env: dict[str, str]) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/campaign/run_g2v2_campaign.py"),
            action,
            "--settings",
            str(settings),
        ],
        check=True,
        env=env,
    )


def verify_subblock(directory: Path) -> dict[str, Any]:
    report = json.loads((directory / "validation_report.json").read_text())
    positions: list[int] = []
    for placement in ("edge", "cloud", "distributed"):
        plan = json.loads((directory / "run" / placement / "plan.json").read_text())
        rows = [
            json.loads(line)
            for line in (directory / "run" / placement / "requests.jsonl")
            .read_text()
            .splitlines()
        ]
        trials = [row["trial_id"] for row in rows]
        if sorted(trials) != sorted(t["trial_id"] for t in plan["schedule"]):
            raise AssertionError(f"{directory.name}/{placement}: gaps or duplicates")
        positions += [row["interleave_position"] for row in rows]
    if sorted(positions) != list(range(1, len(positions) + 1)):
        raise AssertionError(f"{directory.name}: interleave positions not contiguous")
    restarts = (
        [
            json.loads(line)
            for line in (directory / "restarts.jsonl").read_text().splitlines()
        ]
        if (directory / "restarts.jsonl").exists()
        else []
    )
    return {
        "valid": report["valid"],
        "reasons": report["reasons"],
        "attempts": len(positions),
        "restart_exit_codes": [r["exit_code"] for r in restarts],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        type=Path,
        default=ROOT / "configs/campaigns/g2v2_loopback_rehearsal.json",
    )
    parser.add_argument("--delay-ms", type=float, default=400)
    parser.add_argument("--fresh", action="store_true", help="delete earlier output")
    args = parser.parse_args()
    os.chdir(ROOT)
    settings = load_settings(args.settings.resolve(), ROOT)
    if settings["mode"] != "loopback":
        parser.error("fault injection runs only in loopback mode")
    output = Path(settings["output_root"])
    derived = output.with_name(output.name + "-derived")
    report_path = output.with_name(output.name + "-faults-report.json")
    if output.exists() or derived.exists():
        if not args.fresh:
            parser.error(f"{output} exists; pass --fresh to replace it")
        shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(derived, ignore_errors=True)
    output.mkdir(parents=True)
    env = {**os.environ, DELAY_ENVIRONMENT: str(args.delay_ms)}
    report: dict[str, Any] = {"delay_ms": args.delay_ms, "steps": []}

    def step(name: str, **details: Any) -> None:
        report["steps"].append({"step": name, "time": time.time(), **details})
        print(f"== {name} {json.dumps(details)}", flush=True)

    server_log = (output / "fake-server.log").open("w")
    server = subprocess.Popen(
        [
            settings["adaptive_vlm"],
            "adaptive-server",
            "--config",
            str(ROOT / "configs/experiments/fake_g2v2_rehearsal.yaml"),
            "--artifact-manifest",
            settings["artifact_manifest"],
            "--bind-host",
            "127.0.0.1",
            "--report-resource-state",
        ],
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        wait(
            lambda: '"ready": true' in (output / "fake-server.log").read_text(),
            180,
            "the fake server",
        )
        step("fake server ready")

        campaign("start", args.settings, env)
        first = output / "subblocks"
        run_dir = wait(
            lambda: (
                next(
                    (d / "run" for d in first.glob("*") if committed(d / "run") >= 4),
                    None,
                )
                if first.exists()
                else None
            ),
            300,
            "attempts in the first sub-block",
        )
        supervisor_pid = int((output / "supervisor.pid").read_text())
        before = committed(run_dir)
        os.kill(supervisor_pid, signal.SIGKILL)
        step(
            "hard kill",
            supervisor_pid=supervisor_pid,
            subblock=run_dir.parent.name,
            committed_rows=before,
        )
        try:
            wait(
                lambda: not find_processes(str(output / "subblocks")),
                30,
                "children to exit after the supervisor died",
            )
            step("children exited with their supervisor")
        except TimeoutError:
            step("children survived", pids=find_processes(str(output / "subblocks")))
        wait(lambda: not supervisor_running(output), 30, "the campaign lock")

        campaign("start", args.settings, env)
        step("restarted")
        heartbeat = wait(
            lambda: next(
                (
                    path
                    for path in first.glob("*-L1-*/.run-edge-load/heartbeat.json")
                    if json.loads(path.read_text() or "{}").get("status") == "running"
                    and committed(path.parent.parent / "run") >= 4
                ),
                None,
            ),
            900,
            "the loaded sub-block",
            interval=0.1,
        )
        worker = int(json.loads(heartbeat.read_text())["pid"])
        os.kill(worker, signal.SIGKILL)
        step(
            "load worker killed",
            pid=worker,
            subblock=heartbeat.parent.parent.name,
            committed_rows=committed(heartbeat.parent.parent / "run"),
        )
        wait(lambda: not supervisor_running(output), 1800, "the campaign to finish", 1)
    finally:
        server.terminate()
        server.wait(timeout=30)
        server_log.close()

    status = json.loads((output / "status.json").read_text())
    subblocks = {
        d.name: verify_subblock(d) for d in sorted((output / "subblocks").iterdir())
    }
    strays = find_processes(str(output / "subblocks"))
    derive = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/campaign/derive_g2v2_tables.py"),
            "all",
            "--campaign",
            str(output),
            "--contract",
            str(ROOT / "configs/predictor/rehearsal_dataset_contract.json"),
            "--output",
            str(derived),
            "--sealed-root",
            str(Path("/tmp") / f"{output.name}-sealed"),
        ],
        capture_output=True,
        text=True,
    )
    audit_path = derived / "partitions" / "partition_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    restart_codes = Counter(
        code for block in subblocks.values() for code in block["restart_exit_codes"]
    )
    checks = {
        "campaign_completed": status["state"] == "completed",
        "all_attempts_recorded": status["measured_attempts"]
        == status["target_attempts"],
        "all_subblocks_valid": all(b["valid"] for b in subblocks.values()),
        "children_exited_with_supervisor": any(
            s["step"] == "children exited with their supervisor"
            for s in report["steps"]
        ),
        "load_failure_restarted_with_76": restart_codes.get(76, 0) >= 1,
        "no_stray_processes": not strays,
        "no_invalid_reruns": not (output / "invalid").exists(),
        "derived_tables_built": derive.returncode == 0
        and audit.get("evaluation_profiles_opened") == []
        and (derived / "oracle-training" / "oracle_cells.json").exists(),
    }
    report.update(
        status=status,
        subblocks=subblocks,
        stray_processes=strays,
        derive_exit_code=derive.returncode,
        derive_stderr_tail=derive.stderr[-2000:],
        partition_counts=audit.get("classification"),
        checks=checks,
        passed=all(checks.values()),
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": checks, "subblocks": subblocks}, indent=2))
    print(f"report: {report_path}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
