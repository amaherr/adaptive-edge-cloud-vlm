"""Unattended G2 v2 campaign: start, run, preflight, status or stop.

``start`` detaches a supervisor and returns once it holds the writer lock;
``run`` stays in the foreground; ``preflight`` checks everything and runs the
correctness smoke without measuring. See docs/runbooks/physical-campaign.md.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from adaptive_vlm.campaign.supervisor import (
    CampaignSupervisor,
    load_settings,
    log,
    progress,
)
from adaptive_vlm.environment import disable_core_dumps

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("start", "run", "preflight", "status", "stop")
    )
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    disable_core_dumps()  # Inherited by every child process.
    settings = load_settings(args.settings.resolve(), ROOT)
    output = Path(settings["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        path = output / "status.json"
        status = json.loads(path.read_text()) if path.exists() else {}
        with (output / "campaign.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                alive = False
            except BlockingIOError:
                alive = True
        target = status.get("target_attempts")
        print(
            json.dumps(
                {**status, **progress(output, target), "supervisor_running": alive},
                indent=2,
            )
        )
        return
    if args.action == "start":
        with (output / "campaign.log").open("a") as logfile:
            child = subprocess.Popen(
                [
                    settings["python"],
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
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError(f"launch failed; see {output / 'campaign.log'}")
            pid_file = output / "supervisor.pid"
            if pid_file.exists() and pid_file.read_text().strip() == str(child.pid):
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
                log(f"stop requested for pid={pid}; cleanup is logged in campaign.log")
                return
            raise RuntimeError("another campaign supervisor holds the lock") from None
        if args.action == "stop":
            log("no supervisor is running")
            return
        (output / "supervisor.pid").write_text(f"{os.getpid()}\n")

        def interrupted(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt(f"received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        CampaignSupervisor(settings, root=ROOT).execute(
            preflight_only=args.action == "preflight"
        )


if __name__ == "__main__":
    main()
