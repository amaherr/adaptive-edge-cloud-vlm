"""Process lifetime safeguards for unattended campaigns."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from adaptive_vlm.process import exclusive_lock, exit_with_parent, find_processes

LINUX = sys.platform.startswith("linux")


def test_exclusive_lock_refuses_a_second_holder(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    with (
        exclusive_lock(path, "run writer"),
        pytest.raises(RuntimeError, match="run writer lock"),
        exclusive_lock(path, "run writer"),
    ):
        pass
    with exclusive_lock(path, "run writer"):
        pass


def test_find_processes_matches_command_lines(tmp_path: Path) -> None:
    for pid, command in ((101, b"python\0bench\0/out/subblocks/T01"), (102, b"bash")):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cmdline").write_bytes(command)
    (tmp_path / "self").mkdir()
    assert find_processes("/out/subblocks", proc=tmp_path) == [101]


@pytest.mark.skipif(not LINUX, reason="prctl is Linux-only")
def test_child_exits_when_its_parent_is_killed(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    grandchild = (
        "import os, time; from adaptive_vlm.process import exit_with_parent; "
        f"exit_with_parent(); open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(60)"
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(60)",
        ]
    )
    deadline = time.monotonic() + 20
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline, "grandchild did not start"
        time.sleep(0.05)
    pid = int(pid_file.read_text())
    parent.kill()
    parent.wait()
    deadline = time.monotonic() + 10
    while Path(f"/proc/{pid}").exists() and (
        Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    ):
        assert time.monotonic() < deadline, "grandchild outlived its parent"
        time.sleep(0.05)
    assert exit_with_parent() is True
