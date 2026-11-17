"""Run a command on the cloud host: SSH (physical pair), SSM (AWS) or locally.

Every runner takes an argv list and returns stdout, raising
:class:`RemoteCommandError` on a non-zero exit, a timeout or a transport
failure. The supervisor, network shaping and cloud load only see this
interface, so the same campaign code drives the physical pair over SSH and
the AWS pair over SSM, which needs no SSH keys and no inbound port.

The SSM runner uses the AWS CLI (``aws ssm send-command`` with the
``AWS-RunShellScript`` document, then polls ``get-command-invocation``), so
the campaign gains no Python dependency. Commands run as root on the target.
SSM returns at most 24,000 characters of stdout, which is plenty for the
short control commands sent per sub-block; nothing is sent per request.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Execute = Callable[[list[str], float], "subprocess.CompletedProcess[str]"]
_TERMINAL = {"Success", "Failed", "Cancelled", "TimedOut", "Cancelling"}


class RemoteCommandError(RuntimeError):
    """A remote command failed, timed out or could not be delivered."""


class Runner(Protocol):
    def __call__(self, command: Sequence[str]) -> str: ...


def _execute(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )


def _checked(result: subprocess.CompletedProcess[str], what: str) -> str:
    if result.returncode != 0:
        raise RemoteCommandError(
            f"{what} exited {result.returncode}: {result.stderr.strip()[:500]}"
        )
    return result.stdout


@dataclass
class LocalRunner:
    """Run on this machine (loopback rehearsals)."""

    timeout: float = 90.0
    execute: Execute = _execute

    def __call__(self, command: Sequence[str]) -> str:
        argv = list(command)
        try:
            result = self.execute(argv, self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise RemoteCommandError(f"timed out: {shlex.join(argv)}") from exc
        return _checked(result, argv[0])


@dataclass
class SshRunner:
    """Run on a Linux host over key-based SSH (the physical pair)."""

    target: str
    timeout: float = 90.0
    execute: Execute = _execute

    def __call__(self, command: Sequence[str]) -> str:
        argv = [
            "ssh",
            "-oBatchMode=yes",
            "-oConnectTimeout=10",
            "-oServerAliveInterval=15",
            "-oServerAliveCountMax=2",
            self.target,
            shlex.join(list(command)),
        ]
        try:
            result = self.execute(argv, self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise RemoteCommandError(f"ssh timed out: {shlex.join(command)}") from exc
        return _checked(result, f"ssh {self.target}")


@dataclass
class SsmRunner:
    """Run on an EC2 instance through SSM ``AWS-RunShellScript``."""

    instance_id: str
    region: str
    profile: str | None = None
    timeout: float = 120.0
    poll_seconds: float = 1.0
    execute: Execute = _execute
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    working_directory: str | None = None
    _aws: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._aws = ["aws", "--region", self.region, "--output", "json"]
        if self.profile:
            self._aws += ["--profile", self.profile]

    def _call(self, arguments: list[str]) -> dict[str, Any]:
        try:
            result = self.execute([*self._aws, *arguments], 60.0)
        except subprocess.TimeoutExpired as exc:
            raise RemoteCommandError(f"aws {arguments[0]} timed out") from exc
        output: dict[str, Any] = json.loads(_checked(result, f"aws {arguments[1]}"))
        return output

    def __call__(self, command: Sequence[str]) -> str:
        script = shlex.join(list(command))
        if self.working_directory:
            script = f"cd {shlex.quote(self.working_directory)} && {script}"
        parameters = {
            "commands": [script],
            "executionTimeout": [str(int(self.timeout))],
        }
        sent = self._call(
            [
                "ssm",
                "send-command",
                "--instance-ids",
                self.instance_id,
                "--document-name",
                "AWS-RunShellScript",
                "--timeout-seconds",
                "60",
                "--parameters",
                json.dumps(parameters),
            ]
        )
        command_id = sent["Command"]["CommandId"]
        deadline = self.clock() + self.timeout + 30
        while True:
            self.sleep(self.poll_seconds)
            try:
                invocation = self._call(
                    [
                        "ssm",
                        "get-command-invocation",
                        "--command-id",
                        command_id,
                        "--instance-id",
                        self.instance_id,
                    ]
                )
            except RemoteCommandError as exc:
                # The invocation can lag a moment behind send-command.
                if "InvocationDoesNotExist" not in str(exc):
                    raise
                invocation = {"Status": "Pending"}
            status = invocation.get("Status")
            if status in _TERMINAL:
                break
            if self.clock() > deadline:
                raise RemoteCommandError(
                    f"ssm timed out after {self.timeout:.0f}s: {script}"
                )
        if status != "Success" or invocation.get("ResponseCode") != 0:
            raise RemoteCommandError(
                f"ssm {status} (exit {invocation.get('ResponseCode')}): "
                f"{str(invocation.get('StandardErrorContent', '')).strip()[:500]}"
            )
        return str(invocation.get("StandardOutputContent", ""))


def runner_from_settings(settings: Mapping[str, Any]) -> Runner:
    """``cloud_access`` = {"kind": "ssm"|"ssh"|"local", ...}; legacy ``cloud_ssh``."""

    access = settings.get("cloud_access")
    if access is None and settings.get("cloud_ssh"):
        return SshRunner(str(settings["cloud_ssh"]))
    if not isinstance(access, Mapping):
        raise ValueError("set cloud_access (ssm, ssh or local) in the settings")
    kind = access.get("kind")
    if kind == "ssm":
        return SsmRunner(
            instance_id=str(access["instance_id"]),
            region=str(access["region"]),
            profile=access.get("profile"),
            working_directory=access.get("working_directory"),
        )
    if kind == "ssh":
        return SshRunner(str(access["target"]))
    if kind == "local":
        return LocalRunner()
    raise ValueError(f"unknown cloud_access kind: {kind}")
