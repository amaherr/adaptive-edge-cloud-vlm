"""Remote command runners: SSM through a fake AWS CLI, SSH quoting, settings."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from adaptive_vlm.campaign.remote import (
    LocalRunner,
    RemoteCommandError,
    SshRunner,
    SsmRunner,
    runner_from_settings,
)


def _done(stdout: str = "", code: int = 0, stderr: str = "") -> Any:
    return subprocess.CompletedProcess([], code, stdout, stderr)


class FakeAws:
    """Records aws CLI calls and replays scripted get-command-invocation states."""

    def __init__(self, states: list[dict[str, Any] | Exception]) -> None:
        self.calls: list[list[str]] = []
        self.states = states

    def __call__(self, argv: list[str], timeout: float) -> Any:
        self.calls.append(argv)
        if "send-command" in argv:
            return _done(json.dumps({"Command": {"CommandId": "cmd-1"}}))
        state = self.states.pop(0)
        if isinstance(state, Exception):
            return _done("", 254, str(state))
        return _done(json.dumps(state))


def _runner(fake: FakeAws, **kwargs: Any) -> SsmRunner:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    return SsmRunner(
        instance_id="i-0123",
        region="us-east-1",
        execute=fake,
        sleep=sleep,
        clock=lambda: now[0],
        **kwargs,
    )


def test_ssm_sends_then_polls_until_success() -> None:
    fake = FakeAws(
        [
            Exception("An error occurred (InvocationDoesNotExist)"),
            {"Status": "InProgress"},
            {
                "Status": "Success",
                "ResponseCode": 0,
                "StandardOutputContent": "qdisc mq 0: root\n",
            },
        ]
    )
    runner = _runner(fake, profile="cloud-infrastructure", working_directory="/app")
    assert runner(["tc", "qdisc", "show", "dev", "ens5"]) == "qdisc mq 0: root\n"
    send = fake.calls[0]
    assert send[:7] == [
        "aws", "--region", "us-east-1", "--output", "json",
        "--profile", "cloud-infrastructure",
    ]  # fmt: skip
    parameters = json.loads(send[send.index("--parameters") + 1])
    assert parameters["commands"] == ["cd /app && tc qdisc show dev ens5"]
    assert send[send.index("--instance-ids") + 1] == "i-0123"
    assert len(fake.calls) == 4


def test_ssm_reports_a_failed_command() -> None:
    fake = FakeAws(
        [{"Status": "Failed", "ResponseCode": 2, "StandardErrorContent": "no such dev"}]
    )
    with pytest.raises(RemoteCommandError, match=r"Failed .*exit 2.*no such dev"):
        _runner(fake)(["tc", "qdisc", "show", "dev", "bogus"])


def test_ssm_times_out_and_surfaces_cli_errors() -> None:
    fake = FakeAws([{"Status": "InProgress"}] * 500)
    with pytest.raises(RemoteCommandError, match="timed out"):
        _runner(fake, timeout=5, poll_seconds=1)(["sleep", "999"])
    broken = FakeAws([Exception("AccessDeniedException: not authorized")])
    with pytest.raises(RemoteCommandError, match="AccessDenied"):
        _runner(broken)(["true"])


def test_ssh_quotes_the_remote_command() -> None:
    seen: list[list[str]] = []

    def execute(argv: list[str], timeout: float) -> Any:
        seen.append(argv)
        return _done("ok")

    assert SshRunner("edge@cloud", execute=execute)(["echo", "a b;c"]) == "ok"
    assert seen[0][-2:] == ["edge@cloud", "echo 'a b;c'"]
    failing = SshRunner("edge@cloud", execute=lambda argv, t: _done("", 255, "refused"))
    with pytest.raises(RemoteCommandError, match="refused"):
        failing(["true"])


def test_local_runner_and_timeout() -> None:
    assert LocalRunner()(["echo", "hello"]).strip() == "hello"

    def slow(argv: list[str], timeout: float) -> Any:
        raise subprocess.TimeoutExpired(argv, timeout)

    with pytest.raises(RemoteCommandError, match="timed out"):
        LocalRunner(execute=slow)(["sleep", "5"])


def test_runner_from_settings() -> None:
    ssm = runner_from_settings(
        {"cloud_access": {"kind": "ssm", "instance_id": "i-1", "region": "us-east-1"}}
    )
    assert isinstance(ssm, SsmRunner) and ssm.instance_id == "i-1"
    assert isinstance(
        runner_from_settings({"cloud_access": {"kind": "ssh", "target": "a@b"}}),
        SshRunner,
    )
    assert isinstance(runner_from_settings({"cloud_ssh": "a@b"}), SshRunner)
    assert isinstance(
        runner_from_settings({"cloud_access": {"kind": "local"}}), LocalRunner
    )
    with pytest.raises(ValueError, match="unknown"):
        runner_from_settings({"cloud_access": {"kind": "telnet"}})
    with pytest.raises(ValueError, match="cloud_access"):
        runner_from_settings({})
