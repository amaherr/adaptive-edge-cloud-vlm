"""Supervisor side of the cloud load (C1): drive the cloud control command.

Every call goes through the cloud runner (SSM into the inference container on
AWS, SSH on the physical pair, a local shell in loopback rehearsals) and runs
``python -m adaptive_vlm.benchmark.cloud_load``, which prints one JSON
object. The supervisor reaps before every sub-block (C0 and C1), starts the
level after shaping and before the benchmark of a C1 sub-block, and stops it
after the benchmark and at every exit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from adaptive_vlm.benchmark.edge_load import LoadLevel
from adaptive_vlm.campaign.remote import RemoteCommandError, Runner

MODULE = "adaptive_vlm.benchmark.cloud_load"


class CloudLoadFailure(RuntimeError):
    """The cloud load could not be started, checked or stopped."""


class CloudLoad:
    def __init__(
        self,
        runner: Runner,
        python: Sequence[str],
        *,
        heartbeat: str | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.runner = runner
        self.python = list(python)
        self.heartbeat = heartbeat
        self.device = device

    def _call(self, action: str, *extra: str) -> dict[str, Any]:
        command = [*self.python, "-m", MODULE, action]
        if self.heartbeat:
            command += ["--heartbeat-file", self.heartbeat]
        try:
            output = self.runner([*command, *extra])
        except RemoteCommandError as exc:
            raise CloudLoadFailure(f"cloud load {action} failed: {exc}") from exc
        lines = [line for line in output.splitlines() if line.strip()]
        try:
            value = json.loads(lines[-1])
        except (IndexError, ValueError) as exc:
            raise CloudLoadFailure(
                f"cloud load {action} printed no JSON: {output[-300:]!r}"
            ) from exc
        if not isinstance(value, dict):
            raise CloudLoadFailure(f"cloud load {action} printed {value!r}")
        return value

    def start(self, level: LoadLevel) -> dict[str, Any]:
        result = self._call(
            "start",
            "--level",
            level.name,
            "--working-set-mib",
            str(level.working_set_mib),
            "--busy-ms",
            str(level.busy_ms),
            "--idle-ms",
            str(level.idle_ms),
            "--device",
            self.device,
            "--timeout-s",
            "60",
        )
        if result.get("started") is not True:
            raise CloudLoadFailure(f"cloud load {level.name} did not start: {result}")
        return result

    def status(self) -> dict[str, Any]:
        return self._call("status")

    def stop(self) -> dict[str, Any]:
        return self._call("stop")

    def reap(self) -> list[int]:
        """Stop any leftover worker; returns the PIDs that were stopped."""

        return [int(pid) for pid in self._call("reap").get("stopped", [])]
