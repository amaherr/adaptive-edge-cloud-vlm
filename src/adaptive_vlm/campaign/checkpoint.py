"""Off-host copies of finished evidence (S3) and watchdog heartbeats.

The campaign's primary copy is the edge's EBS data volume. After each valid
sub-block, :class:`S3Checkpoint` writes a SHA-256 receipt into the sub-block
folder and syncs it to S3, then syncs the campaign's top-level files. Upload
failures are reported to the caller, never raised: losing a checkpoint must
not stop a run whose data is still safe on EBS.

:class:`Heartbeats` keeps the host idle watchdogs from stopping either VM
while a job runs: the edge file is touched on every status update, the cloud
file through the cloud runner at most once per interval.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.campaign.remote import LocalRunner, RemoteCommandError, Runner

RECEIPT = ".checkpoint.json"


def write_receipt(directory: Path) -> dict[str, Any]:
    """SHA-256 of every file under ``directory`` (except the receipt itself)."""

    files = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != RECEIPT and not path.name.endswith(".lock")
    }
    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": files,
    }
    (directory / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


class S3Checkpoint:
    def __init__(
        self,
        s3_uri: str,
        *,
        region: str | None = None,
        profile: str | None = None,
        runner: Runner | None = None,
    ) -> None:
        if not s3_uri.startswith("s3://"):
            raise ValueError("checkpoint s3_uri must start with s3://")
        self.s3_uri = s3_uri.rstrip("/")
        self.runner = runner or LocalRunner(timeout=900)
        self.flags = ["--only-show-errors"]
        if region:
            self.flags += ["--region", region]
        if profile:
            self.flags += ["--profile", profile]

    def _sync(self, source: Path, key: str, *excludes: str) -> None:
        command = ["aws", "s3", "sync", str(source), f"{self.s3_uri}/{key}"]
        for pattern in excludes:
            command += ["--exclude", pattern]
        self.runner([*command, *self.flags])

    def push_subblock(self, output: Path, subblock_id: str) -> str | None:
        """Receipt and sync one sub-block, then the top-level files; error or None."""

        try:
            directory = output / "subblocks" / subblock_id
            write_receipt(directory)
            self._sync(directory, f"subblocks/{subblock_id}", "*.lock")
            self.push_campaign_files(output)
        except (RemoteCommandError, OSError) as exc:
            return f"{subblock_id}: {exc}"
        return None

    def push_campaign_files(self, output: Path) -> None:
        self._sync(output, "", "subblocks/*", "preflights/*", "*.lock", "mypy_cache/*")


class Heartbeats:
    def __init__(
        self,
        *,
        edge_path: str | None,
        cloud_path: str | None,
        cloud_runner: Runner | None,
        cloud_every_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.edge_path = Path(edge_path) if edge_path else None
        self.cloud_path = cloud_path
        self.cloud_runner = cloud_runner
        self.cloud_every_s = cloud_every_s
        self.clock = clock
        self._last_cloud: float | None = None

    def beat(self) -> str | None:
        """Touch both heartbeats as due; returns a cloud error message, if any."""

        if self.edge_path is not None:
            self.edge_path.parent.mkdir(parents=True, exist_ok=True)
            self.edge_path.touch()
        if self.cloud_path is None or self.cloud_runner is None:
            return None
        now = self.clock()
        if self._last_cloud is not None and now - self._last_cloud < self.cloud_every_s:
            return None
        try:
            self.cloud_runner(["touch", self.cloud_path])
        except RemoteCommandError as exc:
            return f"cloud heartbeat failed: {exc}"
        self._last_cloud = now
        return None
