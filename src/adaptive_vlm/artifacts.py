"""Creation and safe writing of structured experiment artifacts."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.config import ExperimentConfig
from adaptive_vlm.contracts import RunManifest


def create_run_manifest(
    config: ExperimentConfig,
    environment: Mapping[str, Any],
    *,
    now: datetime | None = None,
    dataset_manifest_sha256: str | None = None,
    dependency_lock_sha256: str | None = None,
) -> RunManifest:
    """Create the immutable metadata record for one experiment run."""

    timestamp = (now or datetime.now(UTC)).astimezone(UTC)
    created_at = timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")
    run_id = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    git = environment.get("git")
    git_mapping = git if isinstance(git, Mapping) else {}
    commit = git_mapping.get("commit")
    dirty = git_mapping.get("dirty")

    return RunManifest(
        schema_version=1,
        run_id=run_id,
        created_at_utc=created_at,
        git_commit=commit if isinstance(commit, str) else None,
        git_dirty=dirty if isinstance(dirty, bool) else None,
        config=config.to_dict(),
        environment=environment,
        dataset_manifest_sha256=dataset_manifest_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
    )


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON artifact on the same filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Append one compact JSON record to a JSONL file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_atomic(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a complete JSONL file with atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

