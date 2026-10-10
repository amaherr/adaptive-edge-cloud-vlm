from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from adaptive_vlm.artifacts import (
    append_jsonl,
    create_run_manifest,
    write_json_atomic,
)
from adaptive_vlm.config import load_experiment_config

ROOT = Path(__file__).parents[1]


def test_write_json_atomic_replaces_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "result.json"
    write_json_atomic(destination, {"version": 1})
    write_json_atomic(destination, {"version": 2})

    assert json.loads(destination.read_text()) == {"version": 2}
    assert list(destination.parent.glob("*.tmp")) == []


def test_append_jsonl_writes_one_record_per_line(tmp_path: Path) -> None:
    destination = tmp_path / "requests.jsonl"
    append_jsonl(destination, {"request_id": "one"})
    append_jsonl(destination, {"request_id": "two"})

    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["request_id"] for row in rows] == ["one", "two"]


def test_manifest_captures_git_state() -> None:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk1_smoke.yaml"
    )
    manifest = create_run_manifest(
        config,
        {"git": {"commit": "abc123", "dirty": True}},
        now=datetime(2026, 10, 10, 12, 0, tzinfo=UTC),
    )

    assert manifest.run_id.startswith("20261010T120000Z-")
    assert manifest.git_commit == "abc123"
    assert manifest.git_dirty is True

