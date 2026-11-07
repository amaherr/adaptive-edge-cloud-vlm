"""Crash boundaries, conditional writers, strict recovery and timing isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.runner import run_benchmark
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.ops.checkpoints import CheckpointSession, read_checkpoint, restore
from adaptive_vlm.ops.store import MissingObject, Object, ObjectConflict

ROOT = Path(__file__).parents[1]
IMAGE = "test/research@sha256:" + "a" * 64


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, Object] = {}
        self.version = 0
        self.fail_kind: str | None = None

    def get(self, key: str) -> Object:
        if key not in self.objects:
            raise MissingObject(key)
        return self.objects[key]

    def put(self, key: str, data: bytes, *, etag: str | None = None) -> str:
        if self.fail_kind and self.fail_kind in key:
            raise RuntimeError("injected object-store failure")
        previous = self.objects.get(key)
        if (etag is None and previous is not None) or (
            etag is not None and (previous is None or previous.etag != etag)
        ):
            raise ObjectConflict(key)
        self.version += 1
        new_etag = str(self.version)
        self.objects[key] = Object(data, new_etag)
        return new_etag

    def delete(self, key: str, *, etag: str) -> None:
        if self.get(key).etag != etag:
            raise ObjectConflict(key)
        del self.objects[key]


def settings(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"
    )
    manifest = load_frozen_manifest(config.dataset.manifest_path)
    contract = {
        "campaign_id": "offline-acceptance",
        "block_id": "N1",
        "policy": "edge",
        "host_pair_id": "test-pair",
        "provider": "test",
        "image_digest": IMAGE,
        "model_sha256": model_contract_fingerprint(config.model),
        "dataset_sha256": manifest.sha256,
        "split_sha256": "b" * 64,
        "grid_sha256": "c" * 64,
        "hardware_contract_sha256": "d" * 64,
    }
    kwargs = {
        "config": config,
        "manifest": manifest,
        "policy": LocalExecutionPolicy(FakeModelAdapter()),
        "run_manifest": create_run_manifest(
            config,
            {"container": {"declared_image_reference": IMAGE}},
            dataset_manifest_sha256=manifest.sha256,
        ),
        "run_dir": tmp_path / "run",
        "warmups": 1,
        "repetitions": 1,
        "pipeline": "synchronous",
    }
    return contract, kwargs


def test_restore_then_existing_resume_skips_every_committed_trial(tmp_path):
    contract, kwargs = settings(tmp_path)
    store = MemoryStore()
    session = CheckpointSession(store, "acceptance/run", contract, every_trials=2)
    result = run_benchmark(**kwargs, checkpoint=session)
    target = tmp_path / "restored"
    restore(store, "acceptance/run", contract, target)
    for name in ["plan.json", "run_manifest.json", "requests.jsonl", "summary.json"]:
        assert (target / name).read_bytes() == (kwargs["run_dir"] / name).read_bytes()
    resumed = run_benchmark(
        **{**kwargs, "run_dir": target},
        resume=True,
        checkpoint=CheckpointSession(store, "acceptance/run", contract),
    )
    assert resumed == result
    with pytest.raises(FileExistsError):
        restore(store, "acceptance/run", contract, target)
    assert "acceptance/run/writer.json" not in store.objects


@pytest.mark.parametrize("failure", ["segments/", "commits/", "head.json"])
def test_failed_upload_never_advances_head_or_schedules_more_trials(tmp_path, failure):
    contract, kwargs = settings(tmp_path)
    store = MemoryStore()
    run_benchmark(**kwargs)
    raw = kwargs["run_dir"] / "requests.jsonl"
    all_rows = raw.read_bytes()
    raw.write_bytes(b"")
    session = CheckpointSession(store, "run", contract, every_trials=1)
    session.start(kwargs["run_dir"])
    old_head = store.get("run/head.json")
    raw.write_bytes(all_rows)
    store.fail_kind = failure
    with pytest.raises(RuntimeError, match="injected"):
        session.flush(force=True)
    assert store.get("run/head.json") == old_head
    assert read_checkpoint(store, "run", contract)[0] == b""
    session.close()


def test_corruption_and_contract_changes_refuse_recovery(tmp_path):
    contract, kwargs = settings(tmp_path)
    store = MemoryStore()
    run_benchmark(**kwargs, checkpoint=CheckpointSession(store, "run", contract))
    for field in [
        "image_digest",
        "model_sha256",
        "dataset_sha256",
        "hardware_contract_sha256",
        "grid_sha256",
    ]:
        changed = {
            **contract,
            field: ("other@sha256:" if field == "image_digest" else "") + "e" * 64,
        }
        with pytest.raises(ValueError, match="contract differs"):
            read_checkpoint(store, "run", changed)
    key = next(k for k in store.objects if "/segments/" in k)
    store.objects[key] = Object(b"corrupted", "different")
    with pytest.raises(ValueError, match="hash mismatch"):
        restore(store, "run", contract, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_lock_is_exclusive_and_wrong_etag_cannot_release_it(tmp_path):
    contract, kwargs = settings(tmp_path)
    store = MemoryStore()
    run_benchmark(**kwargs)
    first = CheckpointSession(store, "run", contract)
    first.start(kwargs["run_dir"])
    second = CheckpointSession(store, "run", contract)
    with pytest.raises(ObjectConflict):
        second.start(kwargs["run_dir"])
    with pytest.raises(ObjectConflict):
        store.delete("run/writer.json", etag="stale")
    first.close()
    second.start(kwargs["run_dir"])
    second.close()


def test_timer_flushes_fewer_than_25_rows_before_next_attempt(tmp_path):
    contract, kwargs = settings(tmp_path)
    store = MemoryStore()
    run_benchmark(**kwargs)
    raw = kwargs["run_dir"] / "requests.jsonl"
    row = raw.read_bytes().splitlines(keepends=True)[0]
    raw.write_bytes(b"")
    clock = [0.0]
    session = CheckpointSession(store, "run", contract, clock=lambda: clock[0])
    session.start(kwargs["run_dir"])
    raw.write_bytes(row)
    session.after_attempt()
    assert read_checkpoint(store, "run", contract)[0] == b""
    clock[0] = 300
    session.before_attempt()
    assert read_checkpoint(store, "run", contract)[0] == row
    store.put("run/STOP", b"stop")
    with pytest.raises(RuntimeError, match="STOP"):
        session.before_attempt()
    session.close()


def test_local_tail_restores_only_committed_rows_and_rejects_rollback(tmp_path):
    contract, kwargs = settings(tmp_path)
    store = MemoryStore()
    run_benchmark(**kwargs)
    raw = kwargs["run_dir"] / "requests.jsonl"
    rows = raw.read_bytes().splitlines(keepends=True)
    raw.write_bytes(rows[0])
    session = CheckpointSession(store, "run", contract)
    session.start(kwargs["run_dir"])
    raw.write_bytes(b"".join(rows))
    # An uncommitted local tail is absent from a restored machine.
    restore(store, "run", contract, tmp_path / "restore")
    assert (tmp_path / "restore/requests.jsonl").read_bytes() == rows[0]
    raw.write_bytes(b"")
    with pytest.raises(ValueError, match="truncated"):
        session.flush(force=True)
    session.close()


def test_store_failure_after_attempt_preserves_local_trial_and_stops(tmp_path):
    contract, kwargs = settings(tmp_path)
    store = MemoryStore()
    policy = kwargs["policy"]
    original = policy.infer
    calls = []

    def fail_store_after_infer(request):
        calls.append(request.request_id)
        result = original(request)
        store.fail_kind = "segments/"
        return result

    policy.infer = fail_store_after_infer
    with pytest.raises(RuntimeError, match="injected"):
        run_benchmark(
            **kwargs,
            checkpoint=CheckpointSession(store, "run", contract, every_trials=1),
        )
    rows = (kwargs["run_dir"] / "requests.jsonl").read_text().splitlines()
    assert len(calls) == len(rows) == 1
    assert json.loads(rows[0])["status"] == "ok"
    assert read_checkpoint(store, "run", contract)[0] == b""


def test_summary_can_be_regenerated_after_final_trial(tmp_path):
    contract, kwargs = settings(tmp_path)
    store = MemoryStore()
    run_benchmark(**kwargs)
    (kwargs["run_dir"] / "summary.json").unlink()
    session = CheckpointSession(store, "run", contract)
    session.start(kwargs["run_dir"])
    session.close()
    restore(store, "run", contract, tmp_path / "restore")
    result = run_benchmark(**{**kwargs, "run_dir": tmp_path / "restore"}, resume=True)
    assert result["execution_state"] == "complete"
