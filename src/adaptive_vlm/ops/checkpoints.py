"""Immutable, incremental checkpoints outside inference timing regions.

A persistent conditional writer lock never expires automatically. After a crash,
stop/verify destruction of the old writer before explicitly releasing its lock.
Recovery follows only the committed head; orphan objects are never authoritative.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from time import monotonic
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.ops.store import MissingObject, ObjectConflict, Store


def encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_contract(contract: dict[str, Any]) -> None:
    for field in (
        "campaign_id",
        "block_id",
        "policy",
        "host_pair_id",
        "provider",
        "image_digest",
        "model_sha256",
        "dataset_sha256",
        "split_sha256",
        "grid_sha256",
        "hardware_contract_sha256",
    ):
        if not isinstance(contract.get(field), str) or not contract[field].strip():
            raise ValueError(f"checkpoint contract requires {field}")
    if contract["policy"] not in {"edge", "cloud", "distributed"}:
        raise ValueError("invalid checkpoint policy")
    if not re.fullmatch(r".+@sha256:[a-f0-9]{64}", contract["image_digest"]):
        raise ValueError("checkpoint requires an immutable image digest")
    for field in (
        "model_sha256",
        "dataset_sha256",
        "split_sha256",
        "grid_sha256",
        "hardware_contract_sha256",
    ):
        if not re.fullmatch(r"[a-f0-9]{64}", contract[field]):
            raise ValueError(f"checkpoint {field} must be SHA-256")


def _rows(
    data: bytes, plan: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in data.splitlines()]
    schedule = {trial["trial_id"]: trial for trial in plan["schedule"]}
    seen = set()
    for row in rows:
        trial_id = row.get("trial_id")
        if trial_id in seen or trial_id not in schedule:
            raise ValueError("checkpoint has duplicate or unknown trial keys")
        seen.add(trial_id)
        trial = schedule[trial_id]
        for field in ("request_id", "repetition", "warmup"):
            if row.get(field) != trial[field]:
                raise ValueError("checkpoint trial differs from schedule")
        if row.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("checkpoint row has a different plan")
        if row.get("run_id") != manifest["run_id"] or not row.get("status"):
            raise ValueError("checkpoint row run/status is invalid")
    return rows


class CheckpointSession:
    def __init__(
        self,
        store: Store,
        prefix: str,
        contract: dict[str, Any],
        *,
        every_trials: int = 25,
        every_seconds: float = 300,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        validate_contract(contract)
        if (
            not prefix
            or prefix.startswith("/")
            or any(part in {"", ".", ".."} for part in prefix.split("/"))
        ):
            raise ValueError("invalid checkpoint prefix")
        if not 1 <= every_trials <= 25 or not 0 < every_seconds <= 300:
            raise ValueError("checkpoint cadence cannot exceed 25 trials / 300 seconds")
        self.store = store
        self.prefix = prefix
        self.contract = contract
        self.every_trials = every_trials
        self.every_seconds = every_seconds
        self.clock = clock
        self.owner = uuid.uuid4().hex
        self.lock_etag: str | None = None
        self.head_etag: str | None = None
        self.previous: dict[str, str] | None = None
        self.sequence = 0
        self.committed = b""
        self.last_flush = clock()
        self.run_dir: Path | None = None

    def _key(self, relative: str) -> str:
        return f"{self.prefix}/{relative}"

    def _immutable(self, kind: str, data: bytes) -> dict[str, str]:
        digest = sha(data)
        key = self._key(f"{kind}/{digest}")
        with suppress(ObjectConflict):
            self.store.put(key, data)
        if self.store.get(key).data != data:
            raise ValueError("uploaded checkpoint failed read-back hash verification")
        return {"key": key, "sha256": digest}

    def start(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        try:
            self.lock_etag = self.store.put(
                self._key("writer.json"), encode({"owner": self.owner})
            )
            try:
                head = self.store.get(self._key("head.json"))
            except MissingObject:
                head = None
            if head is not None:
                self.head_etag = head.etag
                self.previous = json.loads(head.data)
                self.committed, _, commit = read_checkpoint(
                    self.store, self.prefix, self.contract
                )
                self.sequence = commit["sequence"]
            # Upload immutable plan/manifest before scheduling the first attempt.
            self.flush(force=True)
        except BaseException:
            self.close()
            raise

    def before_attempt(self) -> None:
        if self.run_dir is None or self.lock_etag is None:
            raise RuntimeError("checkpoint session is not active")
        if self.clock() - self.last_flush >= self.every_seconds:
            self.flush(force=True)
        # This control-plane marker is polled outside the timed inference region.
        try:
            self.store.get(self._key("STOP"))
        except MissingObject:
            return
        self.flush(force=True)
        raise RuntimeError("remote STOP requested; checkpointed before scheduling")

    def after_attempt(self) -> None:
        self.flush()

    def flush(self, *, force: bool = False) -> None:
        if self.run_dir is None or self.lock_etag is None:
            raise RuntimeError("checkpoint session is not active")
        lock = self.store.get(self._key("writer.json"))
        if lock.etag != self.lock_etag:
            raise ObjectConflict("checkpoint writer lock was replaced")
        raw_path = self.run_dir / "requests.jsonl"
        raw = raw_path.read_bytes() if raw_path.exists() else b""
        if not raw.startswith(self.committed) or (raw and not raw.endswith(b"\n")):
            raise ValueError("journal is truncated, divergent or has an incomplete row")
        delta = raw[len(self.committed) :]
        count = len(delta.splitlines())
        if (
            not force
            and count < self.every_trials
            and (self.clock() - self.last_flush < self.every_seconds)
        ):
            return
        plan = json.loads((self.run_dir / "plan.json").read_bytes())
        manifest = json.loads((self.run_dir / "run_manifest.json").read_bytes())
        _rows(raw, plan, manifest)
        if plan["mode"] != self.contract["policy"]:
            raise ValueError("checkpoint policy differs from plan")
        if plan["dataset_manifest_sha256"] != self.contract["dataset_sha256"]:
            raise ValueError("checkpoint dataset differs from contract")
        if plan["model_contract_sha256"] != self.contract["model_sha256"]:
            raise ValueError("checkpoint model differs from contract")
        if (
            plan.get("container", {}).get("declared_image_reference")
            != self.contract["image_digest"]
        ):
            raise ValueError("checkpoint image differs from plan")
        files = {}
        for name in ("plan.json", "run_manifest.json", "summary.json"):
            path = self.run_dir / name
            if path.exists():
                files[name] = self._immutable("files", path.read_bytes())
        segment = self._immutable("segments", delta) if delta else None
        commit = {
            "schema_version": 1,
            "sequence": self.sequence + 1,
            "contract": self.contract,
            "previous": self.previous,
            "files": files,
            "segment": segment,
            "journal_sha256": sha(raw),
            "row_count": len(raw.splitlines()),
        }
        ref = self._immutable("commits", encode(commit))
        # Only this CAS advances authoritative state. Crashes before it leave
        # harmless orphan objects that recovery never interprets as committed.
        self.head_etag = self.store.put(
            self._key("head.json"), encode(ref), etag=self.head_etag
        )
        self.previous = ref
        self.sequence += 1
        self.committed = raw
        self.last_flush = self.clock()
        write_json_atomic(
            self.run_dir / "checkpoint_index.json",
            {
                "head": ref,
                "row_count": commit["row_count"],
                "journal_sha256": commit["journal_sha256"],
            },
        )

    def finish(self) -> None:
        self.flush(force=True)

    def close(self) -> None:
        if self.lock_etag is not None:
            self.store.delete(self._key("writer.json"), etag=self.lock_etag)
            self.lock_etag = None


def read_checkpoint(
    store: Store, prefix: str, contract: dict[str, Any]
) -> tuple[bytes, dict[str, bytes], dict[str, Any]]:
    validate_contract(contract)

    def fetch(ref: dict[str, str], kind: str) -> bytes:
        digest = ref["sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("invalid checkpoint object hash")
        if ref["key"] != f"{prefix}/{kind}/{digest}":
            raise ValueError("checkpoint object escapes its prefix")
        data = store.get(ref["key"]).data
        if sha(data) != digest:
            raise ValueError("checkpoint object hash mismatch")
        return data

    ref = json.loads(store.get(f"{prefix}/head.json").data)
    chain = []
    seen = set()
    while ref is not None:
        if ref["sha256"] in seen:
            raise ValueError("checkpoint commit cycle")
        seen.add(ref["sha256"])
        commit = json.loads(fetch(ref, "commits"))
        if commit["contract"] != contract:
            raise ValueError("checkpoint contract differs; resume refused")
        chain.append(commit)
        ref = commit["previous"]
    journal = b""
    for sequence, commit in enumerate(reversed(chain), 1):
        if commit["sequence"] != sequence:
            raise ValueError("checkpoint sequence is discontinuous")
        if commit["segment"]:
            journal += fetch(commit["segment"], "segments")
        if (
            sha(journal) != commit["journal_sha256"]
            or len(journal.splitlines()) != commit["row_count"]
        ):
            raise ValueError("checkpoint journal hash/count mismatch")
    latest = chain[0]
    if not {"plan.json", "run_manifest.json"} <= latest["files"].keys():
        raise ValueError("checkpoint lacks original plan/manifest")
    if not latest["files"].keys() <= {"plan.json", "run_manifest.json", "summary.json"}:
        raise ValueError("unexpected checkpoint filename")
    files = {name: fetch(value, "files") for name, value in latest["files"].items()}
    _rows(
        journal, json.loads(files["plan.json"]), json.loads(files["run_manifest.json"])
    )
    return journal, files, latest


def restore(
    store: Store, prefix: str, contract: dict[str, Any], destination: Path
) -> None:
    """Validate everything before creating a fresh run directory for --resume."""
    journal, files, commit = read_checkpoint(store, prefix, contract)
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (destination / name).write_bytes(data)
    (destination / "requests.jsonl").write_bytes(journal)
    write_json_atomic(
        destination / "checkpoint_index.json",
        {
            "row_count": commit["row_count"],
            "journal_sha256": sha(journal),
        },
    )
