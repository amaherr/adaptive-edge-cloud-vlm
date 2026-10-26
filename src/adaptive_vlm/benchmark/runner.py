"""Scheduled, resumable execution with explicit failure and quality records."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from adaptive_vlm.artifacts import append_jsonl, write_json_atomic
from adaptive_vlm.config import (
    ExecutionMode,
    ExperimentConfig,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import InferenceStatus, RunManifest
from adaptive_vlm.data.manifest import FrozenManifest, RequestItem
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.remote import RemoteInferenceError
from adaptive_vlm.transport.protocol import ProtocolError


@dataclass(frozen=True)
class Trial:
    trial_id: str
    request_id: str
    warmup: bool
    repetition: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "request_id": self.request_id,
            "warmup": self.warmup,
            "repetition": self.repetition,
        }


def make_schedule(
    manifest: FrozenManifest, *, warmups: int, repetitions: int, seed: int
) -> tuple[Trial, ...]:
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions positive")
    ids = [item.request_id for item in manifest.items]
    rng = random.Random(seed)
    schedule = [
        Trial(f"w{index:06d}", ids[index % len(ids)], True, -1)
        for index in range(warmups)
    ]
    for repetition in range(repetitions):
        ordered = ids.copy()
        rng.shuffle(ordered)
        schedule.extend(
            Trial(f"r{repetition:04d}-{position:06d}", request_id, False, repetition)
            for position, request_id in enumerate(ordered)
        )
    return tuple(schedule)


def _quality(item: RequestItem, answer: str) -> float | None:
    if item.scoring == "none":
        return None
    normalized = answer.strip().casefold()
    if item.scoring == "exact_match":
        return float(
            any(normalized == ref.strip().casefold() for ref in item.references)
        )
    # This is a provisional consensus score, not the official VQA normalization.
    references = [ref.strip().casefold() for ref in item.references]
    return sum(
        min(1.0, (references.count(normalized) - int(ref == normalized)) / 3)
        for ref in references
    ) / len(references)


def _status(exc: Exception) -> InferenceStatus:
    if isinstance(exc, TimeoutError):
        return InferenceStatus.TIMEOUT
    if isinstance(exc, MemoryError) or "OutOfMemory" in type(exc).__name__:
        return InferenceStatus.OOM
    if isinstance(exc, ProtocolError):
        return InferenceStatus.PROTOCOL_ERROR
    if isinstance(exc, RemoteInferenceError):
        if exc.code in {"timeout", "deadline_exceeded"}:
            return InferenceStatus.TIMEOUT
        if exc.code in {"oom", "out_of_memory"}:
            return InferenceStatus.OOM
        return InferenceStatus.PROTOCOL_ERROR
    return InferenceStatus.MODEL_ERROR


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid raw result line {number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"raw result line {number} is not an object")
        rows.append(value)
    return rows


def semantic_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash experimental settings, excluding machine-specific file locations."""

    semantic = dict(config)
    semantic.pop("source_path", None)
    semantic.pop("artifacts_dir", None)
    dataset = dict(semantic["dataset"])
    dataset.pop("manifest_path", None)  # Content is checked separately by SHA-256.
    semantic["dataset"] = dataset
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_benchmark(
    *,
    config: ExperimentConfig,
    manifest: FrozenManifest,
    policy: ExecutionPolicy,
    run_manifest: RunManifest,
    run_dir: Path,
    warmups: int,
    repetitions: int,
    pipeline: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute one policy and persist each attempt before advancing."""

    if manifest.revision != config.dataset.revision:
        raise ValueError("request manifest revision differs from experiment config")
    if policy.mode not in ExecutionMode:
        raise ValueError("unsupported execution mode")
    schedule = make_schedule(
        manifest, warmups=warmups, repetitions=repetitions, seed=config.runtime.seed
    )
    plan = {
        "schema_version": 1,
        "config": config.to_dict(),
        "config_sha256": semantic_config_sha256(config.to_dict()),
        "model_contract_sha256": model_contract_fingerprint(config.model),
        "dataset_manifest_sha256": manifest.sha256,
        "mode": policy.mode.value,
        "pipeline": pipeline,
        "schedule": [trial.to_dict() for trial in schedule],
    }
    plan_hash = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan["plan_sha256"] = plan_hash
    plan_path = run_dir / "plan.json"
    raw_path = run_dir / "requests.jsonl"
    if resume:
        if not plan_path.is_file():
            raise ValueError("cannot resume: plan.json does not exist")
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("cannot resume: plan, config, or dataset differs")
        saved_run_manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        if (
            saved_run_manifest.get("dataset_manifest_sha256") != manifest.sha256
            or saved_run_manifest.get("config") != config.to_dict()
        ):
            raise ValueError("cannot resume: run manifest differs")
        run_id = str(saved_run_manifest["run_id"])
    else:
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        write_json_atomic(plan_path, plan)
        write_json_atomic(run_dir / "run_manifest.json", run_manifest.to_dict())
        run_id = run_manifest.run_id

    rows = _read_rows(raw_path)
    expected = {trial.trial_id: trial for trial in schedule}
    completed: set[str] = set()
    for existing_row in rows:
        trial_id = existing_row.get("trial_id")
        if trial_id not in expected or trial_id in completed:
            raise ValueError(f"unknown or duplicate trial in raw results: {trial_id}")
        trial = expected[trial_id]
        if (
            existing_row.get("request_id") != trial.request_id
            or existing_row.get("warmup") != trial.warmup
            or existing_row.get("repetition") != trial.repetition
            or existing_row.get("plan_sha256") != plan_hash
            or existing_row.get("run_id") != run_id
            or existing_row.get("config_sha256") != plan["config_sha256"]
            or existing_row.get("model_contract_sha256")
            != plan["model_contract_sha256"]
            or existing_row.get("dataset_manifest_sha256") != manifest.sha256
        ):
            raise ValueError(f"raw result does not match plan: {trial_id}")
        completed.add(trial_id)

    items = {item.request_id: item for item in manifest.items}
    for position, trial in enumerate(schedule, 1):
        if trial.trial_id in completed:
            continue
        item = items[trial.request_id]
        print(
            f"[{position}/{len(schedule)}] {policy.mode.value} "
            f"{trial.trial_id} {trial.request_id}",
            flush=True,
        )
        started = perf_counter_ns()
        row: dict[str, Any] = {
            **trial.to_dict(),
            "started_at_utc": datetime.now(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "plan_sha256": plan_hash,
            "config_sha256": plan["config_sha256"],
            "model_contract_sha256": plan["model_contract_sha256"],
            "dataset_manifest_sha256": manifest.sha256,
            "run_id": run_id,
            "mode": policy.mode.value,
            "image_sha256": item.image_sha256,
            "scoring": item.scoring,
        }
        try:
            result = policy.infer(item.to_request())
            if result.request_id != item.request_id:
                raise ValueError("policy returned a different request_id")
            row.update(
                status=InferenceStatus.OK.value,
                output_text=result.output_text,
                output_token_ids=list(result.output_token_ids),
                metrics=dict(result.metrics),
                quality_score=_quality(item, result.output_text),
            )
        except Exception as exc:
            row.update(
                status=_status(exc).value,
                output_text=None,
                output_token_ids=[],
                metrics={},
                quality_score=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        row["observed_end_to_end_ms"] = (perf_counter_ns() - started) / 1_000_000
        append_jsonl(raw_path, row)
        rows.append(row)
    summary = summarize(rows, manifest, policy.mode, plan_hash)
    write_json_atomic(run_dir / "summary.json", summary)
    return summary


def summarize(
    rows: Sequence[Mapping[str, Any]],
    manifest: FrozenManifest,
    mode: ExecutionMode,
    plan_hash: str,
) -> dict[str, Any]:
    """Summarize measured attempts only, retaining failures in the denominator."""

    measured = [row for row in rows if not row["warmup"]]
    successful = [row for row in measured if row["status"] == "ok"]
    latency = sorted(float(row["observed_end_to_end_ms"]) for row in successful)
    quality = [
        float(row["quality_score"])
        for row in successful
        if row["quality_score"] is not None
    ]
    wire_bytes: list[int] = []
    for row in successful:
        metrics = row["metrics"]
        if "total_wire_bytes" in metrics:
            wire_bytes.append(int(metrics["total_wire_bytes"]))
        elif "request_wire_bytes" in metrics and "response_wire_bytes" in metrics:
            wire_bytes.append(
                int(metrics["request_wire_bytes"]) + int(metrics["response_wire_bytes"])
            )
    status_counts = dict(Counter(str(row["status"]) for row in measured))
    return {
        "schema_version": 1,
        "mode": mode.value,
        "source": manifest.source,
        "revision": manifest.revision,
        "dataset_manifest_sha256": manifest.sha256,
        "plan_sha256": plan_hash,
        "warmup_attempts": len(rows) - len(measured),
        "measured_attempts": len(measured),
        "status_counts": status_counts,
        "failure_rate": (
            (len(measured) - len(successful)) / len(measured) if measured else None
        ),
        "latency_ms_mean_success": sum(latency) / len(latency) if latency else None,
        "latency_ms_median_success": (
            (latency[(len(latency) - 1) // 2] + latency[len(latency) // 2]) / 2
            if latency
            else None
        ),
        "latency_ms_p95_success": _percentile(latency, 0.95) if latency else None,
        "latency_ms_iqr_success": (
            _percentile(latency, 0.75) - _percentile(latency, 0.25) if latency else None
        ),
        "wire_bytes_mean_success": (
            sum(wire_bytes) / len(wire_bytes) if wire_bytes else None
        ),
        "quality_mean_success": sum(quality) / len(quality) if quality else None,
        "quality_scored_attempts": len(quality),
        "quality_metric": "fixture_exact_match_or_provisional_vqa_consensus",
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires observations")
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
