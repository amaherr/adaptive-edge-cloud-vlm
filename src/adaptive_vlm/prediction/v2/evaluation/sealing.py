"""Seal V2 validation forecasts before any outcome is read."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.prediction.v1.calibration import canonical_sha256
from adaptive_vlm.prediction.v1.predictor import PredictionTask
from adaptive_vlm.prediction.v2.dataset.contract import (
    BLOCKS,
    validate_task_features,
)
from adaptive_vlm.prediction.v2.dataset.parsing import digest, strict_json
from adaptive_vlm.prediction.v2.predictor import (
    VARIANTS,
    CalibrationV2,
    PredictorV2,
    state_from_snapshot,
)
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS
from adaptive_vlm.timing import perf_counter_ns


def trial_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return row["request_id"], row["block"], row["configuration_id"], row["repetition"]


def _models(fit_dir: Path) -> dict[str, CalibrationV2]:
    receipt = strict_json((fit_dir / "file_hashes.json").read_bytes())
    models = {}
    for variant in VARIANTS:
        path = fit_dir / f"calibration_{variant}.json"
        if digest(path) != receipt[path.name]:
            raise ValueError("fitted calibration file changed")
        models[variant] = CalibrationV2.load(path)
    return models


def _validation_inputs(bundle: Path) -> list[dict[str, Any]]:
    audit = strict_json((bundle / "partition_audit.json").read_bytes())
    path = bundle / "validation_inputs.jsonl"
    if digest(path) != audit["files"][path.name]:
        raise ValueError("validation feature snapshot changed")
    split_path = bundle / "split.json"
    if digest(split_path) != audit["split_sha256"]:
        raise ValueError("validation split identity changed")
    split = strict_json(split_path.read_bytes())
    rows = [strict_json(line) for line in path.read_bytes().splitlines()]
    expected = {
        (rid, b, c.id.value, r)
        for rid in split["validation_request_ids"]
        for b in BLOCKS["validation"]
        for c in CONFIGURATIONS
        for r in range(3)
    }
    if len(rows) != 360 or {trial_key(r) for r in rows} != expected:
        raise ValueError("validation inputs do not match exact frozen slots")
    for row in rows:
        validate_task_features(row["task"])
        if set(row) != {
            "schema_version",
            "request_id",
            "block",
            "configuration_id",
            "repetition",
            "trial_id",
            "task",
            "network",
        }:
            raise ValueError("validation input contains an unapproved/outcome key")
    return rows


def seal_validation_predictions(
    bundle: Path, fit_dir: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("choose a fresh validation output directory")
    models = _models(fit_dir)
    inputs = _validation_inputs(bundle)
    audit = strict_json((bundle / "partition_audit.json").read_bytes())
    records = []
    for variant, model in models.items():
        predictor = PredictorV2(model)
        for before in inputs:
            spec = next(
                s for s in CONFIGURATIONS if s.id.value == before["configuration_id"]
            )
            state = state_from_snapshot(before, model.policy)
            task = PredictionTask(**before["task"])
            started = perf_counter_ns()
            prediction = predictor.predict(spec, state, task)
            overhead_ms = (perf_counter_ns() - started) / 1_000_000
            records.append(
                {
                    "variant": variant,
                    "prediction_overhead_ms": overhead_ms,
                    "before": before,
                    "state": asdict(state),
                    "task": asdict(task),
                    "prediction": prediction.to_dict(),
                }
            )
    output.mkdir(parents=True)
    path = output / "predictions_validation.jsonl"
    path.write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records)
    )
    receipt = {
        "schema_version": 2,
        "prediction_rows": len(records),
        "prediction_sha256": digest(path),
        "input_snapshots_sha256": audit["files"]["validation_inputs.jsonl"],
        "partition_audit_sha256": digest(bundle / "partition_audit.json"),
        "calibration_file_sha256": {
            v: digest(fit_dir / f"calibration_{v}.json") for v in VARIANTS
        },
        "calibration_ids": {v: m.calibration_id for v, m in models.items()},
        "validation_outcomes_read": False,
        "final_outcomes_read": False,
    }
    receipt["seal_sha256"] = canonical_sha256(receipt)
    write_json_atomic(output / "predictions_seal.json", receipt)
    return receipt


def _open_validation_after_seal(
    bundle: Path, fit_dir: Path, output: Path
) -> list[dict[str, Any]]:
    receipt = strict_json((output / "predictions_seal.json").read_bytes())
    if (
        canonical_sha256({k: v for k, v in receipt.items() if k != "seal_sha256"})
        != receipt["seal_sha256"]
    ):
        raise ValueError("prediction seal content changed")
    if (
        digest(output / "predictions_validation.jsonl") != receipt["prediction_sha256"]
        or digest(bundle / "partition_audit.json") != receipt["partition_audit_sha256"]
    ):
        raise ValueError("predictions/source changed after sealing")
    models = _models(fit_dir)
    if any(
        digest(fit_dir / f"calibration_{v}.json")
        != receipt["calibration_file_sha256"][v]
        or models[v].calibration_id != receipt["calibration_ids"][v]
        for v in VARIANTS
    ):
        raise ValueError("model changed after predictions were sealed")
    audit = strict_json((bundle / "partition_audit.json").read_bytes())
    path = Path(audit["sealed_validation_path"])
    if path.stat().st_mode & 0o777:
        raise ValueError("validation target permission seal was not intact")
    try:
        path.chmod(0o400)
        data = path.read_bytes()
    finally:
        path.chmod(0o000)

    if hashlib.sha256(data).hexdigest() != audit["files"]["validation_targets.jsonl"]:
        raise ValueError("sealed validation targets changed")
    return [strict_json(line) for line in data.splitlines()]
