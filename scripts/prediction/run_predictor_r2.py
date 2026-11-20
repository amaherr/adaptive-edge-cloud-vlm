"""Predictor V2 r2: fit, review, predict (sealed) and the single validation pass.

docs/predictor-v2/r2-specification.md fixes every rule used here. Steps:

  fit       training partition -> calibrations for the three models
  review    grouped 5-fold CV and leave-one-condition-out on training
  predict   validation inputs only -> predictions, hashed before any outcome
  validate  unseal the validation targets once, score, decide, freeze

``validate`` refuses to run if a decision already exists, and checks the
sealed targets against the hash the partition audit recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.prediction.r2.evaluation import (
    breakdown,
    decide,
    predict_units,
    score_model,
    training_review,
)
from adaptive_vlm.prediction.r2.models import FITTERS, MODELS
from adaptive_vlm.prediction.r2.units import (
    Unit,
    read_inputs,
    read_training,
    scored_units,
)

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    return _sha256(path)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _source() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty_paths": git("status", "--porcelain").splitlines(),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _units(path: Path) -> list[Unit]:
    return [Unit.from_json(json.loads(line)) for line in path.read_text().splitlines()]


def fit(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    audit = _read(args.partitions / "partition_audit.json")
    training = args.partitions / "training.jsonl"
    if _sha256(training) != audit["files"]["training.jsonl"]:
        raise ValueError("training partition hash differs from its audit")
    units = read_training(training)
    units_path = args.output / "training_units.jsonl"
    units_path.write_text(
        "".join(json.dumps(u.to_json(), sort_keys=True) + "\n" for u in units)
    )
    provenance = {
        "training_jsonl_sha256": audit["files"]["training.jsonl"],
        "training_units_sha256": _sha256(units_path),
        "partition_audit_sha256": _sha256(args.partitions / "partition_audit.json"),
        "fitted_at_utc": _now(),
        "source": _source(),
    }
    summary: dict[str, Any] = {"provenance": provenance, "calibrations": {}}
    for model in MODELS:
        calibration = FITTERS[model](units) | {"provenance": provenance}
        summary["calibrations"][model] = _write(
            args.output / f"calibration_{model}.json", calibration
        )
    summary["units"] = len(units)
    _write(args.output / "fit_summary.json", summary)
    print(json.dumps(summary["calibrations"], indent=1))


def review(args: argparse.Namespace) -> None:
    units = _units(args.fit / "training_units.jsonl")
    component = _read(args.fit / "calibration_r2_component.json")
    result = training_review(units, component) | {
        "training_units_sha256": _sha256(args.fit / "training_units.jsonl"),
        "reviewed_at_utc": _now(),
        "source": _source(),
    }
    _write(args.fit / "training_review.json", result)
    for model in MODELS:
        cv = result["grouped_cv"][model]["by_configuration"]
        print(
            model,
            {
                c.split("/")[-1]: (round(m["median_ape"], 1), round(m["p90_ape"], 1))
                for c, m in cv.items()
            },
        )
    print("candidate:", result["candidate"])


def predict(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite sealed predictions")
    audit = _read(args.partitions / "partition_audit.json")
    inputs = args.partitions / "validation_inputs.jsonl"
    if _sha256(inputs) != audit["files"]["validation_inputs.jsonl"]:
        raise ValueError("validation inputs hash differs from its audit")
    review_result = _read(args.fit / "training_review.json")
    units = read_inputs(inputs)
    args.output.mkdir(parents=True)
    records = []
    for model in MODELS:
        calibration = _read(args.fit / f"calibration_{model}.json")
        records += predict_units(calibration, units, model)
    path = args.output / "validation_predictions.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    manifest = {
        "predictions_sha256": _sha256(path),
        "predictions": len(records),
        "units": len(units),
        "candidate": review_result["candidate"],
        "calibrations": {
            m: _sha256(args.fit / f"calibration_{m}.json") for m in MODELS
        },
        "training_review_sha256": _sha256(args.fit / "training_review.json"),
        "validation_inputs_sha256": _sha256(inputs),
        "sealed_validation_targets_sha256": audit["files"]["validation_targets.jsonl"],
        "predicted_at_utc": _now(),
        "source": _source(),
    }
    _write(args.output / "prediction_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("predictions", "units", "candidate")}))


def validate(args: argparse.Namespace) -> None:
    decision_path = args.output / "decision.json"
    if decision_path.exists():
        raise FileExistsError("a G3 decision exists; the validation pass is single")
    manifest = _read(args.predictions / "prediction_manifest.json")
    predictions = args.predictions / "validation_predictions.jsonl"
    if _sha256(predictions) != manifest["predictions_sha256"]:
        raise ValueError("sealed predictions changed since they were hashed")
    for model, digest in manifest["calibrations"].items():
        if _sha256(args.fit / f"calibration_{model}.json") != digest:
            raise ValueError(f"calibration {model} changed since prediction")
    audit = _read(args.partitions / "partition_audit.json")
    targets = Path(audit["sealed_validation_path"])
    unsealed_at = _now()
    os.chmod(targets, stat.S_IRUSR)
    payload = targets.read_bytes()
    if (
        hashlib.sha256(payload).hexdigest()
        != manifest["sealed_validation_targets_sha256"]
    ):
        raise ValueError("sealed validation targets do not match their hash")
    units = scored_units(
        args.partitions / "validation_inputs.jsonl", payload.splitlines()
    )
    observed = {
        (u.inputs.request_id, u.inputs.cell_id, u.configuration): u for u in units
    }
    records: dict[str, list[dict[str, Any]]] = {m: [] for m in MODELS}
    for line in predictions.read_text().splitlines():
        record = json.loads(line)
        unit = observed[
            (record["request_id"], record["cell_id"], record["configuration"])
        ]
        record["observed"] = unit.targets.get("e2e") if unit.successes else None
        record["successes"], record["attempts"] = unit.successes, unit.attempts
        records[record["model"]].append(record)
    intervals = _read(args.fit / "training_review.json")["intervals_q95_ape"]
    scores = {
        m: score_model(records[m], records["constant_median"], intervals[m])
        for m in MODELS
    }
    decision = decide(manifest["candidate"], scores) | {
        "unsealed_at_utc": unsealed_at,
        "decided_at_utc": _now(),
        "prediction_manifest_sha256": _sha256(
            args.predictions / "prediction_manifest.json"
        ),
        "source": _source(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scores": scores,
        "breakdowns": {
            m: {
                f: breakdown(records[m], f)
                for f in ("condition_id", "load_state", "task")
            }
            for m in MODELS
        },
    }
    report_sha = _write(args.output / "validation_report.json", report)
    decision["validation_report_sha256"] = report_sha
    decision_sha = _write(decision_path, decision)
    if decision["selected"]:
        _write(
            args.output / "model_freeze_marker.json",
            {
                "model": decision["selected"],
                "calibration_sha256": manifest["calibrations"][decision["selected"]],
                "decision_sha256": decision_sha,
                "commit": decision["source"]["commit"],
                "frozen_at_utc": _now(),
            },
        )
    print(json.dumps({k: decision[k] for k in ("candidate", "selected", "g3")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    for name in ("fit", "review", "predict", "validate"):
        step = sub.add_parser(name)
        step.add_argument("--partitions", type=Path, required=name != "review")
        step.add_argument("--fit", type=Path, required=name != "fit")
        step.add_argument("--output", type=Path, required=name != "review")
        if name == "validate":
            step.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "fit":
        fit(args)
    elif args.step == "review":
        review(args)
    elif args.step == "predict":
        predict(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
