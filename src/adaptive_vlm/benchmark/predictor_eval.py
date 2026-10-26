"""Evaluate a frozen predictor against unseen requests and network blocks."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.calibrate import (
    fit_calibration,
    load_profile_and_split,
    percentile,
    task_and_state_from_row,
)
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS
from adaptive_vlm.scheduler.predictor import Calibration, Predictor


def _scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        row for row in rows if row["prediction"]["valid"] and row["status"] == "ok"
    ]
    errors = [
        abs(row["prediction"]["total_ms"] - row["observed_ms"]) for row in evaluated
    ]
    apes = [
        error / row["observed_ms"] for error, row in zip(errors, evaluated, strict=True)
    ]
    baseline_apes = [
        abs(row["baseline_ms"] - row["observed_ms"]) / row["observed_ms"]
        for row in evaluated
    ]
    signed = [row["prediction"]["total_ms"] - row["observed_ms"] for row in evaluated]
    return {
        "attempts": len(rows),
        "successful_predictions": len(evaluated),
        "failures": sum(row["status"] != "ok" for row in rows),
        "out_of_coverage": sum(
            row["status"] == "ok" and not row["prediction"]["valid"] for row in rows
        ),
        "median_ape": median(apes) if apes else None,
        "p90_ape": percentile(apes, 0.9) if apes else None,
        "signed_bias_ms_mean": mean(signed) if signed else None,
        "signed_bias_fraction_mean": mean(
            value / row["observed_ms"]
            for value, row in zip(signed, evaluated, strict=True)
        )
        if signed
        else None,
        "interval_coverage": mean(
            row["prediction"]["lower_ms"]
            <= row["observed_ms"]
            <= row["prediction"]["upper_ms"]
            for row in evaluated
        )
        if evaluated
        else None,
        "constant_median_ape": median(baseline_apes) if baseline_apes else None,
        "constant_p90_ape": percentile(baseline_apes, 0.9) if baseline_apes else None,
    }


def _rank_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["request_id"], row["network_bucket_id"], row["task_bucket"])
        ].append(row)
    oracle_matches = 0
    constant_matches = 0
    paired = 0
    for triple in grouped.values():
        if len(triple) != 3 or any(
            row["status"] != "ok" or not row["prediction"]["valid"] for row in triple
        ):
            continue
        observed = min(
            triple, key=lambda row: (row["observed_ms"], row["configuration_id"])
        )
        predicted = min(
            triple,
            key=lambda row: (row["prediction"]["total_ms"], row["configuration_id"]),
        )
        baseline = min(
            triple, key=lambda row: (row["baseline_ms"], row["configuration_id"])
        )
        paired += 1
        oracle_matches += predicted["configuration_id"] == observed["configuration_id"]
        constant_matches += baseline["configuration_id"] == observed["configuration_id"]
    return {
        "paired_conditions": paired,
        "prediction_accuracy": oracle_matches / paired if paired else None,
        "constant_median_accuracy": constant_matches / paired if paired else None,
    }


def evaluate_calibration(
    calibration: Calibration, rows: list[dict[str, Any]], split: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictor = Predictor(calibration)
    request_holdout = set(split["holdout_request_ids"])
    network_holdout = set(split["holdout_network_bucket_ids"])
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        if row["warmup"]:
            continue
        is_request_holdout = row["request_id"] in request_holdout
        is_network_holdout = row["network_state"]["bucket_id"] in network_holdout
        if not is_request_holdout and not is_network_holdout:
            continue
        section = (
            "combined_holdout"
            if is_request_holdout and is_network_holdout
            else "request_holdout"
            if is_request_holdout
            else "network_holdout"
        )
        spec = next(
            (
                candidate
                for candidate in CONFIGURATIONS
                if candidate.id.value == row["configuration_id"]
            ),
            None,
        )
        if spec is None:
            raise ValueError(
                f"unknown profile configuration: {row['configuration_id']}"
            )
        fit = calibration.candidates[spec.id.value]
        if row["status"] == "ok":
            task, state = task_and_state_from_row(row)
            prediction = predictor.predict(spec, state, task)
            task_snapshot = task.__dict__.copy()
            state_snapshot = state.__dict__.copy()
        else:
            prediction = predictor._invalid(spec.id.value, "failed_trial")
            task_snapshot = None
            state_snapshot = None
        evaluated.append(
            {
                "schema_version": 1,
                "section": section,
                "request_id": row["request_id"],
                "trial_id": row["trial_id"],
                "configuration_id": spec.id.value,
                "network_bucket_id": row["network_state"]["bucket_id"],
                "task_bucket": row["task_bucket"],
                "status": row["status"],
                "task_state_before_output": task_snapshot,
                "system_state": state_snapshot,
                "prediction": prediction.to_dict(),
                "baseline_ms": fit.constant_median_ms,
                "observed_ms": row["observed_end_to_end_ms"],
            }
        )
    sections = ("request_holdout", "network_holdout", "combined_holdout")
    report: dict[str, Any] = {
        "schema_version": 1,
        "calibration_id": calibration.calibration_id,
        "source_profile_sha256": calibration.source_profile_sha256,
        "request_holdout_ids": list(calibration.holdout_request_ids),
        "network_holdout_bucket_ids": list(calibration.holdout_network_bucket_ids),
        "sections": {},
    }
    for section in sections:
        section_rows = [row for row in evaluated if row["section"] == section]
        by_candidate = {
            spec.id.value: _scores(
                [
                    row
                    for row in section_rows
                    if row["configuration_id"] == spec.id.value
                ]
            )
            for spec in CONFIGURATIONS
        }
        by_bucket: dict[str, dict[str, Any]] = {}
        for bucket in {row["network_bucket_id"] for row in section_rows}:
            for task_bucket in {row["task_bucket"] for row in section_rows}:
                selected = [
                    row
                    for row in section_rows
                    if row["network_bucket_id"] == bucket
                    and row["task_bucket"] == task_bucket
                ]
                if selected:
                    by_bucket[f"{task_bucket}|{bucket}"] = {
                        name: _scores(
                            [row for row in selected if row["configuration_id"] == name]
                        )
                        for name in by_candidate
                    }
        report["sections"][section] = {
            "rows": len(section_rows),
            "by_candidate": by_candidate,
            "by_task_network_bucket": by_bucket,
            "ranking": _rank_accuracy(section_rows),
        }
    request = report["sections"]["request_holdout"]
    ranking = request["ranking"]
    candidates = request["by_candidate"]
    diagnostic_targets = (
        ranking["paired_conditions"] > 0
        and ranking["prediction_accuracy"] >= ranking["constant_median_accuracy"]
        and all(
            scores["median_ape"] is not None
            and scores["median_ape"] <= split["target_median_ape"]
            and scores["p90_ape"] <= split["target_p90_ape"]
            for scores in candidates.values()
        )
    )
    report["local_request_holdout_targets_met"] = diagnostic_targets
    report["full_g3_gate_evaluable"] = bool(network_holdout) and (
        report["sections"]["network_holdout"]["ranking"]["paired_conditions"] > 0
    )
    report["full_g3_gate_passed"] = (
        bool(diagnostic_targets)
        and report["full_g3_gate_evaluable"]
        and report["sections"]["network_holdout"]["ranking"]["prediction_accuracy"]
        >= report["sections"]["network_holdout"]["ranking"]["constant_median_accuracy"]
        and all(
            scores["median_ape"] is not None
            and scores["median_ape"] <= split["target_median_ape"]
            and scores["p90_ape"] <= split["target_p90_ape"]
            for scores in report["sections"]["network_holdout"]["by_candidate"].values()
        )
    )
    return report, evaluated


def run_predictor_evaluation(
    *, profile_path: Path, split_path: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"predictor output already exists: {output_dir}")
    rows, split, profile_hash, split_hash = load_profile_and_split(
        profile_path, split_path
    )
    calibration = fit_calibration(
        rows, split, profile_sha256=profile_hash, split_sha256=split_hash
    )
    report, evaluated = evaluate_calibration(calibration, rows, split)
    output_dir.mkdir(parents=True)
    write_json_atomic(output_dir / "calibration.json", calibration.to_dict())
    write_json_atomic(output_dir / "evaluation.json", report)
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evaluated),
        encoding="utf-8",
    )
    return report
