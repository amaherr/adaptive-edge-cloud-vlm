"""Strict, reproducible paired analysis of an adaptive scheduler campaign."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic, write_jsonl_atomic
from adaptive_vlm.benchmark.adaptive_analysis import analyze_adaptive_run
from adaptive_vlm.benchmark.runner import semantic_config_sha256

MODES = ("edge", "cloud", "distributed")
CONFIGURATION_MODES = {
    "edge/full": "edge",
    "cloud/raw_image": "cloud",
    "distributed/projected/synchronous": "distributed",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("trial_id"), str):
            raise ValueError(f"invalid trial row at {path}:{number}")
        trial_id = value["trial_id"]
        if trial_id in result:
            raise ValueError(f"duplicate trial {trial_id}: {path}")
        result[trial_id] = value
    return result


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _wire_bytes(row: dict[str, Any]) -> int:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("invalid request metrics")
    if "total_wire_bytes" in metrics:
        total = metrics["total_wire_bytes"]
    else:
        total = metrics.get("request_wire_bytes", 0) + metrics.get(
            "response_wire_bytes", 0
        )
    if not isinstance(total, int) or total < 0:
        raise ValueError("invalid wire byte count")
    return total


def _summary(rows: list[dict[str, Any]], sla_ms: float) -> dict[str, Any]:
    success = [row for row in rows if row["status"] == "ok"]
    latencies = [float(row["observed_end_to_end_ms"]) for row in success]
    wire = [_wire_bytes(row) for row in success]
    sla_success = sum(
        row["status"] == "ok" and float(row["observed_end_to_end_ms"]) <= sla_ms
        for row in rows
    )
    return {
        "attempts": len(rows),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "sla_successes": sla_success,
        "sla_success_rate": sla_success / len(rows) if rows else None,
        "latency_ms_median_success": median(latencies) if latencies else None,
        "latency_ms_p95_success": _percentile(latencies, 0.95),
        "wire_bytes_mean_success": mean(wire) if wire else None,
        "wire_bytes_mean_per_attempt": sum(wire) / len(rows) if rows else None,
    }


def _static_choice(
    rows: dict[str, dict[str, dict[str, Any]]],
    trials: list[str],
    sla_ms: float,
) -> str:
    def key(mode: str) -> tuple[float, float, int]:
        summary = _summary([rows[mode][trial] for trial in trials], sla_ms)
        latency = summary["latency_ms_median_success"]
        return (
            -float(summary["sla_success_rate"]),
            float(latency) if latency is not None else math.inf,
            MODES.index(mode),
        )

    return min(MODES, key=key)


def _paired_interval(
    paired: list[dict[str, Any]],
    field: str,
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, Any]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in paired:
        value = row.get(field)
        if value is not None:
            clusters[row["request_id"]].append(float(value))
    ids = sorted(clusters)
    values = [value for group in clusters.values() for value in group]
    if not values:
        return {"paired_trials": 0, "request_clusters": 0, "mean": None, "ci": None}
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        selected = [rng.choice(ids) for _ in ids]
        replicate = [value for request_id in selected for value in clusters[request_id]]
        samples.append(mean(replicate))
    alpha = (1 - confidence) / 2
    return {
        "paired_trials": len(values),
        "request_clusters": len(ids),
        "mean": mean(values),
        "ci": [_percentile(samples, alpha), _percentile(samples, 1 - alpha)],
    }


def analyze_thesis_campaign(
    specification_path: Path,
    adaptive_dir: Path,
    fixed_dirs: list[Path],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Report matched controls; refuse incomplete provenance and split leakage."""

    spec = _object(specification_path)
    required = {
        "schema_version",
        "campaign_id",
        "design",
        "dataset_manifest_sha256",
        "model_contract_sha256",
        "calibration_id",
        "predictor_training_request_ids",
        "validation_request_ids",
        "evaluation_request_ids",
        "validation_network_block_ids",
        "evaluation_network_block_ids",
        "sla_ms",
        "bootstrap_seed",
        "bootstrap_resamples",
        "confidence_level",
        "task_type",
        "network_regime",
    }
    if set(spec) != required or spec["schema_version"] != 1:
        raise ValueError("unsupported thesis evaluation specification")
    if spec["design"] not in {"retrospective_diagnostic", "prospective"}:
        raise ValueError("invalid evaluation design")
    validation = set(spec["validation_request_ids"])
    evaluation = set(spec["evaluation_request_ids"])
    training = set(spec["predictor_training_request_ids"])
    if (
        not validation
        or not evaluation
        or validation & evaluation
        or training & evaluation
        or set(spec["validation_network_block_ids"])
        & set(spec["evaluation_network_block_ids"])
    ):
        raise ValueError("training/validation/evaluation split is empty or leaks")
    sla = float(spec["sla_ms"])
    confidence = float(spec["confidence_level"])
    resamples = spec["bootstrap_resamples"]
    seed = spec["bootstrap_seed"]
    if (
        not math.isfinite(sla)
        or sla <= 0
        or not 0 < confidence < 1
        or not isinstance(resamples, int)
        or resamples < 100
        or not isinstance(seed, int)
    ):
        raise ValueError("invalid SLA or bootstrap specification")
    audit = analyze_adaptive_run(adaptive_dir, fixed_dirs)
    if any(
        audit[key]
        for key in (
            "decision_trace_errors",
            "wire_accounting_errors",
            "placement_mismatch_trials",
            "same_placement_token_mismatches",
        )
    ):
        raise ValueError("adaptive run failed provenance or equivalence audit")
    plan = _object(adaptive_dir / "plan.json")
    if (
        plan["dataset_manifest_sha256"] != spec["dataset_manifest_sha256"]
        or plan["model_contract_sha256"] != spec["model_contract_sha256"]
        or plan["calibration_id"] != spec["calibration_id"]
    ):
        raise ValueError("specification does not match adaptive campaign")
    expected = {row["trial_id"]: row for row in plan["schedule"]}
    if len(expected) != len(plan["schedule"]):
        raise ValueError("duplicate trial in schedule")
    measured = {
        trial_id: trial for trial_id, trial in expected.items() if not trial["warmup"]
    }
    scheduled_ids = {trial["request_id"] for trial in measured.values()}
    if validation | evaluation != scheduled_ids:
        raise ValueError("specification does not partition measured request IDs")
    if not measured:
        raise ValueError("no measured trials")
    adaptive_rows = _rows(adaptive_dir / "requests.jsonl")
    rows: dict[str, dict[str, dict[str, Any]]] = {"adaptive": adaptive_rows}
    manifests = {"adaptive": _object(adaptive_dir / "run_manifest.json")}
    paths: dict[str, Path] = {"adaptive": adaptive_dir}
    for directory in fixed_dirs:
        fixed_plan = _object(directory / "plan.json")
        mode = fixed_plan["mode"]
        if mode not in MODES or mode in rows:
            raise ValueError("exactly one run per fixed policy is required")
        if (
            fixed_plan["schedule"] != plan["schedule"]
            or fixed_plan["dataset_manifest_sha256"] != spec["dataset_manifest_sha256"]
            or fixed_plan["model_contract_sha256"] != spec["model_contract_sha256"]
            or semantic_config_sha256(fixed_plan["config"])
            != semantic_config_sha256(plan["config"])
        ):
            raise ValueError(f"{mode} plan does not match adaptive campaign")
        rows[mode] = _rows(directory / "requests.jsonl")
        manifests[mode] = _object(directory / "run_manifest.json")
        paths[mode] = directory
    if set(rows) != {"adaptive", *MODES}:
        raise ValueError("all three fixed controls are required")
    for mode, run_rows in rows.items():
        run_plan = _object(paths[mode] / "plan.json")
        if set(run_rows) != set(expected):
            raise ValueError(f"{mode} has missing or unexpected trial rows")
        if (
            manifests[mode]["dataset_manifest_sha256"]
            != spec["dataset_manifest_sha256"]
        ):
            raise ValueError(f"{mode} run manifest has wrong dataset hash")
        for trial_id, row in run_rows.items():
            trial = expected[trial_id]
            if any(
                row.get(key) != value
                for key, value in (
                    ("trial_id", trial_id),
                    ("request_id", trial["request_id"]),
                    ("warmup", trial["warmup"]),
                    ("repetition", trial["repetition"]),
                    ("plan_sha256", run_plan["plan_sha256"]),
                    ("run_id", manifests[mode]["run_id"]),
                    ("dataset_manifest_sha256", spec["dataset_manifest_sha256"]),
                    ("model_contract_sha256", spec["model_contract_sha256"]),
                )
            ):
                raise ValueError(f"{mode} trial provenance mismatch: {trial_id}")
            latency = row.get("observed_end_to_end_ms")
            if (
                not isinstance(latency, (int, float))
                or not math.isfinite(latency)
                or latency < 0
            ):
                raise ValueError(f"{mode} trial has invalid latency: {trial_id}")
    validation_trials = sorted(
        trial_id
        for trial_id, trial in measured.items()
        if trial["request_id"] in validation
    )
    evaluation_trials = sorted(
        trial_id
        for trial_id, trial in measured.items()
        if trial["request_id"] in evaluation
    )
    best_static = _static_choice(rows, validation_trials, sla)
    trial_table: list[dict[str, Any]] = []
    for trial_id in evaluation_trials:
        trial = measured[trial_id]
        adaptive = rows["adaptive"][trial_id]
        fixed_success = {
            mode: float(rows[mode][trial_id]["observed_end_to_end_ms"])
            for mode in MODES
            if rows[mode][trial_id]["status"] == "ok"
        }
        oracle_mode = (
            min(fixed_success, key=lambda mode: fixed_success[mode])
            if fixed_success
            else None
        )
        record: dict[str, Any] = {
            "trial_id": trial_id,
            "request_id": trial["request_id"],
            "repetition": trial["repetition"],
            "selected_configuration_id": adaptive.get("selected_configuration_id"),
            "network_block_id": adaptive.get("system_state", {}).get(
                "network_bucket_id"
            ),
            "oracle_mode": oracle_mode,
            "oracle_latency_ms": fixed_success.get(oracle_mode)
            if oracle_mode
            else None,
        }
        for mode in ("adaptive", *MODES):
            row = rows[mode][trial_id]
            record[f"{mode}_status"] = row["status"]
            record[f"{mode}_latency_ms"] = row["observed_end_to_end_ms"]
            record[f"{mode}_wire_bytes"] = (
                _wire_bytes(row) if row["status"] == "ok" else None
            )
            record[f"{mode}_sla_success"] = int(
                row["status"] == "ok" and row["observed_end_to_end_ms"] <= sla
            )
            if mode != "adaptive":
                record[f"adaptive_minus_{mode}_sla_success"] = (
                    record["adaptive_sla_success"] - record[f"{mode}_sla_success"]
                )
                record[f"adaptive_minus_{mode}_latency_ms"] = (
                    float(adaptive["observed_end_to_end_ms"])
                    - float(row["observed_end_to_end_ms"])
                    if adaptive["status"] == "ok" and row["status"] == "ok"
                    else None
                )
        record["adaptive_minus_oracle_latency_ms"] = (
            float(adaptive["observed_end_to_end_ms"]) - min(fixed_success.values())
            if adaptive["status"] == "ok" and fixed_success
            else None
        )
        attempts = adaptive.get("attempts", [])
        predictions = (
            attempts[-1].get("decision", {}).get("candidates", []) if attempts else []
        )
        for candidate in predictions:
            mode = CONFIGURATION_MODES.get(candidate.get("configuration_id"))
            prediction = candidate.get("prediction") or {}
            if mode is None or not prediction.get("valid"):
                continue
            predicted = prediction.get("total_ms")
            if not isinstance(predicted, (int, float)) or not math.isfinite(predicted):
                raise ValueError(f"invalid prediction in trial {trial_id}")
            record[f"{mode}_predicted_latency_ms"] = predicted
            if mode in fixed_success and fixed_success[mode] > 0:
                record[f"{mode}_prediction_ape"] = (
                    abs(predicted - fixed_success[mode]) / fixed_success[mode]
                )
        trial_table.append(record)
    prediction_error: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        errors = [
            float(row[f"{mode}_prediction_ape"])
            for row in trial_table
            if f"{mode}_prediction_ape" in row
        ]
        prediction_error[mode] = {
            "paired_trials": len(errors),
            "median_ape": median(errors) if errors else None,
            "p90_ape": _percentile(errors, 0.9),
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": spec["campaign_id"],
        "design": spec["design"],
        "interpretation": "descriptive_only"
        if spec["design"] == "retrospective_diagnostic"
        else "prospective",
        "task_type": spec["task_type"],
        "network_regime": spec["network_regime"],
        "specification_sha256": _sha256(specification_path),
        "input_sha256": {
            mode: {
                name: _sha256(directory / name)
                for name in ("plan.json", "run_manifest.json", "requests.jsonl")
            }
            for mode, directory in paths.items()
        },
        "dataset_manifest_sha256": spec["dataset_manifest_sha256"],
        "model_contract_sha256": spec["model_contract_sha256"],
        "calibration_id": spec["calibration_id"],
        "sla_ms": sla,
        "validation_request_count": len(validation),
        "evaluation_request_count": len(evaluation),
        "evaluation_trials": len(evaluation_trials),
        "best_static_validation_choice": best_static,
        "validation_fixed": {
            mode: _summary([rows[mode][trial] for trial in validation_trials], sla)
            for mode in MODES
        },
        "evaluation": {
            mode: _summary([rows[mode][trial] for trial in evaluation_trials], sla)
            for mode in ("adaptive", *MODES)
        },
        "selection_counts": dict(
            Counter(
                rows["adaptive"][trial].get("selected_configuration_id")
                for trial in evaluation_trials
            )
        ),
        "selection_overhead_ms_median": median(
            float(rows["adaptive"][trial].get("selection_overhead_ms", 0))
            for trial in evaluation_trials
        ),
        "monitor_overhead_ms_median": median(
            float(rows["adaptive"][trial].get("monitor_overhead_ms", 0))
            for trial in evaluation_trials
        ),
        "oracle_coverage": sum(row["oracle_mode"] is not None for row in trial_table),
        "prediction_error_against_fixed_controls": prediction_error,
        "bootstrap": {
            "seed": seed,
            "resamples": resamples,
            "confidence_level": confidence,
            "cluster": "request_id",
        },
        "adaptive_audit": audit,
        "official_vqa_accuracy": None,
        "edge_memory_peak_mib": None,
        "g7_passed": False,
    }
    report["paired"] = {
        mode: {
            "latency_ms": _paired_interval(
                trial_table,
                f"adaptive_minus_{mode}_latency_ms",
                seed=seed,
                resamples=resamples,
                confidence=confidence,
            ),
            "sla_success": _paired_interval(
                trial_table,
                f"adaptive_minus_{mode}_sla_success",
                seed=seed,
                resamples=resamples,
                confidence=confidence,
            ),
        }
        for mode in MODES
    }
    report["paired"]["best_static"] = report["paired"][best_static]
    report["measured_oracle_regret_ms"] = _paired_interval(
        trial_table,
        "adaptive_minus_oracle_latency_ms",
        seed=seed,
        resamples=resamples,
        confidence=confidence,
    )
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    write_jsonl_atomic(output_dir / "paired_trials.jsonl", trial_table)
    report["paired_trials_sha256"] = _sha256(output_dir / "paired_trials.jsonl")
    write_json_atomic(output_dir / "report.json", report)
    return report
