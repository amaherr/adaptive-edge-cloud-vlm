"""Seal pre-execution validation predictions before permitting outcome access."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.calibrate import canonical_sha256, percentile
from adaptive_vlm.benchmark.predictor_v2_data import (
    BLOCKS,
    digest,
    strict_json,
    validate_target,
    validate_task_features,
)
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS
from adaptive_vlm.scheduler.predictor import PredictionTask
from adaptive_vlm.scheduler.predictor_v2 import (
    VARIANTS,
    CalibrationV2,
    PredictorV2,
    state_from_snapshot,
)
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


def scores(rows: list[dict[str, Any]], *, diagnostic: bool = False) -> dict[str, Any]:
    usable = [
        r
        for r in rows
        if r["status"] == "ok"
        and r["prediction"]["total_ms"] is not None
        and (diagnostic or r["prediction"]["valid"])
    ]
    errors = [r["prediction"]["total_ms"] - r["observed_ms"] for r in usable]
    apes = [
        abs(error) / r["observed_ms"] for error, r in zip(errors, usable, strict=True)
    ]
    return {
        "scheduled": len(rows),
        "successful_attempts": sum(r["status"] == "ok" for r in rows),
        "failed_attempts": sum(r["status"] != "ok" for r in rows),
        "valid_predictions": sum(r["prediction"]["valid"] for r in rows),
        "prediction_coverage": sum(r["prediction"]["valid"] for r in rows) / len(rows)
        if rows
        else 0.0,
        "scored": len(usable),
        "median_ape": median(apes) if apes else None,
        "p90_ape": percentile(apes, 0.9) if apes else None,
        "mae_ms": mean(abs(e) for e in errors) if errors else None,
        "signed_bias_ms": mean(errors) if errors else None,
        "signed_bias_fraction": mean(
            e / r["observed_ms"] for e, r in zip(errors, usable, strict=True)
        )
        if errors
        else None,
        "interval_coverage": mean(
            r["prediction"]["lower_ms"]
            <= r["observed_ms"]
            <= r["prediction"]["upper_ms"]
            for r in usable
        )
        if usable
        else None,
        "invalid_reasons": dict(
            Counter(
                r["prediction"]["reason"] for r in rows if not r["prediction"]["valid"]
            )
        ),
        "coverage_labels": dict(Counter(r["prediction"]["coverage"] for r in rows)),
        "live_selection_eligible": sum(
            r["prediction"]["selection_eligible"] for r in rows
        ),
    }


def _request_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        b = row["before"]
        groups[b["request_id"], b["block"], b["configuration_id"]].append(row)
    aggregated = []
    for group in groups.values():
        success = all(r["status"] == "ok" for r in group) and len(group) == 3
        predictions = [r["prediction"] for r in group]
        nonnull = all(p["total_ms"] is not None for p in predictions)
        p = {**predictions[0], "valid": all(p["valid"] for p in predictions)}
        for field in ("total_ms", "lower_ms", "upper_ms"):
            p[field] = median(x[field] for x in predictions) if nonnull else None
        aggregated.append(
            {
                **group[0],
                "status": "ok" if success else "incomplete_successful_repetitions",
                "observed_ms": median(r["observed_ms"] for r in group)
                if success
                else None,
                "prediction": p,
            }
        )
    return aggregated


def ranking(
    rows: list[dict[str, Any]], models: dict[str, CalibrationV2], variant: str
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in _request_rows(rows):
        grouped[r["before"]["request_id"], r["before"]["block"]].append(r)
    hits = ranked = unsafe = 0
    choices: Counter[str] = Counter()
    conditions = []
    for group in grouped.values():
        feasible = []
        for r in group:
            cid = r["before"]["configuration_id"]
            fit = models[variant].candidates[cid]
            policy = models[variant].policy
            edge_ok = cid == "cloud/raw_image" or (
                fit["edge_memory_requirement_mib"] is not None
                and fit["edge_memory_requirement_mib"]
                <= policy["edge_dedicated_memory_mib"]
            )
            cloud_ok = cid == "edge/full" or (
                fit["cloud_memory_requirement_mib"] is not None
                and fit["cloud_memory_requirement_mib"]
                <= policy["cloud_dedicated_memory_mib"]
            )
            okay = edge_ok and cloud_ok
            if r["prediction"]["execution_feasible"] and not okay:
                unsafe += 1
            if okay:
                feasible.append(r)
        # A missing prediction for an executable counterfactual is not dropped.
        condition = {
            "request_id": group[0]["before"]["request_id"],
            "block": group[0]["before"]["block"],
            "matched": False,
            "ranked": False,
        }
        if not feasible or any(
            r["status"] != "ok" or not r["prediction"]["valid"] for r in feasible
        ):
            conditions.append(condition)
            continue
        observed = min(
            feasible, key=lambda r: (r["observed_ms"], r["before"]["configuration_id"])
        )
        predicted = min(
            feasible,
            key=lambda r: (
                r["prediction"]["total_ms"],
                r["before"]["configuration_id"],
            ),
        )
        ranked += 1
        hits += (
            observed["before"]["configuration_id"]
            == predicted["before"]["configuration_id"]
        )
        choices[predicted["before"]["configuration_id"]] += 1
        conditions.append(
            {
                **condition,
                "ranked": True,
                "matched": observed["before"]["configuration_id"]
                == predicted["before"]["configuration_id"],
                "observed_winner": observed["before"]["configuration_id"],
                "predicted_winner": predicted["before"]["configuration_id"],
                "eligible_configurations": sorted(
                    r["before"]["configuration_id"] for r in feasible
                ),
            }
        )
    by_request: dict[str, list[float]] = defaultdict(list)
    for condition in conditions:
        by_request[condition["request_id"]].append(float(condition["matched"]))
    request_values = [mean(v) for v in by_request.values()]
    rng = random.Random(1707)
    samples = (
        [mean(rng.choices(request_values, k=len(request_values))) for _ in range(2000)]
        if request_values
        else []
    )
    return {
        "scheduled_request_block_conditions": len(grouped),
        "ranked_conditions": ranked,
        "matches": hits,
        "agreement_all_scheduled": hits / len(grouped) if grouped else None,
        "agreement_comparable": hits / ranked if ranked else None,
        "unranked_conditions": len(grouped) - ranked,
        "known_unsafe_feasibility_claims": unsafe,
        "predicted_choices": dict(choices),
        "independent_requests": len({k[0] for k in grouped}),
        "network_blocks": sorted({k[1] for k in grouped}),
        "request_cluster_ci95_all_scheduled": [
            percentile(samples, 0.025),
            percentile(samples, 0.975),
        ]
        if samples
        else None,
        "conditions": conditions,
        "scope": (
            "Offline fastest-feasible rank; not live execution "
            "or measured switching overhead"
        ),
    }


def paired_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(1707)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(2000)]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def choose_variant(
    approved: list[str], report: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[str | None, dict[str, Any]]:
    if not approved:
        return None, {"reason": "No model passed all mandatory gates"}
    if len(approved) == 1:
        return approved[0], {"reason": "Only one predeclared model passed"}
    rankings = {v: report["models"][v]["ranking_request_primary"] for v in approved}
    conditions = {
        v: {
            (r["request_id"], r["block"]): float(r["matched"])
            for r in rankings[v]["conditions"]
        }
        for v in approved
    }
    by_request: dict[str, list[float]] = defaultdict(list)
    for key, value in conditions["v2_a"].items():
        by_request[key[0]].append(value - conditions["v1_recalibrated"][key])
    ranking_delta = [mean(v) for v in by_request.values()]
    ranking_ci = paired_interval(ranking_delta)
    if ranking_ci is not None and ranking_ci[0] > 0:
        return "v2_a", {
            "reason": "Stable positive paired ranking gain",
            "ranking_gain_ci95": ranking_ci,
        }
    error_by_variant: dict[str, dict[str, list[float]]] = {
        v: defaultdict(list) for v in approved
    }
    for row in rows:
        if (
            row["variant"] in approved
            and row["status"] == "ok"
            and row["prediction"]["valid"]
        ):
            error_by_variant[row["variant"]][row["before"]["request_id"]].append(
                abs(row["prediction"]["total_ms"] - row["observed_ms"])
            )
    error_deltas = [
        mean(error_by_variant["v1_recalibrated"][rid])
        - mean(error_by_variant["v2_a"][rid])
        for rid in sorted(error_by_variant["v1_recalibrated"])
    ]
    baseline = mean(mean(v) for v in error_by_variant["v1_recalibrated"].values())
    relative_gain = mean(error_deltas) / baseline if baseline else 0.0
    error_ci = paired_interval(error_deltas)
    tied = rankings["v2_a"]["matches"] == rankings["v1_recalibrated"]["matches"]
    selected = (
        "v2_a"
        if tied and relative_gain >= 0.05 and error_ci is not None and error_ci[0] > 0
        else "v1_recalibrated"
    )
    return selected, {
        "reason": "Prefer simplicity unless ranking or MAE gain is stable",
        "ranking_gain_ci95": ranking_ci,
        "mae_gain_ci95_ms": error_ci,
        "relative_mae_gain": relative_gain,
        "minimum_relative_mae_gain": 0.05,
    }


def evaluate_validation(bundle: Path, fit_dir: Path, output: Path) -> dict[str, Any]:
    if (output / "evaluation_validation.json").exists():
        raise FileExistsError(
            "one validation decision already exists; no repeated selection"
        )
    inputs = {trial_key(r): r for r in _validation_inputs(bundle)}
    targets = _open_validation_after_seal(bundle, fit_dir, output)
    observations = {}
    for target in targets:
        key = (
            target["request_id"],
            target["network_state"]["bucket_id"],
            target["configuration_id"],
            target["repetition"],
        )
        if key not in inputs or key in observations:
            raise ValueError("unexpected/duplicate validation target")
        validate_target(target, inputs[key]["task"])
        observations[key] = target
    if set(observations) != set(inputs):
        raise ValueError("incomplete validation outcomes")
    predictions = [
        strict_json(line)
        for line in (output / "predictions_validation.jsonl").read_bytes().splitlines()
    ]
    if len(predictions) != 1080 or {
        (r["variant"], trial_key(r["before"])) for r in predictions
    } != {(v, k) for v in VARIANTS for k in inputs}:
        raise ValueError("sealed predictions do not cover every candidate trial")
    models = _models(fit_dir)
    evaluated = []
    for p in predictions:
        target = observations[trial_key(p["before"])]
        evaluated.append(
            {
                **p,
                "status": target["status"],
                "observed_ms": target["observed_end_to_end_ms"],
            }
        )
    report: dict[str, Any] = {
        "schema_version": 2,
        "validation_attempts": 360,
        "unique_validation_requests": 20,
        "blocks": ["N2", "N7"],
        "final_outcomes_read": False,
        "predictions_sealed_before_outcomes": True,
        "models": {},
    }
    for variant in VARIANTS:
        rows = [r for r in evaluated if r["variant"] == variant]
        cells = {}
        for block in ("N2", "N7"):
            for spec in CONFIGURATIONS:
                selected_rows = [
                    r
                    for r in rows
                    if r["before"]["block"] == block
                    and r["before"]["configuration_id"] == spec.id.value
                ]
                cells[block + "|" + spec.id.value] = {
                    "request_primary": scores(_request_rows(selected_rows)),
                    "repeat_secondary": scores(selected_rows),
                    "diagnostic_request_not_eligibility": scores(
                        _request_rows(selected_rows), diagnostic=True
                    ),
                }
        report["models"][variant] = {
            "by_block_configuration": cells,
            "ranking_request_primary": ranking(rows, models, variant),
            "aggregate": scores(rows),
            "ranking_by_block": {
                b: ranking(
                    [r for r in rows if r["before"]["block"] == b], models, variant
                )
                for b in ("N2", "N7")
            },
            "prediction_overhead_ms": {
                "median": median(r["prediction_overhead_ms"] for r in rows),
                "p95": percentile([r["prediction_overhead_ms"] for r in rows], 0.95),
                "scope": (
                    "predict() only; feature sensing, tokenization "
                    "and switching excluded"
                ),
            },
        }
    constant = report["models"]["constant_median"]["ranking_request_primary"][
        "agreement_all_scheduled"
    ]
    decisions = {}
    policy = models["v2_a"].policy
    for variant in ("v1_recalibrated", "v2_a"):
        model = report["models"][variant]
        primary = [
            s["request_primary"] for s in model["by_block_configuration"].values()
        ]
        coverage = all(
            s["scored"] == 20
            and s["prediction_coverage"] >= policy["minimum_validation_coverage"]
            for s in primary
        )
        numeric = all(
            s["median_ape"] is not None
            and s["median_ape"] <= policy["target_median_ape"]
            and s["p90_ape"] <= policy["target_p90_ape"]
            for s in primary
        )
        rank = model["ranking_request_primary"]
        ranking_met = (
            rank["ranked_conditions"] == 40
            and rank["agreement_all_scheduled"] >= constant
        )
        decisions[variant] = {
            "coverage_gate": coverage,
            "numerical_gate": numeric,
            "ranking_gate": ranking_met,
            "known_unsafe_claims": rank["known_unsafe_feasibility_claims"],
            "passed": coverage
            and numeric
            and ranking_met
            and rank["known_unsafe_feasibility_claims"] == 0,
        }
    approved = [v for v, d in decisions.items() if d["passed"]]
    chosen, selection_evidence = choose_variant(approved, report, evaluated)
    primary_v2 = [
        cell["request_primary"]
        for cell in report["models"]["v2_a"]["by_block_configuration"].values()
    ]
    supported_numerical_failure = any(
        cell["scored"]
        and (
            cell["median_ape"] > policy["target_median_ape"]
            or cell["p90_ape"] > policy["target_p90_ape"]
        )
        for cell in primary_v2
    )
    gate = (
        "G3 PASSED"
        if chosen
        else "G3 FAILED"
        if supported_numerical_failure
        else "G3 BLOCKED"
    )
    decision = {
        "schema_version": 2,
        "gate": gate,
        "selected_variant": chosen,
        "selection_evidence": selection_evidence,
        "candidate_gates": decisions,
        "reason": (
            "All-three-configuration coverage/numerical/ranking targets "
            "required; no edge/shared-memory eligibility waiver"
        ),
        "validation_seal_sha256": strict_json(
            (output / "predictions_seal.json").read_bytes()
        )["seal_sha256"],
        "model_ids": {v: m.calibration_id for v, m in models.items()},
        "selection_rule": (
            "Ranking gains need a positive paired request-bootstrap "
            "lower bound. At tied ranking, V2-A needs >=5% MAE gain "
            "and a positive paired lower bound; otherwise prefer V1. "
            "No validation refit."
        ),
        "final_evaluation_authorized": False,
        "prior_final_inspection_disclosed": True,
    }
    decision["decision_sha256"] = canonical_sha256(decision)
    write_json_atomic(output / "evaluation_validation.json", report)
    write_json_atomic(output / "frozen_model_decision.json", decision)
    (output / "validation_scores.jsonl").write_text(
        "".join(
            json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in evaluated
        )
    )
    if chosen:
        model = models[chosen]
        audit = strict_json((bundle / "partition_audit.json").read_bytes())
        fit_source = strict_json((fit_dir / "source_manifest.json").read_bytes())
        marker = {
            "schema_version": 2,
            "selected_variant": chosen,
            "calibration_sha256": digest(fit_dir / f"calibration_{chosen}.json"),
            "calibration_id": model.calibration_id,
            "frozen_request_split_sha256": model.split_sha256,
            "frozen_network_profile_sha256": canonical_sha256(
                {
                    "source": policy["network_source"],
                    "candidate_network_fits": {
                        cid: f["network"] for cid, f in model.candidates.items()
                    },
                }
            ),
            "training_rows_sha256": model.training_rows_sha256,
            "source_gate2_campaign_id": policy["campaign_id"],
            "source_gate2_archive_sha256": audit["archive_sha256"],
            "host_pair_id": policy["host_pair_id"],
            "environment_kind": "physical_wsl",
            "terraform_environment_id": None,
            "application_image_digest": None,
            "container_identity_scope": "Not applicable to native physical baseline",
            "model_contract_sha256": policy["model_contract_sha256"],
            "model_artifact_sha256": policy["model_artifact_sha256"],
            "dataset_manifest_sha256": policy["dataset_manifest_sha256"],
            "tokenizer_sha256": policy["tokenizer_sha256"],
            "code_fit_commit_sha": fit_source["code_git_commit"],
            "code_prediction_commit_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "feature_schema_version": 2,
            "network_extrapolation_and_safety_policy": {
                k: policy[k]
                for k in (
                    "network_envelope",
                    "automatic_extrapolation_enabled",
                    "memory_headroom_fraction",
                )
            },
            "uncertainty_method": (
                "Training request-grouped OOF maxima quantile; "
                "widened on physics extrapolation"
            ),
            "validation_decision_sha256": decision["decision_sha256"],
            "sla_ms": 60000,
            "max_new_tokens": 32,
            "random_seed": 1707,
            "final_evaluation_authorized": False,
            "prior_final_inspection_disclosed": True,
        }
        marker["freeze_sha256"] = canonical_sha256(marker)
        write_json_atomic(output / "model_freeze_marker.json", marker)
    return {
        "gate": decision["gate"],
        "selected_variant": chosen,
        "selection_evidence": selection_evidence,
        "candidate_gates": decisions,
    }


def review_training(bundle: Path, fit_dir: Path, output: Path) -> dict[str, Any]:
    """Training-only residual/cross-validation review; does not alter any fit."""
    import csv

    from adaptive_vlm.benchmark.predictor_v2_data import load_training

    if output.exists():
        raise FileExistsError("refusing to overwrite training review")
    training = load_training(bundle)
    split = strict_json((bundle / "split.json").read_bytes())
    lookup = {trial_key(r["inputs"]): r for r in training}
    receipts = strict_json((fit_dir / "file_hashes.json").read_bytes())
    reports = {}
    residuals = []
    manifests = []
    for filename in ("training_oof.jsonl", "training_lobo.jsonl"):
        path = fit_dir / filename
        if digest(path) != receipts[filename]:
            raise ValueError("training diagnostic hash changed")
        rows = [strict_json(line) for line in path.read_bytes().splitlines()]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                row["request_id"],
                row["block"],
                row["configuration_id"],
                row["repetition"],
            )
            source = lookup[key]
            groups[row["diagnostic"]].append(row)
            if row["prediction"]["total_ms"] is None or row["status"] != "ok":
                continue
            before = source["inputs"]
            residuals.append(
                {
                    "diagnostic": filename,
                    "fold": row["diagnostic"],
                    "variant": row["variant"],
                    "request_id": row["request_id"],
                    "block": row["block"],
                    "configuration_id": row["configuration_id"],
                    "repetition": row["repetition"],
                    "predicted_ms": row["prediction"]["total_ms"],
                    "observed_ms": row["observed_ms"],
                    "signed_error_ms": row["prediction"]["total_ms"]
                    - row["observed_ms"],
                    "uplink_bps": before["network"]["uplink_bps"],
                    "rtt_ms": before["network"]["rtt_ms"],
                    "image_bytes": before["task"]["image_bytes"],
                    "image_pixels": before["task"]["image_width"]
                    * before["task"]["image_height"],
                    "prompt_tokens": before["task"]["prompt_tokens"],
                    "predicted_expected_output_tokens": row["prediction"][
                        "estimated_output_tokens"
                    ],
                    "actual_output_tokens_training_diagnostic_only": source["target"][
                        "fields"
                    ]["actual_output_tokens"],
                }
            )
        by_model: dict[str, Any] = {}
        for variant in VARIANTS:
            by_model[variant] = {}
            for spec in CONFIGURATIONS:
                chosen = [
                    r
                    for r in rows
                    if r["variant"] == variant
                    and r["configuration_id"] == spec.id.value
                ]
                by_model[variant][spec.id.value] = {
                    "valid_only": scores(chosen),
                    "all_numeric_diagnostics": scores(chosen, diagnostic=True),
                    "by_block": {
                        b: scores(
                            [r for r in chosen if r["block"] == b], diagnostic=True
                        )
                        for b in sorted(BLOCKS["train"])
                    },
                }
        reports[filename] = by_model
        ids = sorted(split["training_request_ids"])
        for label, observed in groups.items():
            if label.startswith("grouped_request_fold_"):
                fold = int(label.rsplit("_", 1)[1])
                held_ids = {rid for i, rid in enumerate(ids) if i % 5 == fold}
                kept = [
                    r for r in training if r["inputs"]["request_id"] not in held_ids
                ]
                assert {r["request_id"] for r in observed} == held_ids
                held_blocks = []
            else:
                held_block = label.removeprefix("leave_training_block_").removesuffix(
                    "_out"
                )
                held_blocks = [held_block]
                held_ids = set()
                kept = [r for r in training if r["inputs"]["block"] != held_block]
                assert {r["block"] for r in observed} == {held_block}
            manifests.append(
                {
                    "label": label,
                    "training_requests": sorted(
                        {r["inputs"]["request_id"] for r in kept}
                    ),
                    "held_training_request_ids": sorted(held_ids),
                    "training_blocks": sorted({r["inputs"]["block"] for r in kept}),
                    "held_training_blocks": held_blocks,
                    "training_rows_sha256": canonical_sha256(kept),
                    "fold_calibration_ids": {
                        v: sorted(
                            {
                                r["prediction"]["calibration_id"]
                                for r in observed
                                if r["variant"] == v
                            }
                        )
                        for v in VARIANTS
                    },
                }
            )
    output.mkdir(parents=True)
    write_json_atomic(
        output / "training_internal_review.json",
        {
            "schema_version": 2,
            "reports": reports,
            "independent_requests": 60,
            "training_network_blocks": 5,
            "validation_outcomes_read": False,
            "final_outcomes_read": False,
            "uncertainty_note": (
                "OOF/LOBO point predictions precede interval fitting; their "
                "zero-width interval fields are not calibrated coverage."
            ),
        },
    )
    write_json_atomic(output / "training_cv_manifests.json", {"folds": manifests})
    if residuals:
        with (output / "training_residuals.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(residuals[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(residuals)
    return {
        "training_requests": 60,
        "training_network_blocks": 5,
        "cross_validation_manifests": len(manifests),
        "residual_diagnostics": len(residuals),
        "fits_modified": False,
        "validation_outcomes_read": False,
    }
