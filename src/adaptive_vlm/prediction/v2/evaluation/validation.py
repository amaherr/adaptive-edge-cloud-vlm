"""One-shot V2 validation pass and gate decision."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.prediction.v1.calibration import canonical_sha256, percentile
from adaptive_vlm.prediction.v2.dataset.contract import (
    validate_target,
)
from adaptive_vlm.prediction.v2.dataset.parsing import digest, strict_json
from adaptive_vlm.prediction.v2.evaluation.scoring import (
    _request_rows,
    choose_variant,
    ranking,
    scores,
)
from adaptive_vlm.prediction.v2.evaluation.sealing import (
    _models,
    _open_validation_after_seal,
    _validation_inputs,
    trial_key,
)
from adaptive_vlm.prediction.v2.predictor import (
    VARIANTS,
)
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS


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
