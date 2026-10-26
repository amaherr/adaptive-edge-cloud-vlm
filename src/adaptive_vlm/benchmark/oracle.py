"""Retrospective measured oracle; never used to route a live request."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.calibrate import percentile
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS, CapabilityStatus
from adaptive_vlm.scheduler.predictor import Prediction
from adaptive_vlm.scheduler.selector import (
    CandidateSnapshot,
    SelectionConstraints,
    predicted_oracle,
    select,
)


def _interval(
    values: list[float], *, seed: int, resamples: int
) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    samples = [
        median(generator.choices(values, k=len(values))) for _ in range(resamples)
    ]
    return {
        "lower": percentile(samples, 0.025),
        "upper": percentile(samples, 0.975),
    }


def evaluate_oracle(
    rows: list[dict[str, Any]], specification: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare precomputed decisions with matched outcomes from every safe candidate."""
    if specification.get("schema_version") != 1:
        raise ValueError("unsupported oracle specification")
    if not isinstance(specification.get("safety_evidence_verified"), bool):
        raise ValueError("oracle specification needs a safety evidence flag")
    sections = specification["sections"]
    request_ids = specification["holdout_request_ids"]
    if not sections or not request_ids or len(request_ids) != len(set(request_ids)):
        raise ValueError("oracle needs unique frozen holdout IDs and sections")
    if (
        not isinstance(specification.get("seed"), int)
        or not isinstance(specification.get("bootstrap_resamples"), int)
        or specification["bootstrap_resamples"] < 100
    ):
        raise ValueError("oracle needs a seed and at least 100 bootstrap resamples")
    constraints = SelectionConstraints(
        sla_ms=specification.get("sla_ms"),
        quality_floor=specification.get("quality_floor"),
        edge_memory_budget_mib=specification.get("edge_memory_budget_mib"),
    )
    availability = specification["availability"]
    expected_ids = {spec.id.value for spec in CONFIGURATIONS}
    if set(availability) != expected_ids:
        raise ValueError("availability must name every configuration")
    allowed = set(request_ids)
    groups: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in rows:
        if row["section"] not in sections or row["request_id"] not in allowed:
            continue
        configuration_id = row["configuration_id"]
        if configuration_id not in expected_ids:
            raise ValueError(f"unknown configuration: {configuration_id}")
        key = (
            row["section"],
            row["request_id"],
            row["network_bucket_id"],
            row["task_bucket"],
            str(row.get("condition_block_id", "0")),
        )
        if configuration_id in groups[key]:
            raise ValueError(f"duplicate configuration within condition block: {key}")
        groups[key][configuration_id] = row
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for key, group in sorted(groups.items()):
        section, request_id, network_bucket_id, task_bucket, block_id = key
        snapshots: list[CandidateSnapshot] = []
        for spec in CONFIGURATIONS:
            config_id = spec.id.value
            state = availability[config_id]
            status = CapabilityStatus(spec.id, state["available"], state.get("reason"))
            raw = group.get(config_id)
            prediction = Prediction(**raw["prediction"]) if raw is not None else None
            if prediction is not None and prediction.reason == "failed_trial":
                raise ValueError(
                    "outcome-derived failed_trial prediction cannot enter a decision"
                )
            if (
                prediction is not None
                and prediction.calibration_id != specification["calibration_id"]
            ):
                raise ValueError(
                    "prediction calibration does not match oracle specification"
                )
            snapshots.append(
                CandidateSnapshot(
                    spec.id,
                    status,
                    prediction,
                    healthy=state["healthy"],
                    quality_score=state.get("quality_score"),
                )
            )
        decision = select(tuple(snapshots), constraints)
        oracle_prediction = predicted_oracle(decision)
        if decision.selected_configuration_id != oracle_prediction:
            raise AssertionError("selector differs from predicted oracle")
        for candidate in decision.candidates:
            if candidate.exclusion_reason is not None:
                exclusions[candidate.exclusion_reason] += 1
        feasible = [
            item.configuration_id for item in decision.candidates if item.eligible
        ]
        observed: dict[str, float] = {}
        incomplete_reason = None
        for config_id in feasible:
            raw = group.get(config_id)
            value = raw.get("observed_ms") if raw is not None else None
            if (
                raw is None
                or raw.get("status") != "ok"
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                incomplete_reason = "missing_or_failed_feasible_observation"
                break
            observed[config_id] = float(value)
        measured_id = (
            min(
                observed,
                key=lambda item: (
                    observed[item],
                    next(
                        i
                        for i, spec in enumerate(CONFIGURATIONS)
                        if spec.id.value == item
                    ),
                ),
            )
            if incomplete_reason is None and observed
            else None
        )
        selected_id = decision.selected_configuration_id
        selected_observed = observed.get(selected_id) if selected_id else None
        oracle_ms = observed.get(measured_id) if measured_id else None
        regret_ms = (
            selected_observed - oracle_ms
            if selected_observed is not None and oracle_ms is not None
            else None
        )
        records.append(
            {
                "schema_version": 1,
                "section": section,
                "request_id": request_id,
                "network_bucket_id": network_bucket_id,
                "task_bucket": task_bucket,
                "condition_block_id": block_id,
                "decision": decision.to_dict(),
                "predicted_oracle_configuration_id": oracle_prediction,
                "measured_oracle_configuration_id": measured_id,
                "observed_feasible_ms": observed if incomplete_reason is None else None,
                "selected_observed_ms": selected_observed,
                "measured_oracle_ms": oracle_ms,
                "regret_ms": regret_ms,
                "regret_fraction": regret_ms / oracle_ms
                if regret_ms is not None and oracle_ms is not None
                else None,
                "observed_sla_violated": selected_observed > constraints.sla_ms
                if selected_observed is not None and constraints.sla_ms is not None
                else None,
                "evaluation_exclusion_reason": incomplete_reason,
            }
        )
    paired = [item for item in records if item["regret_ms"] is not None]
    regrets_ms = [item["regret_ms"] for item in paired]
    regrets_fraction = [item["regret_fraction"] for item in paired]
    agreement = [
        float(
            item["decision"]["selected_configuration_id"]
            == item["measured_oracle_configuration_id"]
        )
        for item in paired
    ]
    seed = specification["seed"]
    resamples = specification["bootstrap_resamples"]
    repeated: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for item in paired:
        repeated[
            (
                item["section"],
                item["request_id"],
                item["network_bucket_id"],
                item["task_bucket"],
            )
        ].append(item["measured_oracle_ms"])
    report = {
        "schema_version": 1,
        "calibration_id": specification["calibration_id"],
        "evidence_scope": specification["evidence_scope"],
        "condition_blocks": len(records),
        "paired_blocks": len(paired),
        "incomplete_blocks": len(records) - len(paired),
        "no_feasible_blocks": sum(
            item["decision"]["status"] == "no_feasible_candidate" for item in records
        ),
        "predicted_oracle_agreement": all(
            item["decision"]["selected_configuration_id"]
            == item["predicted_oracle_configuration_id"]
            for item in records
        ),
        "median_regret_ms": median(regrets_ms) if paired else None,
        "median_regret_fraction": median(regrets_fraction) if paired else None,
        "median_regret_ms_95pct_bootstrap_ci": _interval(
            regrets_ms, seed=seed, resamples=resamples
        ),
        "median_regret_fraction_95pct_bootstrap_ci": _interval(
            regrets_fraction, seed=seed + 1, resamples=resamples
        ),
        "top1_agreement": sum(agreement) / len(agreement) if agreement else None,
        "top1_agreement_95pct_bootstrap_ci": _interval(
            agreement, seed=seed + 2, resamples=resamples
        ),
        "observed_sla_violations": sum(
            item["observed_sla_violated"] is True for item in paired
        ),
        "exclusions": dict(sorted(exclusions.items())),
        "repeated_condition_oracle_uncertainty": [
            {
                "section": key[0],
                "request_id": key[1],
                "network_bucket_id": key[2],
                "task_bucket": key[3],
                "blocks": len(values),
                "oracle_ms_p10": percentile(values, 0.1),
                "oracle_ms_p90": percentile(values, 0.9),
            }
            for key, values in sorted(repeated.items())
            if len(values) >= 2
        ],
        "small_sample_warning": len(paired) < 10,
        "bootstrap_intervals_descriptive_only": len(paired) < 10,
        "local_regret_target_met": bool(paired) and median(regrets_fraction) <= 0.10,
        "full_g4_gate_evaluable": (
            specification["evidence_scope"] == "physical-controlled"
            and specification["safety_evidence_verified"]
            and len(paired) >= 10
            and len(records) == len(paired)
            and any(item["section"] == "network_holdout" for item in paired)
        ),
        "automatic_activation_allowed": False,
    }
    report["full_g4_gate_passed"] = (
        report["full_g4_gate_evaluable"]
        and report["local_regret_target_met"]
        and report["predicted_oracle_agreement"]
    )
    return report, records


def run_oracle_evaluation(
    *, predictions_path: Path, specification_path: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"oracle output already exists: {output_dir}")
    content = predictions_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    specification_bytes = specification_path.read_bytes()
    specification = json.loads(specification_bytes)
    if (
        not isinstance(specification, dict)
        or specification.get("predictions_sha256") != digest
    ):
        raise ValueError("predictions SHA-256 differs from frozen oracle specification")
    rows = [json.loads(line) for line in content.decode().splitlines()]
    report, records = evaluate_oracle(rows, specification)
    report["predictions_sha256"] = digest
    report["specification_sha256"] = hashlib.sha256(specification_bytes).hexdigest()
    output_dir.mkdir(parents=True)
    write_json_atomic(output_dir / "evaluation.json", report)
    (output_dir / "decisions.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    return report
