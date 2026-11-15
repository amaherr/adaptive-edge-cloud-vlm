"""Error, ranking and paired-interval scoring for V2 forecasts."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from adaptive_vlm.prediction.v1.calibration import percentile
from adaptive_vlm.prediction.v2.predictor import (
    CalibrationV2,
)


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
