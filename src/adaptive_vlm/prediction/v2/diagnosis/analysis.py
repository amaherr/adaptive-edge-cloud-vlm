"""Grouped summaries, repeat drift and feature associations."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any

from adaptive_vlm.prediction.v2.diagnosis.components import metrics
from adaptive_vlm.prediction.v2.diagnosis.rank_statistics import correlation, quantile


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = sorted(rows[0]["component_errors_ms"])
    baseline = metrics(rows)
    stage_reports = {}
    magnitude_sum = sum(
        mean(abs(r["component_errors_ms"][s]) for r in rows) for s in stages
    )
    for stage in stages:
        corrected = metrics(rows, stage)
        stage_reports[stage] = {
            "trial_mean_absolute_component_error_ms": mean(
                abs(r["component_errors_ms"][stage]) for r in rows
            ),
            "trial_signed_component_bias_ms": mean(
                r["component_errors_ms"][stage] for r in rows
            ),
            "share_of_absolute_component_error": mean(
                abs(r["component_errors_ms"][stage]) for r in rows
            )
            / magnitude_sum
            if magnitude_sum
            else 0,
            "oracle_replacement": corrected,
            "oracle_mae_reduction_fraction": 1
            - corrected["mae_ms"] / baseline["mae_ms"]
            if baseline["mae_ms"]
            else 0,
        }
    measured = {}
    for stage in sorted(rows[0]["measured_substages_ms"]):
        values = [r["measured_substages_ms"][stage] for r in rows]
        measured[stage] = {
            "trial_median_ms": median(values),
            "trial_p90_ms": quantile(values, 0.9),
        }
    return {
        "baseline": baseline,
        "components": stage_reports,
        "measured_substages": measured,
    }


def repeat_changes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired third-minus-first repeat contrasts within a block and request."""
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["request_id"]][row["repetition"]] = row
    changes: dict[str, list[float]] = defaultdict(list)
    for repeats in grouped.values():
        first, last = repeats[0], repeats[2]
        changes["observed_total_ms"].append(last["observed_ms"] - first["observed_ms"])
        for stage in first["observed_components_ms"]:
            changes[stage].append(
                last["observed_components_ms"][stage]
                - first["observed_components_ms"][stage]
            )
    return {
        name: {
            "paired_requests": len(values),
            "median_third_minus_first_ms": median(values),
            "p10_ms": quantile(values, 0.1),
            "p90_ms": quantile(values, 0.9),
        }
        for name, values in changes.items()
    }


def associations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Within-block request medians (60 units); repeated cells never pooled as IID."""
    result = []
    for block in sorted({r["block"] for r in rows}):
        group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["block"] == block:
                group[row["request_id"]].append(row)
        feature_names = sorted(rows[0]["features"])
        targets = [
            "signed_error_ms",
            "ape",
            *sorted(rows[0]["component_errors_ms"]),
            *["observed:" + k for k in sorted(rows[0]["observed_components_ms"])],
            *["measured:" + k for k in sorted(rows[0]["measured_substages_ms"])],
        ]
        for feature in feature_names:
            for target in targets:
                x, y = [], []
                for repeats in group.values():
                    values = [r["features"][feature] for r in repeats]
                    if any(v is None for v in values):
                        continue
                    x.append(median(values))
                    y.append(
                        median(
                            r[target]
                            if target in {"signed_error_ms", "ape"}
                            else r["observed_components_ms"][target.split(":", 1)[1]]
                            if target.startswith("observed:")
                            else r["measured_substages_ms"][target.split(":", 1)[1]]
                            if target.startswith("measured:")
                            else r["component_errors_ms"][target]
                            for r in repeats
                        )
                    )
                result.append(
                    {
                        "block": block,
                        "feature": feature,
                        "target": target,
                        "requests": len(x),
                        "spearman": correlation(x, y),
                    }
                )
    return result


def descriptive_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_crop = {}
    for crop in sorted({r["features"]["geometry_crop_count_diagnostic"] for r in rows}):
        group = [
            r for r in rows if r["features"]["geometry_crop_count_diagnostic"] == crop
        ]
        by_crop[str(crop)] = {
            "metrics": metrics(group),
            "observed_component_trial_medians_ms": {
                k: median(r["observed_components_ms"][k] for r in group)
                for k in group[0]["observed_components_ms"]
            },
            "component_error_trial_medians_ms": {
                k: median(r["component_errors_ms"][k] for r in group)
                for k in group[0]["component_errors_ms"]
            },
        }
    memory: dict[str, Any] = {}
    for block in sorted({r["block"] for r in rows}):
        memory[block] = {}
        for name in rows[0]["features"]:
            if "mib" not in name:
                continue
            values = [
                r["features"][name]
                for r in rows
                if r["block"] == block and r["features"][name] is not None
            ]
            memory[block][name] = {
                "observations": len(values),
                "unique_values": len(set(values)),
                "minimum": min(values) if values else None,
                "median": median(values) if values else None,
                "maximum": max(values) if values else None,
            }
    # Compare the same IDs across blocks; this is not an intervention on bandwidth.
    by_request_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_request_block[row["request_id"], row["block"]].append(row)
    cross_block = {}
    if "N3" in {r["block"] for r in rows}:
        for block in sorted({r["block"] for r in rows} - {"N3"}):
            shifts: dict[str, list[float]] = defaultdict(list)
            for rid in sorted({r["request_id"] for r in rows}):
                reference, current = (
                    by_request_block[rid, "N3"],
                    by_request_block[rid, block],
                )
                for stage in rows[0]["observed_components_ms"]:
                    shifts[stage].append(
                        median(r["observed_components_ms"][stage] for r in current)
                        - median(r["observed_components_ms"][stage] for r in reference)
                    )
            cross_block[block] = {
                s: {
                    "paired_requests": len(v),
                    "median_shift_ms": median(v),
                    "p10_shift_ms": quantile(v, 0.1),
                    "p90_shift_ms": quantile(v, 0.9),
                }
                for s, v in shifts.items()
            }
    return {
        "by_geometry_crop_count": by_crop,
        "memory_observations_by_block": memory,
        "matched_request_block_shift_relative_to_N3": cross_block,
    }
