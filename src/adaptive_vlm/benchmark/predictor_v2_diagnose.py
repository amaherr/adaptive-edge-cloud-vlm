"""Training-only attribution of existing out-of-fold Predictor V2 errors.

This module neither fits a predictor nor reads validation/final outcomes.
Oracle component replacements are retrospective diagnostics, never features.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.predictor_v2_data import digest, load_training, strict_json


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        for index in ordered[start:stop]:
            result[index] = (start + stop - 1) / 2
        start = stop
    return result


def correlation(x: list[float], y: list[float]) -> float | None:
    """Spearman association with average ranks for ties; no significance claim."""
    if len(x) < 3:
        return None
    xr, yr = ranks(x), ranks(y)
    xm, ym = mean(xr), mean(yr)
    denominator = math.sqrt(
        sum((v - xm) ** 2 for v in xr) * sum((v - ym) ** 2 for v in yr)
    )
    return (
        sum((a - xm) * (b - ym) for a, b in zip(xr, yr, strict=True)) / denominator
        if denominator
        else None
    )


def crop_count(height: int, width: int) -> int:
    """Pinned Moondream geometry: 378px crops, 14px patch, margin 4, max 12.

    Mirrors select_tiling without importing/executing model code or loading images.
    Diagnostic only: this feature is not added to the frozen predictor.
    """
    height, width = height - 112, width - 112
    size, maximum = 266, 12
    if height <= size or width <= size:
        return 2
    min_h, min_w = math.ceil(height / size), math.ceil(width / size)
    if min_h * min_w > maximum:
        ratio = math.sqrt(maximum / (min_h * min_w))
        h, w = max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio))
    else:
        h = max(math.floor(math.sqrt(maximum * height / width)), min_h)
        w = max(math.floor(math.sqrt(maximum * width / height)), min_w)
        if h * w > maximum:
            if w > h:
                w = math.floor(maximum / h)
            else:
                h = math.floor(maximum / w)
        h, w = max(1, h), max(1, w)
    return h * w + 1


def observed_components(row: dict[str, Any]) -> dict[str, float]:
    target = row["target"]
    f = target["fields"]
    mode = row["inputs"]["configuration_id"].split("/")[0]
    result = {"decode_ms": f["decode_ms"], "client_residual_ms": f["stage_residual_ms"]}
    if mode == "edge":
        result["edge_nondecode_ms"] = f["edge_total_ms"] - f["decode_ms"]
        result["transport_ms"] = 0.0
    else:
        result["cloud_nondecode_including_prefill_ms"] = (
            f["server_processing_ms"] - f["decode_ms"]
        )
        result["transport_ms"] = f["transport_residual_ms"]
        if mode == "cloud":
            result["client_image_read_ms"] = target["raw_metrics"]["image_read_ms"]
        else:
            result["edge_vision_and_projector_ms"] = f["edge_total_ms"]
            result["tensor_serialization_ms"] = f["serialization_ms"]
    if not math.isclose(
        sum(result.values()), target["observed_end_to_end_ms"], abs_tol=1e-6
    ):
        raise ValueError("observed component accounting does not close")
    return result


def metrics(
    rows: list[dict[str, Any]], replacement: str | None = None
) -> dict[str, Any]:
    """Median three repeats first; 300 request/block cells are not independent IDs."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["request_id"], row["block"]].append(row)
    absolute, ape, signed = [], [], []
    for group in grouped.values():
        observed = median(r["observed_ms"] for r in group)
        predicted = median(
            r["predicted_ms"]
            - (r["component_errors_ms"][replacement] if replacement else 0)
            for r in group
        )
        signed.append(predicted - observed)
        absolute.append(abs(predicted - observed))
        ape.append(abs(predicted - observed) / observed)
    return {
        "request_block_cells": len(grouped),
        "independent_request_ids": len({r["request_id"] for r in rows}),
        "mae_ms": mean(absolute),
        "median_ape": median(ape),
        "p90_ape": quantile(ape, 0.9),
        "signed_bias_ms": mean(signed),
    }


def _key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    snapshot = row.get("inputs", row)
    return (
        snapshot["request_id"],
        snapshot["block"],
        snapshot["configuration_id"],
        snapshot["repetition"],
    )


def join_diagnostics(
    training: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source = {_key(row): row for row in training}
    if len(source) != len(training):
        raise ValueError("duplicate training schedule")
    result, seen = [], set()
    for p in predictions:
        if p["variant"] != "v2_a":
            continue
        key = _key(p)
        if key not in source or key in seen:
            raise ValueError("forbidden/duplicate diagnostic request/network slot")
        seen.add(key)
        row = source[key]
        actual = observed_components(row)
        predicted = p["prediction"]["components_ms"]
        if set(actual) != set(predicted):
            raise ValueError("component schema mismatch")
        observed, forecast = (
            row["target"]["observed_end_to_end_ms"],
            p["prediction"]["total_ms"],
        )
        if not math.isclose(observed, p["observed_ms"], abs_tol=1e-6):
            raise ValueError("diagnostic target mismatch")
        errors = {name: predicted[name] - value for name, value in actual.items()}
        if not math.isclose(sum(errors.values()), forecast - observed, abs_tol=1e-6):
            raise ValueError("component residual accounting does not close")
        task, fields = row["inputs"]["task"], row["target"]["fields"]
        result.append(
            {
                "request_id": key[0],
                "block": key[1],
                "configuration_id": key[2],
                "repetition": key[3],
                "trial_id": row["inputs"]["trial_id"],
                "diagnostic": p["diagnostic"],
                "observed_ms": observed,
                "predicted_ms": forecast,
                "signed_error_ms": forecast - observed,
                "ape": abs(forecast - observed) / observed,
                "valid": p["prediction"]["valid"],
                "reason": p["prediction"]["reason"],
                "observed_components_ms": actual,
                "component_errors_ms": errors,
                "features": {
                    "image_pixels": task["image_width"] * task["image_height"],
                    "image_bytes": task["image_bytes"],
                    "aspect_ratio": max(task["image_width"], task["image_height"])
                    / min(task["image_width"], task["image_height"]),
                    "geometry_crop_count_diagnostic": crop_count(
                        task["image_height"], task["image_width"]
                    ),
                    "prompt_tokens": task["prompt_tokens"],
                    "actual_output_tokens_posthoc": fields["actual_output_tokens"],
                    "predicted_output_tokens": p["prediction"].get(
                        "estimated_output_tokens"
                    ),
                    "edge_peak_reserved_mib_posthoc": fields["edge_peak_reserved_mib"],
                    "cloud_peak_reserved_mib_posthoc": fields[
                        "cloud_peak_reserved_mib"
                    ],
                    "edge_process_rss_mib_posthoc": fields["edge_process_rss_mib"],
                    "cloud_process_rss_mib_posthoc": fields["cloud_process_rss_mib"],
                    "time_utc_seconds_posthoc": datetime.fromisoformat(
                        row["target"]["measured_at_utc"]
                    ).timestamp(),
                    "probe_age_seconds": row["inputs"]["network"]["probe_age_seconds"],
                    "repetition": key[3],
                },
                "measured_substages_ms": {
                    name: fields[name]
                    for name in [
                        "edge_vision_ms",
                        "edge_projector_ms",
                        "image_encode_and_prefill_ms",
                        "image_prefill_ms",
                        "prompt_prefill_ms",
                        "prompt_prepare_ms",
                        "deserialization_ms",
                    ]
                    if fields[name] is not None
                },
            }
        )
    if seen != set(source):
        raise ValueError("incomplete diagnostic schedule")
    return result


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


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def diagnose(bundle: Path, fit: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    training = load_training(bundle)
    receipts = strict_json((fit / "file_hashes.json").read_bytes())
    reports: dict[str, Any] = {}
    residuals: list[dict[str, Any]] = []
    assoc_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    for filename in ["training_oof.jsonl", "training_lobo.jsonl"]:
        path = fit / filename
        if digest(path) != receipts[filename]:
            raise ValueError("diagnostic prediction hash mismatch")
        predictions = [strict_json(line) for line in path.read_bytes().splitlines()]
        joined = join_diagnostics(training, predictions)
        reports[filename] = {}
        for cid in sorted({r["configuration_id"] for r in joined}):
            rows = [r for r in joined if r["configuration_id"] == cid]
            report = summarize(rows)
            report["feature_diagnostics"] = descriptive_features(rows)
            report["by_block"] = {
                b: summarize([r for r in rows if r["block"] == b])
                for b in sorted({r["block"] for r in rows})
            }
            report["paired_repeat_changes_by_block"] = {
                b: repeat_changes([r for r in rows if r["block"] == b])
                for b in report["by_block"]
            }
            report["by_block_repetition"] = [
                {
                    "block": b,
                    "repetition": rep,
                    **summarize(
                        [r for r in rows if r["block"] == b and r["repetition"] == rep]
                    ),
                }
                for b in sorted({r["block"] for r in rows})
                for rep in range(3)
            ]
            reports[filename][cid] = report
            for a in associations(rows):
                assoc_rows.append(
                    {"diagnostic": filename, "configuration_id": cid, **a}
                )
            for scope, value in [
                ("all_training_blocks", report),
                *report["by_block"].items(),
            ]:
                for stage, stats in value["components"].items():
                    stage_rows.append(
                        {
                            "diagnostic": filename,
                            "configuration_id": cid,
                            "block": scope,
                            "component": stage,
                            **{
                                k: v
                                for k, v in stats.items()
                                if k != "oracle_replacement"
                            },
                            **{
                                "oracle_" + k: v
                                for k, v in stats["oracle_replacement"].items()
                            },
                        }
                    )
            for sort_key in ["ape", "absolute_error_ms"]:
                for rank, row in enumerate(
                    sorted(
                        rows,
                        key=lambda r: (
                            r["ape"] if sort_key == "ape" else abs(r["signed_error_ms"])
                        ),
                        reverse=True,
                    )[:15],
                    1,
                ):
                    outliers.append(
                        {"ranking": sort_key, "rank": rank, "source": filename, **row}
                    )
        residuals.extend({"source": filename, **r} for r in joined)
    output.mkdir(parents=True)
    report = {
        "scope": (
            "Physical training data only; AWS/Vultr excluded; "
            "no fit or candidate selection"
        ),
        "validation_outcomes_read": False,
        "final_outcomes_read": False,
        "training_attempts": len(training),
        "independent_requests": len({r["inputs"]["request_id"] for r in training}),
        "training_blocks": sorted({r["inputs"]["block"] for r in training}),
        "interpretation": (
            "Descriptive associations, not causal effects or independent "
            "repeated-trial tests. Oracle replacements use measured outcomes "
            "and cannot be deployed. Component absolute-error shares are not "
            "additive contributions to total MAE."
        ),
        "reports": reports,
    }
    write_json_atomic(output / "diagnosis.json", report)
    write_json_atomic(output / "largest_outliers.json", {"outliers": outliers})
    with (output / "component_residuals.jsonl").open("w") as handle:
        for row in residuals:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _csv(output / "component_summary.csv", stage_rows)
    _csv(output / "associations.csv", assoc_rows)
    return report
