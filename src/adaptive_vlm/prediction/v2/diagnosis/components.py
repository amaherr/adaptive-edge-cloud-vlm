"""Join saved V2 forecasts to observed stage components."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from statistics import mean, median
from typing import Any

from adaptive_vlm.models.moondream.geometry import crop_count
from adaptive_vlm.prediction.v2.diagnosis.rank_statistics import quantile


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
