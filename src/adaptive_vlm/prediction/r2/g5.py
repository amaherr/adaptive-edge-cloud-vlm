"""G5 analysis (amendment A4): the live adaptive policy on dynamic traces.

Every measured request of a segment ran the five fixed configurations and the
adaptive policy back to back (interleaved). Per request:

- the **oracle** is the fixed configuration with the lowest observed latency
  in that request's five fixed attempts (hindsight, not achievable online);
- the adaptive policy's **regret** is its observed latency (which includes
  sensing, prediction and selection) minus the oracle's;
- the **label counterfactual** reruns the frozen selector offline with the
  true load labels and the same network probe, and takes that choice's
  observed fixed latency: it separates sensing error from prediction error.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from adaptive_vlm.prediction.r2.g4 import decide
from adaptive_vlm.prediction.r2.models import Calibration
from adaptive_vlm.prediction.r2.units import CONFIGURATIONS, UnitInputs
from adaptive_vlm.prediction.v1.calibration import percentile
from adaptive_vlm.scheduler.configuration import SPEC_BY_PLACEMENT

SEED = 1707
RESAMPLES = 2000
MEDIAN_RELATIVE_REGRET_LIMIT = 10.0
FIXED = {spec.placement: spec.id.value for spec in SPEC_BY_PLACEMENT.values()}


def wire_bytes(row: Mapping[str, Any]) -> int:
    """Application bytes on the link for one attempt (0 for the edge)."""

    metrics = row.get("metrics") or {}
    if "total_wire_bytes" in metrics:
        return int(metrics["total_wire_bytes"])
    return int(metrics.get("request_wire_bytes") or 0) + int(
        metrics.get("response_wire_bytes") or 0
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def requests(campaign: Path) -> list[dict[str, Any]]:
    """One record per measured request of every G5 segment."""

    out = []
    for directory in sorted((campaign / "subblocks").iterdir()):
        plan = json.loads((directory / "plan.json").read_text())
        run = directory / "run"
        if not (run / "adaptive").exists():
            continue
        fixed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for placement, configuration in FIXED.items():
            for row in _rows(run / placement / "requests.jsonl"):
                if not row["warmup"]:
                    fixed[row["trial_id"]][configuration] = row
        for row in _rows(run / "adaptive" / "requests.jsonl"):
            if row["warmup"]:
                continue
            metrics = row.get("metrics") or {}
            out.append(
                {
                    "subblock_id": plan["subblock_id"],
                    "segment": plan["segment"],
                    "trace": str(plan["segment"]).rsplit("-s", 1)[0],
                    "cell_id": plan["cell_id"],
                    "condition_id": plan["condition"]["condition_id"],
                    "edge_load": plan["load_level"],
                    "cloud_load": plan["cloud_load"],
                    "repetition": plan["repetition"],
                    "trial_id": row["trial_id"],
                    "request_id": row["request_id"],
                    "interleave_position": row["interleave_position"],
                    "status": row["status"],
                    "latency_ms": row["observed_end_to_end_ms"]
                    if row["status"] == "ok"
                    else None,
                    "output_text": row.get("output_text"),
                    "selected": metrics.get("adaptive_selected_configuration_id"),
                    "sensed_edge_load": metrics.get("adaptive_sensed_edge_load"),
                    "sensed_cloud_load": metrics.get("adaptive_sensed_cloud_load"),
                    "edge_probe_ms": metrics.get("adaptive_sensed_edge_probe_ms"),
                    "cloud_power_w": metrics.get("adaptive_sensed_cloud_power_w"),
                    "overhead_ms": metrics.get("adaptive_overhead_ms"),
                    "request_wire_bytes": metrics.get("request_wire_bytes"),
                    # The network inputs the adaptive policy used (the
                    # segment's periodic probe on AWS).
                    "uplink_bps": metrics.get("adaptive_sensed_uplink_bps"),
                    "rtt_ms": metrics.get("adaptive_sensed_rtt_ms"),
                    "fixed_ms": {
                        c: (
                            r["observed_end_to_end_ms"] if r["status"] == "ok" else None
                        )
                        for c, r in fixed[row["trial_id"]].items()
                    },
                    "fixed_text": {
                        c: r.get("output_text")
                        for c, r in fixed[row["trial_id"]].items()
                    },
                    "fixed_bytes": {
                        c: wire_bytes(r) for c, r in fixed[row["trial_id"]].items()
                    },
                    "adaptive_bytes": wire_bytes(row),
                    "sensing_ms": metrics.get("adaptive_sensed_sensing_ms"),
                    "selection_ms": metrics.get("adaptive_selection_ms"),
                    "cloud_state_age_s": metrics.get(
                        "adaptive_sensed_cloud_state_age_s"
                    ),
                    "load_state_label": f"{plan['load_level']}/{plan['cloud_load']}",
                }
            )
    return out


def _bootstrap(
    groups: Mapping[str, Sequence[float]], statistic: Callable[[list[float]], float]
) -> dict[str, float]:
    keys = sorted(groups)
    generator = random.Random(SEED)
    estimates = sorted(
        statistic([v for k in generator.choices(keys, k=len(keys)) for v in groups[k]])
        for _ in range(RESAMPLES)
    )
    return {
        "estimate": statistic([v for k in keys for v in groups[k]]),
        "ci_low": percentile(estimates, 0.025),
        "ci_high": percentile(estimates, 0.975),
    }


def _latency_summary(values: list[float], sla_ms: float) -> dict[str, float]:
    return {
        "n": len(values),
        "mean_ms": mean(values),
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "sla_success_percent": 100.0 * sum(v <= sla_ms for v in values) / len(values),
    }


def analyse(
    records: Sequence[dict[str, Any]],
    *,
    calibration: Calibration,
    features: Mapping[str, Mapping[str, Any]],
    best_static: str,
    sla_ms: float,
) -> dict[str, Any]:
    complete = [
        r
        for r in records
        if r["latency_ms"] is not None
        and set(r["fixed_ms"]) == set(CONFIGURATIONS)
        and all(v is not None for v in r["fixed_ms"].values())
    ]
    for r in complete:
        oracle = min(
            CONFIGURATIONS, key=lambda c: (r["fixed_ms"][c], CONFIGURATIONS.index(c))
        )
        r["oracle"] = oracle
        r["oracle_ms"] = r["fixed_ms"][oracle]
        r["regret_ms"] = r["latency_ms"] - r["oracle_ms"]
        r["relative_regret_percent"] = 100.0 * r["regret_ms"] / r["oracle_ms"]
        task = features[r["request_id"]]
        labelled = decide(
            calibration,
            UnitInputs(
                request_id=r["request_id"],
                cell_id=r["cell_id"],
                condition_id=r["condition_id"],
                task="short",
                edge_load=r["edge_load"],
                cloud_load=r["cloud_load"],
                width=int(task["image_width"]),
                height=int(task["image_height"]),
                image_bytes=int(task["image_bytes"]),
                uplink_bps=float(r["uplink_bps"]),
                rtt_ms=float(r["rtt_ms"]),
            ),
        ).selected_configuration_id
        r["labelled_choice"] = labelled
        r["labelled_regret_ms"] = (
            r["fixed_ms"][labelled] - r["oracle_ms"] if labelled else None
        )
    by_request: dict[str, list[float]] = defaultdict(list)
    for r in complete:
        by_request[r["request_id"]].append(r["regret_ms"])
    differences = {}
    for configuration in CONFIGURATIONS:
        groups: dict[str, list[float]] = defaultdict(list)
        for r in complete:
            groups[r["request_id"]].append(
                r["latency_ms"] - r["fixed_ms"][configuration]
            )
        differences[configuration] = _bootstrap(groups, mean)
    vs_static = differences[best_static]
    relative = [r["relative_regret_percent"] for r in complete]
    edge_sensed = [r for r in complete if r["sensed_edge_load"] is not None]
    cloud_sensed = [r for r in complete if r["sensed_cloud_load"] is not None]
    ordered = sorted(
        complete, key=lambda r: (r["subblock_id"], r["interleave_position"])
    )
    switches = sum(
        a["selected"] != b["selected"]
        for a, b in itertools.pairwise(ordered)
        if a["subblock_id"] == b["subblock_id"]
    )
    overhead = [
        float(r["overhead_ms"]) for r in complete if r["overhead_ms"] is not None
    ]
    agree = [r["selected"] == r["oracle"] for r in complete]
    text_match = sum(
        r["output_text"] == r["fixed_text"].get(r["selected"]) for r in complete
    )
    summary: dict[str, Any] = {
        "requests": len(records),
        "complete_requests": len(complete),
        "failed_adaptive_requests": sum(r["status"] != "ok" for r in records),
        "segments": len({r["subblock_id"] for r in records}),
        "latency": {
            "adaptive": _latency_summary([r["latency_ms"] for r in complete], sla_ms),
            **{
                c: _latency_summary([r["fixed_ms"][c] for r in complete], sla_ms)
                for c in CONFIGURATIONS
            },
            "oracle": _latency_summary([r["oracle_ms"] for r in complete], sla_ms),
        },
        "adaptive_minus_fixed_ms": differences,
        "best_static": best_static,
        "regret": {
            "median_relative_percent": percentile(relative, 0.5),
            "p90_relative_percent": percentile(relative, 0.9),
            "mean_ms": _bootstrap(by_request, mean),
            "top1_agreement_percent": 100.0 * sum(agree) / len(agree),
        },
        "label_counterfactual": {
            "mean_regret_ms": mean(
                r["labelled_regret_ms"]
                for r in complete
                if r["labelled_regret_ms"] is not None
            ),
            "choice_matches_adaptive_percent": 100.0
            * sum(r["labelled_choice"] == r["selected"] for r in complete)
            / len(complete),
        },
        "sensing": {
            "edge_accuracy_percent": 100.0
            * sum(r["sensed_edge_load"] == r["edge_load"] for r in edge_sensed)
            / max(1, len(edge_sensed)),
            "cloud_accuracy_percent": 100.0
            * sum(r["sensed_cloud_load"] == r["cloud_load"] for r in cloud_sensed)
            / max(1, len(cloud_sensed)),
        },
        "overhead_ms": {
            "median": percentile(overhead, 0.5) if overhead else None,
            "p95": percentile(overhead, 0.95) if overhead else None,
        },
        "switches_within_segments": switches,
        "choices": dict(sorted(_counts(r["selected"] for r in complete).items())),
        "answer_equals_chosen_configuration": f"{text_match}/{len(complete)}",
    }
    summary["gate"] = {
        "faster_than_best_static": vs_static["ci_high"] < 0,
        "median_relative_regret_ok": summary["regret"]["median_relative_percent"]
        <= MEDIAN_RELATIVE_REGRET_LIMIT,
        "no_failed_adaptive_requests": summary["failed_adaptive_requests"] == 0,
    }
    summary["gate"]["pass"] = all(summary["gate"].values())
    summary["by_segment"] = {
        segment: {
            "state": (
                f"{rs[0]['condition_id']} {rs[0]['edge_load']}/{rs[0]['cloud_load']}"
            ),
            "adaptive_mean_ms": mean(r["latency_ms"] for r in rs),
            "best_static_mean_ms": mean(r["fixed_ms"][best_static] for r in rs),
            "oracle_mean_ms": mean(r["oracle_ms"] for r in rs),
            "choices": dict(sorted(_counts(r["selected"] for r in rs).items())),
        }
        for segment, rs in sorted(_group(complete, "subblock_id").items())
    }
    return summary


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        counts[str(v)] += 1
    return counts


def _group(
    records: Iterable[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[r[key]].append(r)
    return grouped


# --------------------------------------------------------------------------
# Phase 7: paired breakdowns, communication, overhead, stability


def _paired(
    records: Sequence[dict[str, Any]], other: Callable[[dict[str, Any]], float]
) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        groups[r["request_id"]].append(r["latency_ms"] - other(r))
    return _bootstrap(groups, mean)


def _comparisons(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "adaptive_mean_ms": mean(r["latency_ms"] for r in records),
        "cloud_mean_ms": mean(r["fixed_ms"]["cloud/raw_image"] for r in records),
        "edge_mean_ms": mean(r["fixed_ms"]["edge/full"] for r in records),
        "oracle_mean_ms": mean(r["oracle_ms"] for r in records),
        "minus_cloud_ms": _paired(records, lambda r: r["fixed_ms"]["cloud/raw_image"]),
        "minus_edge_ms": _paired(records, lambda r: r["fixed_ms"]["edge/full"]),
        "minus_oracle_ms": _paired(records, lambda r: r["oracle_ms"]),
        "top1_agreement_percent": 100.0
        * sum(r["selected"] == r["oracle"] for r in records)
        / len(records),
    }


def phase7(
    records: Sequence[dict[str, Any]], features: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Breakdowns for the final analysis; ``records`` went through ``analyse``."""

    from adaptive_vlm.prediction.r2.features import crop_count

    complete = [r for r in records if "oracle" in r]
    crops = {
        rid: crop_count(int(f["image_width"]), int(f["image_height"]))
        for rid, f in features.items()
    }
    out: dict[str, Any] = {
        "by_load_state": {
            k: _comparisons(v)
            for k, v in sorted(_group(complete, "load_state_label").items())
        },
        "by_trace": {
            k: _comparisons(v) for k, v in sorted(_group(complete, "trace").items())
        },
        "by_repetition": {
            str(k): _comparisons(v)
            for k, v in sorted(_group(complete, "repetition").items())
        },
    }
    bytes_mean: dict[str, float] = {
        c: mean(r["fixed_bytes"][c] for r in complete) for c in CONFIGURATIONS
    }
    bytes_mean["adaptive"] = mean(r["adaptive_bytes"] for r in complete)
    bytes_mean["oracle"] = mean(r["fixed_bytes"][r["oracle"]] for r in complete)
    out["bytes_per_request_mean"] = bytes_mean
    overhead = [float(r["overhead_ms"]) for r in complete]
    regret = mean(r["regret_ms"] for r in complete)
    probe = {
        level: [float(r["edge_probe_ms"]) for r in complete if r["edge_load"] == level]
        for level in ("L0", "L1")
    }
    ages = [
        float(r["cloud_state_age_s"])
        for r in complete
        if r["cloud_state_age_s"] is not None
    ]
    out["overhead"] = {
        "overhead_ms_mean": mean(overhead),
        "sensing_ms_median": percentile(
            [float(r["sensing_ms"]) for r in complete], 0.5
        ),
        "selection_ms_median": percentile(
            [float(r["selection_ms"]) for r in complete], 0.5
        ),
        "edge_probe_ms_median": {k: percentile(v, 0.5) for k, v in probe.items() if v},
        "cloud_state_age_s_median": percentile(ages, 0.5) if ages else None,
        "saved_vs_cloud_ms_mean": mean(
            r["fixed_ms"]["cloud/raw_image"] - r["latency_ms"] for r in complete
        ),
        "regret_ms_mean": regret,
        "overhead_share_of_regret_percent": 100.0 * mean(overhead) / regret,
    }
    kinds = {"different_crop_counts": 0, "equal_crop_counts": 0}
    flip_flops = 0
    per_segment: list[int] = []
    for _, rows in sorted(_group(complete, "subblock_id").items()):
        rows = sorted(rows, key=lambda r: r["interleave_position"])
        count = 0
        for a, b in itertools.pairwise(rows):
            if a["selected"] != b["selected"]:
                count += 1
                same = crops[a["request_id"]] == crops[b["request_id"]]
                kinds["equal_crop_counts" if same else "different_crop_counts"] += 1
        for a, b, c in zip(rows, rows[1:], rows[2:], strict=False):
            if (
                a["selected"] == c["selected"] != b["selected"]
                and crops[a["request_id"]]
                == crops[b["request_id"]]
                == crops[c["request_id"]]
            ):
                flip_flops += 1
        per_segment.append(count)
    out["stability"] = {
        "switches": sum(per_segment),
        "per_segment_mean": mean(per_segment),
        "per_segment_max": max(per_segment),
        "between_different_crop_counts": kinds["different_crop_counts"],
        "between_equal_crop_counts": kinds["equal_crop_counts"],
        "flip_flops_equal_crops": flip_flops,
    }
    first_ids = {
        id(min(rows, key=lambda r: r["interleave_position"]))
        for rows in _group(complete, "subblock_id").values()
    }
    first = [r for r in complete if id(r) in first_ids]
    rest = [r for r in complete if id(r) not in first_ids]

    def top1(rows: Sequence[dict[str, Any]]) -> float:
        return float(
            100.0 * sum(r["selected"] == r["oracle"] for r in rows) / len(rows)
        )

    out["state_changes"] = {
        "first_request_top1_percent": top1(first),
        "later_requests_top1_percent": top1(rest),
        "first_request_sensing_correct_percent": 100.0
        * sum(
            r["sensed_edge_load"] == r["edge_load"]
            and r["sensed_cloud_load"] == r["cloud_load"]
            for r in first
        )
        / len(first),
    }
    wrong = [r for r in complete if r["sensed_edge_load"] != r["edge_load"]]
    out["edge_misreads"] = {
        "count": len(wrong),
        "by_true_level": dict(sorted(_counts(r["edge_load"] for r in wrong).items())),
        "regret_ms_mean": mean(r["regret_ms"] for r in wrong) if wrong else None,
        "top1_percent": top1(wrong) if wrong else None,
    }
    return out
