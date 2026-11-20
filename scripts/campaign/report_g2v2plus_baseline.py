"""G2 v2+ baseline report tables and figures (roadmap phase 2).

Training cells only for everything that reads latency (amendment A3). The
inputs are the campaign folder and the training oracle produced by
``derive_g2v2_tables.py``. The ``gates`` step reads every sub-block but only
the non-latency evidence of section 8 (probes, queues, failures, memory and
load liveness), reported per split.

Steps:

- ``tables``: crossover, paired load effects with bootstrap CIs, split
  inference anatomy, int8 gain, the ``prompt_first`` null result, static
  policies, the long task, and the hardware comparison with physical G2;
- ``gates``: section 8 validity evidence for all 196 sub-blocks;
- ``figures``: the crossover map and latency against link rate (SVG).

Bootstrap CIs resample request IDs (both repetitions of a sampled ID stay
together), seed 1707, 2,000 resamples, as in section 9.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from adaptive_vlm.campaign.network import measurement_failures

SEED = 1707
RESAMPLES = 2000
EDGE, CLOUD = "edge/full", "cloud/raw_image"
SYNC = "distributed/projected/synchronous"
FIRST = "distributed/projected/prompt_first"
INT8 = "distributed/projected/int8"
CONFIGURATIONS = (EDGE, CLOUD, SYNC, FIRST, INT8)
SHORT_NAMES = {
    EDGE: "edge",
    CLOUD: "cloud",
    SYNC: "synchronous",
    FIRST: "prompt_first",
    INT8: "int8",
}
LOAD_STATES = (("L0", "C0"), ("L0", "C2"), ("L1", "C0"), ("L1", "C2"))


# --------------------------------------------------------------------------
# Inputs


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _cell_parts(cell_id: str) -> tuple[str, str, str, str]:
    """(condition, edge load, cloud load, task) from a cell ID."""

    parts = cell_id.split("-")
    task = "long" if "long" in parts else "short"
    cloud = next((p for p in parts if p.startswith("C")), "C0")
    return parts[0], parts[1], cloud, task


def load_oracle(oracle: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((oracle / "oracle_cells.json").read_text())
    if summary["splits"] != ["training"]:
        raise ValueError("the baseline report reads the training oracle only")
    attempts = list(_jsonl(oracle / "oracle_attempts.jsonl"))
    if any(a["split"] != "training" for a in attempts):
        raise ValueError("non-training attempt in the training oracle")
    return summary, attempts


def training_rows(campaign: Path) -> list[dict[str, Any]]:
    """Measured rows of every training sub-block, with the cell attached."""

    rows = []
    for subblock in sorted((campaign / "subblocks").iterdir()):
        plan = json.loads((subblock / "plan.json").read_text())
        if plan["condition"]["split"] != "training":
            continue
        for run in sorted(p for p in (subblock / "run").iterdir() if p.is_dir()):
            for row in _jsonl(run / "requests.jsonl"):
                if row["warmup"] or row["status"] != "ok":
                    continue
                rows.append(
                    {
                        "cell_id": plan["cell_id"],
                        "condition_id": plan["condition"]["condition_id"],
                        "rate_mbps": plan["condition"]["rate_mbps"],
                        "added_rtt_ms": plan["condition"]["added_rtt_ms"],
                        "load_level": plan["load_level"],
                        "cloud_load": plan["cloud_load"],
                        "task": plan["task"],
                        "repetition": plan["repetition"],
                        "request_id": row["request_id"],
                        "configuration_id": row["configuration_id"],
                        "e2e_ms": row["observed_end_to_end_ms"],
                        "metrics": row["metrics"],
                    }
                )
    return rows


# --------------------------------------------------------------------------
# Statistics


def _cluster_bootstrap(
    groups: Mapping[str, Sequence[Any]],
    statistic: Callable[[list[Any]], float],
) -> tuple[float, float, float]:
    """Point estimate and 95% percentile CI, resampling whole groups."""

    keys = sorted(groups)
    point = statistic([v for k in keys for v in groups[k]])
    generator = random.Random(SEED)
    estimates = sorted(
        statistic([v for k in generator.choices(keys, k=len(keys)) for v in groups[k]])
        for _ in range(RESAMPLES)
    )
    return (
        point,
        estimates[int(0.025 * (RESAMPLES - 1))],
        estimates[math.ceil(0.975 * (RESAMPLES - 1))],
    )


def _paired(
    latency: dict[tuple[str, str, int], float],
    a: str,
    b: str,
) -> dict[str, list[float]]:
    """Per request ID, the paired differences b - a over both repetitions."""

    groups: dict[str, list[float]] = defaultdict(list)
    for (cell, request, rep), value in latency.items():
        if cell == a and (b, request, rep) in latency:
            groups[request].append(latency[(b, request, rep)] - value)
    return groups


def _ratio_groups(
    latency: dict[tuple[str, str, int], float], a: str, b: str
) -> dict[str, list[tuple[float, float]]]:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (cell, request, rep), value in latency.items():
        if cell == a and (b, request, rep) in latency:
            groups[request].append((value, latency[(b, request, rep)]))
    return groups


def _ratio_of_medians(pairs: list[tuple[float, float]]) -> float:
    return median(b for _, b in pairs) / median(a for a, _ in pairs)


def _interaction_groups(
    latency: dict[tuple[str, str, int], float], base: str
) -> dict[str, list[float]]:
    """(L1C2 - L1C0) - (L0C2 - L0C0) per request and repetition."""

    cells = {
        "00": f"{base}-L0",
        "02": f"{base}-L0-C2",
        "10": f"{base}-L1",
        "12": f"{base}-L1-C2",
    }
    groups: dict[str, list[float]] = defaultdict(list)
    for (cell, request, rep), value in latency.items():
        if cell != cells["00"]:
            continue
        keys = {k: (c, request, rep) for k, c in cells.items()}
        if all(k in latency for k in keys.values()):
            groups[request].append(
                (latency[keys["12"]] - latency[keys["10"]])
                - (latency[keys["02"]] - value)
            )
    return groups


def _ci(values: tuple[float, float, float], digits: int = 1) -> dict[str, float]:
    point, low, high = values
    return {
        "estimate": round(point, digits),
        "ci_low": round(low, digits),
        "ci_high": round(high, digits),
    }


def _range(values: Sequence[float], digits: int = 1) -> dict[str, float]:
    return {
        "min": round(min(values), digits),
        "median": round(median(values), digits),
        "max": round(max(values), digits),
    }


# --------------------------------------------------------------------------
# Tables


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def crossover(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell in summary["cells"]:
        condition, edge_load, cloud_load, task = _cell_parts(cell["cell_id"])
        medians = {
            SHORT_NAMES[c]: round(s["median_ms"], 1)
            for c, s in cell["placements"].items()
        }
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "condition": condition,
                "rate_mbps": cell["rate_mbps"],
                "added_rtt_ms": cell["added_rtt_ms"],
                "edge_load": edge_load,
                "cloud_load": cloud_load,
                "task": task,
                "winner": SHORT_NAMES[cell["winner"]],
                "runner_up": SHORT_NAMES[cell["runner_up"]],
                "margin_ms": round(cell["margin_ms"], 1),
                "tie": cell["tie"],
                **{f"{name}_median_ms": v for name, v in medians.items()},
            }
        )
    return rows


def load_effects(
    attempts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Paired cloud-load and edge-load effects per condition and configuration."""

    short = [a for a in attempts if a["task"] == "short"]
    conditions = sorted({a["condition_id"] for a in short})
    rows = []
    for configuration in CONFIGURATIONS:
        latency = {
            (a["cell_id"], a["request_id"], a["repetition"]): a["latency_ms"][
                configuration
            ]
            for a in short
            if a["latency_ms"][configuration] is not None
        }
        for condition in conditions:
            row: dict[str, Any] = {
                "configuration": SHORT_NAMES[configuration],
                "condition": condition,
            }
            for level in ("L0", "L1"):
                base = f"{condition}-{level}"
                effect = _cluster_bootstrap(
                    _paired(latency, base, f"{base}-C2"), median
                )
                for key, value in _ci(effect).items():
                    row[f"c2_minus_c0_at_{level}_ms_{key}"] = value
            for cloud in ("", "-C2"):
                label = "C2" if cloud else "C0"
                factor = _cluster_bootstrap(
                    _ratio_groups(
                        latency, f"{condition}-L0{cloud}", f"{condition}-L1{cloud}"
                    ),
                    _ratio_of_medians,
                )
                for key, value in _ci(factor, 4).items():
                    row[f"l1_over_l0_at_{label}_{key}"] = value
            interaction = _cluster_bootstrap(
                _interaction_groups(latency, condition), median
            )
            for key, value in _ci(interaction).items():
                row[f"interaction_ms_{key}"] = value
            rows.append(row)

    def pick(configuration: str, column: str) -> list[float]:
        return [r[column] for r in rows if r["configuration"] == configuration]

    def covers_zero(configuration: str) -> int:
        return sum(
            r["interaction_ms_ci_low"] <= 0 <= r["interaction_ms_ci_high"]
            for r in rows
            if r["configuration"] == configuration
        )

    summary = {
        name: {
            "c2_minus_c0_at_L0_ms": _range(pick(name, "c2_minus_c0_at_L0_ms_estimate")),
            "c2_minus_c0_at_L1_ms": _range(pick(name, "c2_minus_c0_at_L1_ms_estimate")),
            "l1_over_l0_at_C0": _range(pick(name, "l1_over_l0_at_C0_estimate"), 3),
            "l1_over_l0_at_C2": _range(pick(name, "l1_over_l0_at_C2_estimate"), 3),
            "interaction_ms": _range(pick(name, "interaction_ms_estimate")),
            "interaction_ci_covers_zero": f"{covers_zero(name)}/{len(conditions)}",
        }
        for name in SHORT_NAMES.values()
    }
    return rows, summary


def _median_metric(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], float]
) -> float:
    return median(key(r) for r in rows)


def split_anatomy(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Why split inference loses: payload, transfer and the compute floor.

    ``network`` is the client's round trip minus the server's own time (wire
    transfer, RTT and framing). ``compute`` is end-to-end minus ``network``:
    what the configuration would take on an ideal link.
    """

    short = [r for r in rows if r["task"] == "short"]
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in short:
        by_cell[(r["cell_id"], r["configuration_id"])].append(r)
    out = []
    for (cell, configuration), members in sorted(by_cell.items()):
        first = members[0]
        m0 = first["metrics"]
        entry = {
            "cell_id": cell,
            "condition": first["condition_id"],
            "rate_mbps": first["rate_mbps"],
            "added_rtt_ms": first["added_rtt_ms"],
            "edge_load": first["load_level"],
            "cloud_load": first["cloud_load"],
            "configuration": SHORT_NAMES[configuration],
            "e2e_ms": round(_median_metric(members, lambda r: r["e2e_ms"]), 1),
        }
        if configuration == EDGE:
            entry.update(
                request_bytes=0,
                network_ms=0.0,
                compute_ms=entry["e2e_ms"],
                edge_vision_ms=round(
                    _median_metric(members, lambda r: r["metrics"]["vision_total_ms"]),
                    1,
                ),
                server_ms=0.0,
            )
        else:
            entry.update(
                request_bytes=int(
                    _median_metric(
                        members, lambda r: r["metrics"]["request_wire_bytes"]
                    )
                ),
                network_ms=round(
                    _median_metric(
                        members,
                        lambda r: (
                            r["metrics"]["network_round_trip_ms"]
                            - r["metrics"]["server_total_ms"]
                        ),
                    ),
                    1,
                ),
                compute_ms=round(
                    _median_metric(
                        members,
                        lambda r: (
                            r["e2e_ms"]
                            - (
                                r["metrics"]["network_round_trip_ms"]
                                - r["metrics"]["server_total_ms"]
                            )
                        ),
                    ),
                    1,
                ),
                edge_vision_ms=round(
                    _median_metric(
                        members, lambda r: r["metrics"].get("vision_total_ms", 0.0)
                    ),
                    1,
                ),
                server_ms=round(
                    _median_metric(members, lambda r: r["metrics"]["server_total_ms"]),
                    1,
                ),
            )
        if "payload_format" in m0:
            payload = _median_metric(
                members, lambda r: r["metrics"]["projected_payload_bytes"]
            )
            entry["effective_mbps"] = round(
                8
                * payload
                / 1000
                / max(entry["network_ms"] - first["added_rtt_ms"], 1e-9),
                1,
            )
        else:
            entry["effective_mbps"] = None
        out.append(entry)

    index = {(r["cell_id"], r["configuration"]): r for r in out}
    cells = sorted({r["cell_id"] for r in out})
    floors: dict[str, dict[str, Any]] = {}
    for edge_load, cloud_load in LOAD_STATES:
        chosen = [
            c
            for c in cells
            if index[(c, "cloud")]["edge_load"] == edge_load
            and index[(c, "cloud")]["cloud_load"] == cloud_load
        ]
        floors[f"{edge_load}/{cloud_load}"] = {
            name: _range([index[(c, name)]["compute_ms"] for c in chosen])
            for name in ("edge", "cloud", "synchronous", "int8")
        } | {
            # Positive: int8 needs more compute than the cloud, and it also
            # sends a larger payload, so no link lets it beat the cloud.
            "int8_minus_cloud_compute_ms": _range(
                [
                    index[(c, "int8")]["compute_ms"] - index[(c, "cloud")]["compute_ms"]
                    for c in chosen
                ]
            ),
            # The edge has no network term: edge minus int8 compute is the most
            # network time (RTT plus transfer) int8 can spend and still beat it.
            "edge_minus_int8_compute_ms": _range(
                [
                    index[(c, "edge")]["compute_ms"] - index[(c, "int8")]["compute_ms"]
                    for c in chosen
                ]
            ),
        }
    image = [index[(c, "cloud")]["request_bytes"] for c in cells]
    summary = {
        "payload_bytes": {
            "cloud_request_median": int(median(image)),
            "synchronous_request": index[(cells[0], "synchronous")]["request_bytes"],
            "int8_request": index[(cells[0], "int8")]["request_bytes"],
        },
        "compute_floor_ms_by_load_state": floors,
    }
    return out, summary


def paired_configuration_difference(
    attempts: list[dict[str, Any]], a: str, b: str
) -> list[dict[str, Any]]:
    """Per short cell: median paired b - a with a cluster-bootstrap CI."""

    by_cell: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        if attempt["task"] != "short":
            continue
        x, y = attempt["latency_ms"][a], attempt["latency_ms"][b]
        if x is None or y is None:
            continue
        by_cell[attempt["cell_id"]][attempt["request_id"]].append(y - x)
        meta[attempt["cell_id"]] = attempt
    rows = []
    for cell, groups in sorted(by_cell.items()):
        estimate = _cluster_bootstrap(groups, median)
        _, edge_load, cloud_load, _ = _cell_parts(cell)
        rows.append(
            {
                "cell_id": cell,
                "rate_mbps": meta[cell]["rate_mbps"],
                "added_rtt_ms": meta[cell]["added_rtt_ms"],
                "edge_load": edge_load,
                "cloud_load": cloud_load,
                **{f"difference_ms_{k}": v for k, v in _ci(estimate).items()},
            }
        )
    return rows


def static_policies(
    summary: dict[str, Any], attempts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cell in summary["cells"]:
        best = cell["placements"][cell["winner"]]["median_ms"]
        task = _cell_parts(cell["cell_id"])[3]
        row: dict[str, Any] = {"cell_id": cell["cell_id"], "task": task}
        for configuration in CONFIGURATIONS:
            regret = cell["placements"][configuration]["median_ms"] - best
            row[f"always_{SHORT_NAMES[configuration]}_regret_ms"] = round(regret, 1)
            totals[SHORT_NAMES[configuration]]["regret"].append(regret)
            totals[SHORT_NAMES[configuration]]["relative"].append(regret / best)
        rows.append(row)
    oracle_mean = sum(
        c["placements"][c["winner"]]["median_ms"] for c in summary["cells"]
    ) / len(summary["cells"])
    per_cell = {
        name: {
            "cells_not_best": sum(v > 0 for v in t["regret"]),
            "mean_regret_ms": round(sum(t["regret"]) / len(t["regret"]), 1),
            "max_regret_ms": round(max(t["regret"]), 1),
            "mean_relative_regret_percent": round(
                100 * sum(t["relative"]) / len(t["relative"]), 2
            ),
        }
        for name, t in totals.items()
    }
    per_attempt = {}
    for configuration in CONFIGURATIONS:
        regrets = [
            a["latency_ms"][configuration] - a["latency_ms"][a["winner"]]
            for a in attempts
        ]
        per_attempt[SHORT_NAMES[configuration]] = {
            "attempts_fastest_percent": round(
                100
                * sum(a["winner"] == configuration for a in attempts)
                / len(attempts),
                2,
            ),
            "mean_regret_ms": round(sum(regrets) / len(regrets), 1),
        }
    cell_winner = {c["cell_id"]: c["winner"] for c in summary["cells"]}
    cell_policy = [
        a["latency_ms"][cell_winner[a["cell_id"]]] - a["latency_ms"][a["winner"]]
        for a in attempts
    ]
    return rows, {
        "cells": len(summary["cells"]),
        "attempts": len(attempts),
        "oracle_mean_cell_median_ms": round(oracle_mean, 1),
        "per_cell": per_cell,
        "per_attempt": per_attempt,
        "cell_oracle_vs_attempt_oracle": {
            "mean_regret_ms": round(sum(cell_policy) / len(cell_policy), 1),
            "attempts_not_fastest_percent": round(
                100 * sum(v > 0 for v in cell_policy) / len(cell_policy), 2
            ),
        },
    }


def hardware_comparison(
    rows: list[dict[str, Any]], physical_stage_summary: Path
) -> dict[str, Any]:
    """Stage times on AWS (training L0/C0 short cells) against physical G2."""

    unloaded = [
        r
        for r in rows
        if r["task"] == "short" and r["load_level"] == "L0" and r["cloud_load"] == "C0"
    ]

    def aws(configuration: str, key: str) -> float:
        return round(
            median(
                float(r["metrics"][key])
                for r in unloaded
                if r["configuration_id"] == configuration and key in r["metrics"]
            ),
            1,
        )

    with physical_stage_summary.open(encoding="utf-8") as handle:
        physical = list(csv.DictReader(handle))

    def phys(mode: str, column: str) -> float:
        values = [float(r[column]) for r in physical if r["mode"] == mode]
        return round(sum(values) / len(values), 1)

    pairs = {
        "edge vision encoder, vision-only resident (split path)": (
            phys("distributed", "vision_encoder_ms"),
            aws(SYNC, "vision_encoder_ms"),
        ),
        "edge vision encoder, full model (edge/full)": (
            phys("edge", "vision_encoder_ms"),
            aws(EDGE, "vision_encoder_ms"),
        ),
        "edge image prefill (edge/full)": (
            phys("edge", "image_prefill_ms"),
            aws(EDGE, "image_prefill_ms"),
        ),
        "cloud image encode and prefill": (
            phys("cloud", "cloud_image_encode_and_prefill_ms"),
            aws(CLOUD, "server_model_image_encode_and_prefill_ms"),
        ),
        "cloud prompt prefill": (
            phys("cloud", "prompt_prefill_ms"),
            aws(CLOUD, "server_model_prompt_prefill_ms"),
        ),
    }
    return {
        "note": (
            "Physical values are means over N1-N9 from the physical G2 stage summary "
            "(verbose-answer prompt, swap-free resident edge spilling to shared "
            "memory); AWS values are medians over the training L0/C0 short cells."
        ),
        "stages": {
            name: {
                "physical_ms": p,
                "aws_ms": a,
                "physical_over_aws": round(p / a, 2),
            }
            for name, (p, a) in pairs.items()
        },
    }


def long_task(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in crossover(summary) if r["task"] == "long"]


# --------------------------------------------------------------------------
# Gates (all splits, no latency)


def gate_evidence(campaign: Path) -> dict[str, Any]:
    plan = json.loads((campaign / "campaign_plan.json").read_text())
    settings = plan["settings"]
    limit = settings["edge_dedicated_limit_mib"] - settings["spill_margin_mib"]
    baseline = json.loads((campaign / "network_baseline.json").read_text())[
        "echo_median_rtt_ms"
    ]
    splits: dict[str, dict[str, Any]] = {}
    for subblock in sorted((campaign / "subblocks").iterdir()):
        sub_plan = json.loads((subblock / "plan.json").read_text())
        condition = sub_plan["condition"]
        split = splits.setdefault(
            condition["split"],
            defaultdict(int)
            | {
                "rate_ratio": [],
                "rtt_error_ms": [],
                "l0_peak_edge_memory_mib": [],
                "cloud_heartbeat_age_s": [],
            },
        )
        report = json.loads((subblock / "validation_report.json").read_text())
        split["subblocks"] += 1
        split["valid"] += bool(report["valid"])
        split["restarts"] += report["restarts"]
        for label in ("pre", "post"):
            measurement = json.loads((subblock / f"network_{label}.json").read_text())
            split["probe_failures"] += bool(
                measurement_failures(
                    measurement,
                    rate_mbps=condition["rate_mbps"],
                    added_rtt_ms=condition["added_rtt_ms"],
                    baseline_rtt_ms=baseline,
                )
            )
            for key in ("edge_to_cloud_receiver_bps", "cloud_to_edge_receiver_bps"):
                split["rate_ratio"].append(
                    measurement[key] / (condition["rate_mbps"] * 1e6)
                )
            split["rtt_error_ms"].append(
                measurement["echo_median_rtt_ms"] - baseline - condition["added_rtt_ms"]
            )
        before, after = (
            json.loads((subblock / f"qdisc_{label}.json").read_text())["states"]
            for label in ("before", "after")
        )
        split["netem_drops"] += sum(
            after[h]["dropped"] - before[h]["dropped"] for h in after
        )
        snapshots = []
        for run in sorted(p for p in (subblock / "run").iterdir() if p.is_dir()):
            for row in _jsonl(run / "requests.jsonl"):
                if row["warmup"]:
                    continue
                split["measured"] += 1
                split["failed"] += row["status"] != "ok"
                snapshots.append(row)
                metrics = row.get("metrics") or {}
                if row["status"] == "ok" and row["mode"] != "edge":
                    alive = metrics.get("server_state_load_alive") is True
                    if sub_plan["cloud_load"] == "C0":
                        split["c0_responses"] += 1
                        split["c0_responses_with_cloud_load_alive"] += alive
                    else:
                        split["c2_responses"] += 1
                        split["c2_responses_alive"] += alive
                        split["cloud_heartbeat_age_s"].append(
                            metrics["server_state_load_heartbeat_age_s"]
                        )
        snapshots.sort(key=lambda r: r["interleave_position"])
        states = [r.get("resource_state") or {} for r in snapshots]
        if sub_plan["load_level"] == "L0":
            split["l0_subblocks"] += 1
            split["l0_oom"] += sum(r["status"] == "oom" for r in snapshots)
            split["l0_peak_edge_memory_mib"].append(
                max(s["edge_gpu_memory_used_mib"] for s in states)
            )
        else:
            split["l1_subblocks"] += 1
            split["l1_alive_first_and_last"] += bool(
                states[0].get("edge_load_alive") and states[-1].get("edge_load_alive")
            )
            split["l1_snapshots"] += len(states)
            split["l1_snapshots_alive"] += sum(
                s.get("edge_load_alive") is True for s in states
            )
    out = {
        "limits": {
            "rate_tolerance": "receiver rate within 20% of the target, both directions",
            "rtt_tolerance": (
                "echo RTT within max(5 ms, 20% of added) of baseline + added"
            ),
            "baseline_echo_rtt_ms": round(baseline, 3),
            "l0_edge_memory_limit_mib": limit,
            "cloud_heartbeat_max_age_s": 5.0,
            "max_failure_rate": settings["max_failure_rate"],
        },
        "splits": {},
    }
    for name, split in sorted(splits.items()):
        result = {k: v for k, v in split.items() if not isinstance(v, list)}
        result["rate_ratio"] = _range(split["rate_ratio"], 3)
        result["rtt_error_ms"] = _range(split["rtt_error_ms"], 2)
        result["l0_peak_edge_memory_mib"] = max(split["l0_peak_edge_memory_mib"])
        result["l0_min_headroom_mib"] = limit - result["l0_peak_edge_memory_mib"]
        result["max_cloud_heartbeat_age_s"] = round(
            max(split["cloud_heartbeat_age_s"]), 2
        )
        out["splits"][name] = result
    return out


# --------------------------------------------------------------------------
# Figures (static SVG, light and dark through prefers-color-scheme)

COLORS = {
    # Reference categorical palette, adjacent pairs validated: light / dark.
    "cloud": ("#2a78d6", "#3987e5"),
    "edge": ("#eb6834", "#d95926"),
    "synchronous": ("#1baf7a", "#199e70"),
    "int8": ("#eda100", "#c98500"),
}
STYLE = """
<style>
  svg { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .surface { fill: #fcfcfb; }
  .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .muted { fill: #898781; }
  .on-fill { fill: #ffffff; }
  .grid { stroke: #e1e0d9; stroke-width: 1; }
  .axis { stroke: #c3c2b7; stroke-width: 1; }
  .ring { stroke: #fcfcfb; }
%(series_light)s
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; }
    .grid { stroke: #2c2c2a; } .axis { stroke: #383835; }
    .ring { stroke: #1a1a19; }
%(series_dark)s
  }
</style>
"""


def _style() -> str:
    def rules(index: int, indent: str) -> str:
        return "\n".join(
            f"{indent}.f-{n} {{ fill: {c[index]}; }} .k-{n} {{ stroke: {c[index]}; }}"
            for n, c in COLORS.items()
        )

    return STYLE % {"series_light": rules(0, "  "), "series_dark": rules(1, "    ")}


def _text(x: float, y: float, text: str, cls: str = "ink2", **attrs: Any) -> str:
    extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" {extra}>{text}</text>'


def _open_svg(width: int, height: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{title}</title>',
        f'<desc id="desc">{desc}</desc>',
        _style(),
        f'<rect class="surface" width="{width}" height="{height}"/>',
    ]


def crossover_svg(rows: list[dict[str, Any]]) -> str:
    """Tile grid: training conditions by load state, filled by the winner."""

    short = {
        (r["condition"], r["edge_load"], r["cloud_load"]): r
        for r in rows
        if r["task"] == "short"
    }
    conditions = sorted(
        {(r["added_rtt_ms"], r["rate_mbps"], r["condition"]) for r in short.values()}
    )
    headers = {
        ("L0", "C0"): ("L0 / C0", "no load"),
        ("L0", "C2"): ("L0 / C2", "heavy cloud load"),
        ("L1", "C0"): ("L1 / C0", "edge load"),
        ("L1", "C2"): ("L1 / C2", "both loads"),
    }
    left, top, tile_w, tile_h, gap, group_gap = 190, 112, 150, 30, 2, 8
    groups = len({c[0] for c in conditions})
    width = left + 4 * (tile_w + gap) + 30
    height = top + len(conditions) * (tile_h + gap) + (groups - 1) * group_gap + 86
    parts = _open_svg(
        width,
        height,
        "Measured winner per training cell",
        "Tile grid of the twelve training conditions (rows, by added RTT then "
        "rate) against the four load states (columns). Each tile names the "
        "configuration with the lowest median latency and its margin over the "
        "runner-up; outlined tiles are ties.",
    )
    parts += [
        _text(
            24,
            32,
            "Measured winner per training cell (short task)",
            "ink",
            font_size=16,
            font_weight=600,
        ),
        _text(
            24,
            52,
            "Fastest configuration by median latency, and its margin over "
            "the runner-up. No split configuration wins a cell.",
            "ink2",
            font_size=12,
        ),
    ]
    for column, state in enumerate(LOAD_STATES):
        x = left + column * (tile_w + gap) + tile_w / 2
        parts.append(
            _text(
                x,
                top - 26,
                headers[state][0],
                "ink",
                font_size=13,
                font_weight=600,
                text_anchor="middle",
            )
        )
        parts.append(
            _text(
                x,
                top - 10,
                headers[state][1],
                "muted",
                font_size=11,
                text_anchor="middle",
            )
        )
    parts.append(
        _text(24, top - 10, "Condition, rate, added RTT", "muted", font_size=11)
    )
    y = top
    previous_rtt = None
    for rtt, rate, condition in conditions:
        if previous_rtt is not None and rtt != previous_rtt:
            y += group_gap
        previous_rtt = rtt
        parts.append(_text(24, y + 20, condition, "ink", font_size=12, font_weight=600))
        parts.append(
            _text(64, y + 20, f"{rate:g} Mb/s · {rtt:g} ms", "ink2", font_size=12)
        )
        for column, (edge_load, cloud_load) in enumerate(LOAD_STATES):
            r = short[(condition, edge_load, cloud_load)]
            x = left + column * (tile_w + gap)
            winner = r["winner"]
            tip = (
                f"{r['cell_id']}: {winner} fastest by {r['margin_ms']:.0f} ms over "
                f"{r['runner_up']}{' (CIs overlap: tie)' if r['tie'] else ''}"
            )
            if r["tie"]:
                tile = (
                    f'<rect x="{x + 1}" y="{y + 1}" width="{tile_w - 2}" '
                    f'height="{tile_h - 2}" rx="4" class="surface k-{winner}" '
                    'stroke-width="2"/>'
                )
                label = _text(
                    x + tile_w / 2,
                    y + 20,
                    f"tie · {winner} by {r['margin_ms']:.0f} ms",
                    "ink",
                    font_size=12,
                    text_anchor="middle",
                )
            else:
                tile = (
                    f'<rect x="{x}" y="{y}" width="{tile_w}" height="{tile_h}" '
                    f'rx="4" class="f-{winner}"/>'
                )
                label = _text(
                    x + tile_w / 2,
                    y + 20,
                    f"{winner} · {r['margin_ms']:.0f} ms",
                    "on-fill",
                    font_size=12,
                    font_weight=600,
                    text_anchor="middle",
                )
            parts.append(f"<g><title>{tip}</title>{tile}{label}</g>")
        y += tile_h + gap
    legend_y = y + 34
    items = (
        ("f-cloud", "cloud/raw_image fastest", None),
        ("f-edge", "edge/full fastest", None),
        ("surface k-edge", "tie: top two 95% CIs overlap", "2"),
    )
    lx = 24
    for cls, text, stroke in items:
        extra = f' stroke-width="{stroke}"' if stroke else ""
        parts.append(
            f'<rect x="{lx}" y="{legend_y - 12}" width="22" height="14" '
            f'rx="3" class="{cls}"{extra}/>'
        )
        parts.append(_text(lx + 30, legend_y, text, "ink2", font_size=12))
        lx += 60 + 7 * len(text)
    parts.append(
        _text(
            24,
            legend_y + 24,
            "Margin = runner-up median minus winner "
            "median. Bootstrap CIs: seed 1707, 2,000 resamples.",
            "muted",
            font_size=11,
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def latency_svg(anatomy: list[dict[str, Any]]) -> str:
    """Median latency against rate on the 20 ms RTT sweep, at C0 and C2 (L0)."""

    sweep = ["T01", "T02", "T03", "T04", "T05", "T06"]
    series = ("synchronous", "int8", "edge", "cloud")
    labels = {
        "cloud": "cloud/raw_image",
        "edge": "edge/full",
        "synchronous": "split, bf16",
        "int8": "split, int8",
    }
    width, height = 1030, 470
    panel_w, panel_h = 300, 300
    origins = [(80, 96), (585, 96)]
    y_max = 6.0

    def x_of(rate: float, ox: float) -> float:
        span = math.log(160) - math.log(4.5)
        return ox + panel_w * (math.log(rate) - math.log(4.5)) / span

    def y_of(seconds: float, oy: float) -> float:
        return oy + panel_h - panel_h * seconds / y_max

    parts = _open_svg(
        width,
        height,
        "Median latency against link rate",
        "Training conditions T01 to T06 (added RTT 20 ms) at no edge load, without "
        "and with heavy cloud load: median end-to-end latency of four "
        "configurations against link rate.",
    )
    parts += [
        _text(
            24,
            32,
            "Median end-to-end latency against link rate "
            "(added RTT 20 ms, no edge load)",
            "ink",
            font_size=16,
            font_weight=600,
        ),
        _text(
            24,
            52,
            "Both split configurations stay above cloud/raw_image at every "
            "rate. prompt_first is omitted: it is within 0.4% of the bf16 split.",
            "ink2",
            font_size=12,
        ),
    ]
    index = {
        (r["condition"], r["edge_load"], r["cloud_load"], r["configuration"]): r
        for r in anatomy
    }
    for cloud_load, (ox, oy) in zip(("C0", "C2"), origins, strict=True):
        title = (
            "No cloud load (L0 / C0)"
            if cloud_load == "C0"
            else ("Heavy cloud load (L0 / C2)")
        )
        parts.append(_text(ox, oy - 14, title, "ink", font_size=13, font_weight=600))
        for s in range(0, 7):
            y = y_of(s, oy)
            parts.append(
                f'<line class="{"axis" if s == 0 else "grid"}" x1="{ox}" '
                f'x2="{ox + panel_w}" y1="{y:.1f}" y2="{y:.1f}"/>'
            )
            parts.append(
                _text(ox - 8, y + 4, f"{s}", "muted", font_size=11, text_anchor="end")
            )
        for rate in (5, 15, 25, 45, 80, 150):
            x = x_of(rate, ox)
            parts.append(
                _text(
                    x,
                    oy + panel_h + 18,
                    str(rate),
                    "muted",
                    font_size=11,
                    text_anchor="middle",
                )
            )
        parts.append(
            _text(
                ox + panel_w / 2,
                oy + panel_h + 36,
                "Link rate (Mb/s, log scale)",
                "muted",
                font_size=11,
                text_anchor="middle",
            )
        )
        parts.append(
            _text(
                ox - 42,
                oy + panel_h / 2,
                "Median latency (s)",
                "muted",
                font_size=11,
                text_anchor="middle",
                transform=f"rotate(-90 {ox - 42} {oy + panel_h / 2})",
            )
        )
        ends = []
        for name in series:
            points = []
            for condition in sweep:
                r = index[(condition, "L0", cloud_load, name)]
                points.append(
                    (x_of(r["rate_mbps"], ox), y_of(r["e2e_ms"] / 1000, oy), r)
                )
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            parts.append(
                f'<polyline class="k-{name}" fill="none" stroke-width="2" '
                f'stroke-linejoin="round" points="{path}"/>'
            )
            for x, y, r in points:
                tip = f"{labels[name]}, {r['rate_mbps']:g} Mb/s: {r['e2e_ms']:.0f} ms"
                parts.append(
                    f'<g><title>{tip}</title><circle cx="{x:.1f}" cy="{y:.1f}" '
                    f'r="4.5" class="f-{name} ring" stroke-width="2"/><circle '
                    f'cx="{x:.1f}" cy="{y:.1f}" r="11" fill="transparent"/></g>'
                )
            ends.append([points[-1][1], name, points[-1][2]["e2e_ms"]])
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
        for y, name, value in ends:
            parts.append(
                _text(
                    ox + panel_w + 10,
                    y + 4,
                    f"{labels[name]} {value / 1000:.2f} s",
                    "ink2",
                    font_size=11,
                )
            )
    legend_y = height - 18
    lx = 80
    for name in ("cloud", "edge", "synchronous", "int8"):
        parts.append(
            f'<line class="k-{name}" x1="{lx}" x2="{lx + 22}" '
            f'y1="{legend_y - 4}" y2="{legend_y - 4}" stroke-width="2"/>'
        )
        parts.append(
            f'<circle cx="{lx + 11}" cy="{legend_y - 4}" r="4" class="f-{name}"/>'
        )
        parts.append(_text(lx + 30, legend_y, labels[name], "ink2", font_size=12))
        lx += 60 + 7 * len(labels[name])
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("tables", "gates", "figures", "all"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--oracle",
        type=Path,
        required=True,
        help="training oracle folder (oracle-training)",
    )
    parser.add_argument(
        "--physical-stage-summary",
        type=Path,
        default=Path("results/physical-g2-20261110/analysis/stage_summary.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    steps = ("tables", "gates", "figures") if args.step == "all" else (args.step,)
    summary, attempts = load_oracle(args.oracle)
    rows: list[dict[str, Any]] | None = None
    for step in steps:
        if step == "gates":
            gates = gate_evidence(args.campaign)
            (args.output / "validity_gates.json").write_text(
                json.dumps(gates, indent=1, sort_keys=True) + "\n"
            )
            print(
                "gates",
                {k: (v["valid"], v["subblocks"]) for k, v in gates["splits"].items()},
            )
            continue
        if rows is None:
            rows = training_rows(args.campaign)
        cross = crossover(summary)
        anatomy, anatomy_summary = split_anatomy(rows)
        if step == "figures":
            (args.output / "crossover_map.svg").write_text(crossover_svg(cross))
            (args.output / "latency_vs_rate.svg").write_text(latency_svg(anatomy))
            print("figures written")
            continue
        _write_csv(args.output / "crossover.csv", cross)
        effects, effects_summary = load_effects(attempts)
        _write_csv(args.output / "load_effects.csv", effects)
        _write_csv(args.output / "split_anatomy.csv", anatomy)
        int8_gain = paired_configuration_difference(attempts, INT8, SYNC)
        _write_csv(args.output / "int8_gain.csv", int8_gain)
        first = paired_configuration_difference(attempts, SYNC, FIRST)
        _write_csv(args.output / "prompt_first_vs_synchronous.csv", first)
        policies, policies_summary = static_policies(summary, attempts)
        _write_csv(args.output / "static_policies.csv", policies)
        hardware = hardware_comparison(rows, args.physical_stage_summary)
        unloaded_gain = [
            r for r in int8_gain if r["edge_load"] == "L0" and r["cloud_load"] == "C0"
        ]
        result = {
            "schema_version": 1,
            "splits": ["training"],
            "seed": SEED,
            "bootstrap_resamples": RESAMPLES,
            "oracle_cells_sha256": summary["files"],
            "winners": {
                name: sum(r["winner"] == name for r in cross)
                for name in SHORT_NAMES.values()
            },
            "ties": [r["cell_id"] for r in cross if r["tie"]],
            "load_effects": effects_summary,
            "split_inference": anatomy_summary
            | {
                "int8_gain_ms_at_L0_C0_by_rate": {
                    f"{r['cell_id']} ({r['rate_mbps']:g} Mb/s, "
                    f"{r['added_rtt_ms']:g} ms)": r["difference_ms_estimate"]
                    for r in sorted(
                        unloaded_gain, key=lambda r: (r["rate_mbps"], r["added_rtt_ms"])
                    )
                },
                "prompt_first_minus_synchronous_ms": {
                    "cell_medians": _range(
                        [r["difference_ms_estimate"] for r in first], 1
                    ),
                    "cis_covering_zero": sum(
                        r["difference_ms_ci_low"] <= 0 <= r["difference_ms_ci_high"]
                        for r in first
                    ),
                    "cells": len(first),
                },
            },
            "static_policies": policies_summary,
            "long_task": long_task(summary),
            "hardware": hardware,
        }
        (args.output / "baseline_summary.json").write_text(
            json.dumps(result, indent=1, sort_keys=True) + "\n"
        )
        print(json.dumps({k: result[k] for k in ("winners", "ties")}))


if __name__ == "__main__":
    main()
