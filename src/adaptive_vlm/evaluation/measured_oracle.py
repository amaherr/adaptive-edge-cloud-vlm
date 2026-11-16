"""Measured-oracle tables for a G2 v2 campaign (protocol section 9).

Attempt table: for each (sub-block, request), the measured end-to-end latency
of every placement, the fastest successful one and its margin. Cell table: per
(condition, load level) cell and placement, the median, p90 and a bootstrap
95% CI of the median (seed 1707, 2,000 resamples, as in G2), the measured
winner by median and whether the top two CIs overlap (a tie).

Evaluation cells are sealed until the selector is frozen (G4); reading them
needs ``unseal_evaluation=True``.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.prediction.v1.calibration import percentile
from adaptive_vlm.prediction.v2.dataset.campaign_v2 import (
    CONFIGURATION_BY_MODE,
    _campaign_grid,
)
from adaptive_vlm.prediction.v2.dataset.parsing import digest, strict_json

CONFIGURATIONS = tuple(CONFIGURATION_BY_MODE.values())


def _median_interval(
    values: list[float], *, seed: int, resamples: int
) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    samples = sorted(
        median(generator.choices(values, k=len(values))) for _ in range(resamples)
    )
    return percentile(samples, 0.025), percentile(samples, 0.975)


def build_measured_oracle(
    *,
    campaign: Path,
    profiles: Path,
    output: Path,
    splits: tuple[str, ...] = ("training",),
    unseal_evaluation: bool = False,
    seed: int = 1707,
    resamples: int = 2000,
) -> dict[str, Any]:
    if "evaluation" in splits and not unseal_evaluation:
        raise ValueError("evaluation cells are sealed; pass unseal_evaluation")
    if output.exists():
        raise FileExistsError("refusing to overwrite oracle tables")
    _, grid = _campaign_grid(campaign)
    index = strict_json((profiles / "profiles_index.json").read_bytes())
    attempts: list[dict[str, Any]] = []
    for block in grid.execution_order():
        if block.condition.split not in splits:
            continue
        path = profiles / block.subblock_id / "profile" / "profile.jsonl"
        recorded = index["subblocks"].get(block.subblock_id, {}).get("profile_sha256")
        if not path.exists() or digest(path) != recorded:
            raise ValueError(f"profile missing or changed: {block.subblock_id}")
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for line in path.read_bytes().splitlines():
            row = strict_json(line)
            if row["warmup"]:
                continue
            if row["configuration_id"] in grouped[row["request_id"]]:
                raise ValueError("duplicate configuration for one request")
            grouped[row["request_id"]][row["configuration_id"]] = row
        for request_id, rows in sorted(grouped.items()):
            if set(rows) != set(CONFIGURATIONS):
                raise ValueError(f"{block.subblock_id} {request_id} lacks a placement")
            latency = {
                cid: (
                    float(rows[cid]["observed_end_to_end_ms"])
                    if rows[cid]["status"] == "ok"
                    else None
                )
                for cid in CONFIGURATIONS
            }
            ranked = sorted(
                (value, cid) for cid, value in latency.items() if value is not None
            )
            attempts.append(
                {
                    "subblock_id": block.subblock_id,
                    "cell_id": block.cell_id,
                    "condition_id": block.condition.condition_id,
                    "split": block.condition.split,
                    "rate_mbps": block.condition.rate_mbps,
                    "added_rtt_ms": block.condition.added_rtt_ms,
                    "load_level": block.load_level,
                    "repetition": block.repetition,
                    "request_id": request_id,
                    "latency_ms": latency,
                    "status": {cid: rows[cid]["status"] for cid in CONFIGURATIONS},
                    "winner": ranked[0][1] if ranked else None,
                    "margin_ms": (
                        ranked[1][0] - ranked[0][0] if len(ranked) > 1 else None
                    ),
                }
            )

    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        by_cell[attempt["cell_id"]].append(attempt)
    cells = []
    for cell_id, members in sorted(by_cell.items()):
        first = members[0]
        stats = {}
        for cid in CONFIGURATIONS:
            values = sorted(
                a["latency_ms"][cid]
                for a in members
                if a["latency_ms"][cid] is not None
            )
            interval = _median_interval(values, seed=seed, resamples=resamples)
            stats[cid] = {
                "ok": len(values),
                "failed": len(members) - len(values),
                "median_ms": median(values) if values else None,
                "p90_ms": percentile(values, 0.9) if values else None,
                "median_ci_ms": list(interval) if interval else None,
                "attempt_wins": sum(a["winner"] == cid for a in members),
            }
        ranked = sorted(
            (s["median_ms"], cid)
            for cid, s in stats.items()
            if s["median_ms"] is not None
        )
        winner = ranked[0][1] if ranked else None
        runner = ranked[1][1] if len(ranked) > 1 else None
        tie = False
        if winner and runner:
            a, b = stats[winner]["median_ci_ms"], stats[runner]["median_ci_ms"]
            tie = a is None or b is None or a[1] >= b[0]
        cells.append(
            {
                "cell_id": cell_id,
                "condition_id": first["condition_id"],
                "split": first["split"],
                "rate_mbps": first["rate_mbps"],
                "added_rtt_ms": first["added_rtt_ms"],
                "load_level": first["load_level"],
                "attempts": len(members),
                "placements": stats,
                "winner": winner,
                "runner_up": runner,
                "margin_ms": ranked[1][0] - ranked[0][0] if runner else None,
                "tie": tie,
            }
        )
    output.mkdir(parents=True)
    (output / "oracle_attempts.jsonl").write_text(
        "".join(json.dumps(a, sort_keys=True) + "\n" for a in attempts)
    )
    with (output / "oracle_cells.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["cell_id", "split", "rate_mbps", "added_rtt_ms", "load_level", "attempts"]
            + [f"{c}_median_ms" for c in CONFIGURATIONS]
            + ["winner", "margin_ms", "tie"]
        )
        for cell in cells:
            writer.writerow(
                [
                    cell["cell_id"],
                    cell["split"],
                    cell["rate_mbps"],
                    cell["added_rtt_ms"],
                    cell["load_level"],
                    cell["attempts"],
                ]
                + [cell["placements"][c]["median_ms"] for c in CONFIGURATIONS]
                + [cell["winner"], cell["margin_ms"], cell["tie"]]
            )
    summary = {
        "schema_version": 1,
        "splits": list(splits),
        "seed": seed,
        "bootstrap_resamples": resamples,
        "attempts": len(attempts),
        "cells": cells,
        "profiles_index_sha256": digest(profiles / "profiles_index.json"),
        "files": {
            name: digest(output / name)
            for name in ("oracle_attempts.jsonl", "oracle_cells.csv")
        },
    }
    write_json_atomic(output / "oracle_cells.json", summary)
    return summary
