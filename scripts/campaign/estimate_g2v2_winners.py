"""Predict the winning placement of every G2 v2 cell before the campaign.

Inputs, all measured:

* G2 (physical, 9 blocks) per-stage means: cloud server time split into image
  encode + prefill, prompt prefill and decode; cloud raw-image and distributed
  tensor transport residuals at nine network settings; the distributed cloud
  half (prefill of projected tokens, prompt prefill, decode).
* Prompt calibration: P0 (G2's prompt) gave 12.28 mean output tokens, P1 1.47,
  so decode stages are scaled by 1.47 / 12.28.
* W8 edge pilot (real model, swap, P1): edge/full and the distributed edge
  half (vision + projector) at L0, L1 and L2, plus the W5 calibration curve
  for intermediate duty cycles.

Transport is fitted as ``a + b / rate_mbps + c * rtt_ms`` separately for the
raw image and the projected tensor. These are model-based estimates; the
on-site pilot measures the real values. Sensitivity rows scale the cloud
server time and the transport times by ±15% / ±20%.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import pairwise, product
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
G2 = ROOT / "results/physical-g2-20261110/analysis"
PILOT = ROOT / "results/g2v2-local-rehearsal-20261116/edge_pilot_summary.json"
CALIBRATION = ROOT / "results/edge-load-calibration-20261116/runs.csv"
LONG_EDGE = (
    ROOT / "results/long-answer-calibration-20261116/edge_decode_under_load.json"
)
LONG_TOKENS = 32.0  # E1 reached the 32-token cap on all 200 calibration questions
GRID = ROOT / "configs/profiles/physical_network_grid_v2.json"
P0_TOKENS, P1_TOKENS = 12.28, 1.47
BASELINE_RTT_MS = 4.0  # G2 N7: 5 ms added, 9.0 ms measured


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _solve3(rows: list[tuple[float, float, float, float]]) -> tuple[float, ...]:
    """Least squares for y = a + b*x1 + c*x2 (normal equations, 3x3)."""

    import numpy as np

    design = np.array([[1.0, x1, x2] for x1, x2, _y, _w in rows])
    target = np.array([y for _x1, _x2, y, _w in rows])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return tuple(float(c) for c in coefficients)


def load_model() -> dict[str, object]:
    stages = {(r["block"], r["mode"]): r for r in _rows(G2 / "stage_summary.csv")}
    network = {r["block"]: r for r in _rows(G2 / "network_conditions.csv")}
    blocks = sorted(network)
    scale = P1_TOKENS / P0_TOKENS

    def f(block: str, mode: str, name: str) -> float:
        return float(stages[(block, mode)][name])

    cloud_server = mean(
        f(b, "cloud", "cloud_image_encode_and_prefill_ms")
        + f(b, "cloud", "prompt_prefill_ms")
        + f(b, "cloud", "decode_ms") * scale
        + f(b, "cloud", "server_other_ms")
        + f(b, "cloud", "prompt_prepare_ms")
        for b in blocks
    )
    cloud_client = mean(f(b, "cloud", "client_or_local_other_ms") for b in blocks)
    samples = []
    for block in blocks:
        rate = float(network[block]["measured_uplink_mbps"])
        rtt = float(network[block]["measured_rtt_ms"])
        samples.append(
            (
                block,
                rate,
                rtt,
                f(block, "cloud", "transport_residual_ms"),
                f(block, "distributed", "transport_residual_ms"),
            )
        )
    image = _solve3([(1 / r, t, y, 1.0) for _b, r, t, y, _d in samples])
    tensor = _solve3([(1 / r, t, d, 1.0) for _b, r, t, _y, d in samples])
    # The distributed cloud half: its projected-token prefill is slower when
    # the slow transfer leaves the cloud GPU idle (clock ramp-down), so keep
    # it as a function of rate, interpolated in log rate.
    prefill = sorted(
        (
            float(network[b]["configured_rate_mbps"]),
            f(b, "distributed", "image_prefill_ms"),
        )
        for b in blocks
        if network[b]["configured_added_rtt_ms"] in ("20", "20.0")
    )
    distributed_fixed = mean(
        f(b, "distributed", "prompt_prefill_ms")
        + f(b, "distributed", "decode_ms") * scale
        + f(b, "distributed", "server_other_ms")
        + f(b, "distributed", "prompt_prepare_ms")
        + f(b, "distributed", "serialization_ms")
        + f(b, "distributed", "deserialization_ms")
        + f(b, "distributed", "client_or_local_other_ms")
        for b in blocks
    )
    pilot = json.loads(PILOT.read_text())
    edge = {
        level: (
            pilot[f"{level}/edge"]["median_ms"],
            pilot[f"{level}/distributed"]["median_ms"],
        )
        for level in ("L0", "L1", "L2")
    }
    return {
        "cloud_server_ms": cloud_server,
        "cloud_client_ms": cloud_client,
        "image_transport": image,
        "tensor_transport": tensor,
        "distributed_prefill_by_rate": prefill,
        "distributed_fixed_ms": distributed_fixed,
        "edge_by_level": edge,
        "decode_scale": scale,
        "cloud_decode_per_token_ms": mean(f(b, "cloud", "decode_ms") for b in blocks)
        / P0_TOKENS,
    }


def _interpolate_log(points: list[tuple[float, float]], rate: float) -> float:
    import math

    if rate <= points[0][0]:
        return points[0][1]
    if rate >= points[-1][0]:
        return points[-1][1]
    for (r0, y0), (r1, y1) in pairwise(points):
        if r0 <= rate <= r1:
            w = (math.log(rate) - math.log(r0)) / (math.log(r1) - math.log(r0))
            return y0 + w * (y1 - y0)
    raise ValueError(rate)


def predict(
    model: dict[str, object],
    rate: float,
    added_rtt: float,
    edge_full: float,
    edge_vision: float,
    *,
    cloud_scale: float = 1.0,
    transport_scale: float = 1.0,
) -> dict[str, float]:
    rtt = BASELINE_RTT_MS + added_rtt
    a, b, c = model["image_transport"]  # type: ignore[misc]
    ta, tb, tc = model["tensor_transport"]  # type: ignore[misc]
    effective = 0.96 * rate  # G2 measured ~96% of the configured rate
    image = max(0.0, a + b / effective + c * rtt) * transport_scale
    tensor = max(0.0, ta + tb / effective + tc * rtt) * transport_scale
    prefill = _interpolate_log(model["distributed_prefill_by_rate"], rate)  # type: ignore[arg-type]
    return {
        "edge": edge_full,
        "cloud": float(model["cloud_server_ms"]) * cloud_scale  # type: ignore[arg-type]
        + float(model["cloud_client_ms"])  # type: ignore[arg-type]
        + image,
        "distributed": edge_vision
        + tensor
        + (prefill + float(model["distributed_fixed_ms"])) * cloud_scale,  # type: ignore[arg-type]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = load_model()
    grid = json.loads(GRID.read_text())
    edge = model["edge_by_level"]  # type: ignore[assignment]
    cells = [(cid, "L0", "short") for cid in grid["conditions"]] + [
        (cid, level, "short")
        for level in grid["load_levels"]
        for cid in grid["loaded_conditions"]
    ]
    for task, spec in grid.get("extra_tasks", {}).items():
        cells += [(cid, level, task) for cid, level in spec["cells"]]
    # Long answers: edge timings measured at L0/L1/L2 with ~16-token answers,
    # extrapolated to 32 tokens with the measured per-token decode cost; the
    # cloud decodes the extra tokens at G2's per-token cost.
    long_edge = json.loads(LONG_EDGE.read_text())
    cloud_per_token = float(model["cloud_decode_per_token_ms"])  # type: ignore[arg-type]
    rows = []
    for (cid, level, task), cloud_scale, transport_scale in product(
        cells, (0.85, 1.0, 1.15), (0.8, 1.0, 1.2)
    ):
        condition = grid["conditions"][cid]
        if task == "short":
            full, vision = edge[level]
            extra_remote = 0.0
        else:
            measured = long_edge[level]
            full = (
                measured["edge_full_ms"]
                + (LONG_TOKENS - measured["mean_tokens"])
                * measured["decode_ms_per_token"]
            )
            vision = measured["vision_half_ms"]
            extra_remote = (LONG_TOKENS - P1_TOKENS) * cloud_per_token * cloud_scale
        times = predict(
            model,
            condition["rate_mbps"],
            condition["added_rtt_ms"],
            full,
            vision,
            cloud_scale=cloud_scale,
            transport_scale=transport_scale,
        )
        times["cloud"] += extra_remote
        times["distributed"] += extra_remote
        ranked = sorted(times.items(), key=lambda kv: kv[1])
        rows.append(
            {
                "cell": f"{cid}-{level}" + ("" if task == "short" else f"-{task}"),
                "task": task,
                "split": condition["split"],
                "rate_mbps": condition["rate_mbps"],
                "added_rtt_ms": condition["added_rtt_ms"],
                "level": level,
                "cloud_scale": cloud_scale,
                "transport_scale": transport_scale,
                **{f"{k}_ms": round(v, 1) for k, v in times.items()},
                "winner": ranked[0][0],
                "margin_ms": round(ranked[1][1] - ranked[0][1], 1),
            }
        )
    # Edge-load sweep: which duty levels (W5 curve) would let distributed win?
    pilot_points = [(edge[level][0], edge[level][1]) for level in ("L0", "L1", "L2")]

    def vision_for(full: float) -> float:
        # Interpolate the edge half against edge/full across the pilot levels.
        pts = sorted(pilot_points)
        if full <= pts[0][0]:
            return pts[0][1] * full / pts[0][0]
        for (f0, v0), (f1, v1) in pairwise(pts):
            if f0 <= full <= f1:
                return v0 + (full - f0) / (f1 - f0) * (v1 - v0)
        f0, v0 = pts[-2]
        f1, v1 = pts[-1]
        return v1 + (full - f1) / (f1 - f0) * (v1 - v0)

    sweep = []
    for full in range(1150, 3001, 50):
        vision = vision_for(full)
        winners = {}
        for cid, condition in grid["conditions"].items():
            times = predict(
                model, condition["rate_mbps"], condition["added_rtt_ms"], full, vision
            )
            winners[cid] = min(times, key=times.__getitem__)
        sweep.append(
            {
                "edge_full_ms": full,
                "edge_vision_ms": round(vision, 1),
                "distributed_wins": sorted(
                    c for c, w in winners.items() if w == "distributed"
                ),
                "cloud_wins": sorted(c for c, w in winners.items() if w == "cloud"),
                "edge_wins": sorted(c for c, w in winners.items() if w == "edge"),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    central = [
        r for r in rows if r["cloud_scale"] == 1.0 and r["transport_scale"] == 1.0
    ]
    summary = {
        "model": {
            k: v for k, v in model.items() if k not in ("distributed_prefill_by_rate",)
        },
        "central_winner_counts": {
            w: sum(r["winner"] == w for r in central)
            for w in ("edge", "cloud", "distributed")
        },
        "winner_counts_by_task_and_level": {
            f"{task}/{level}": {
                w: sum(
                    r["winner"] == w and r["level"] == level and r["task"] == task
                    for r in central
                )
                for w in ("edge", "cloud", "distributed")
            }
            for task in sorted({r["task"] for r in central})
            for level in ("L0", "L1", "L2")
        },
        "winners_by_split": {
            split: sorted({r["winner"] for r in central if r["split"] == split})
            for split in ("training", "validation", "evaluation")
        },
        "robust_cells": sorted(
            cell
            for cell in {r["cell"] for r in rows}
            if len({r["winner"] for r in rows if r["cell"] == cell}) == 1
        ),
        "any_scenario_distributed_wins": sorted(
            {r["cell"] for r in rows if r["winner"] == "distributed"}
        ),
        "central": central,
        "edge_load_sweep": sweep,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "central_winner_counts",
                    "winner_counts_by_task_and_level",
                    "winners_by_split",
                    "any_scenario_distributed_wins",
                )
            },
            indent=2,
        )
    )
    for r in central:
        print(
            r["cell"],
            r["edge_ms"],
            r["cloud_ms"],
            r["distributed_ms"],
            r["winner"],
            r["margin_ms"],
        )
    print("sweep (edge/full ms -> distributed-winning conditions):")
    for s in sweep:
        if s["distributed_wins"] or s["edge_full_ms"] % 250 == 0:
            print(
                s["edge_full_ms"],
                s["edge_vision_ms"],
                "D:",
                s["distributed_wins"],
                "C:",
                len(s["cloud_wins"]),
                "E:",
                len(s["edge_wins"]),
            )


if __name__ == "__main__":
    main()
