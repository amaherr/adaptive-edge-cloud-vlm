"""Predict the winning configuration of every AWS cell (amendment A2a-5, E5).

Inputs are phase E runs on the AWS pair (``artifacts/phase-e``):

* the unshaped profile ``prof-<task>-<edge level>-<cloud level>``: all five
  configurations on 20 calibration questions, per task (``p1`` short, ``e1``
  long) and load combination;
* shaped references at C0 / L0 on the short task: ``ref-35-20``,
  ``ref-T12`` (25 Mb/s / 150 ms) and ``ref-T06`` (150 Mb/s / 20 ms).

For each configuration, the network adds ``beta * request_bits / rate +
gamma * rtt`` to its unshaped median: a least-squares fit to the three shaped
references (overhead = shaped median - unshaped C0/L0 median). ``edge/full``
has no network term. A cell's estimate is the unshaped median for its task
and loads plus that overhead at the condition's rate and added RTT. These are
estimates; the pilot and the campaign measure the real values.

  python scripts/campaign/estimate_aws_winners.py --output results/<dir>/winners.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts/phase-e"
GRID = ROOT / "configs/profiles/physical_network_grid_v2.json"
CONFIGURATION = {
    "edge": "edge/full",
    "cloud": "cloud/raw_image",
    "distributed": "distributed/projected/synchronous",
    "distributed_prompt_first": "distributed/projected/prompt_first",
    "distributed_int8": "distributed/projected/int8",
}
REFERENCES = {
    "ref-35-20": (35.0, 20.0),
    "ref-T12": (25.0, 150.0),
    "ref-T06": (150.0, 20.0),
}
EDGE_LEVELS = {"L0": "L0", "E-I10": "L1"}
CLOUD_LEVELS = {"C0": "C0", "C50": "C1", "C100": "C2"}


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if not r["warmup"] and r["status"] == "ok"]


def _medians(run: Path) -> dict[str, dict[str, float]]:
    """Per placement: median end-to-end ms and median request wire bytes."""

    out: dict[str, dict[str, float]] = {}
    for placement in CONFIGURATION:
        path = run / placement / "requests.jsonl"
        if not path.exists():
            continue
        rows = _rows(path)
        if not rows:
            continue
        wire = [
            (r.get("metrics") or {}).get("total_request_wire_bytes")
            or (r.get("metrics") or {}).get("request_wire_bytes")
            or 0
            for r in rows
        ]
        out[placement] = {
            "median_ms": statistics.median(r["observed_end_to_end_ms"] for r in rows),
            "request_bytes": float(statistics.median(wire)),
            "n": len(rows),
        }
    return out


def _fit(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Least squares for overhead = beta * x + gamma * rtt (no intercept)."""

    sxx = sum(x * x for x, _, _ in points)
    sxr = sum(x * r for x, r, _ in points)
    srr = sum(r * r for _, r, _ in points)
    sxy = sum(x * y for x, _, y in points)
    sry = sum(r * y for _, r, y in points)
    det = sxx * srr - sxr * sxr
    return (sxy * srr - sry * sxr) / det, (sry * sxx - sxy * sxr) / det


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--grid", type=Path, default=GRID)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = _medians(args.runs / "prof-p1-L0-C0")
    network: dict[str, dict[str, Any]] = {}
    for placement, values in base.items():
        if placement == "edge":
            network[placement] = {"beta": 0.0, "gamma": 0.0, "residuals_ms": []}
            continue
        points = []
        for name, (rate, rtt) in REFERENCES.items():
            shaped = _medians(args.runs / name).get(placement)
            if shaped is None:
                continue
            transfer_ms = shaped["request_bytes"] * 8 / (rate * 1000)
            points.append((transfer_ms, rtt, shaped["median_ms"] - values["median_ms"]))
        beta, gamma = _fit(points)
        network[placement] = {
            "beta": round(beta, 3),
            "gamma": round(gamma, 3),
            "residuals_ms": [round(y - beta * x - gamma * r, 1) for x, r, y in points],
        }

    profile: dict[str, dict[str, dict[str, float]]] = {}
    for run in sorted(args.runs.glob("prof-*")):
        _, task, rest = run.name.split("-", 2)
        edge_level, cloud_level = rest.rsplit("-", 1)
        key = f"{task}/{EDGE_LEVELS[edge_level]}/{CLOUD_LEVELS[cloud_level]}"
        profile[key] = {p: v for p, v in _medians(run).items()}

    conditions = json.loads(args.grid.read_text())["conditions"]
    cells: list[dict[str, Any]] = []
    for key, medians in sorted(profile.items()):
        task, edge_level, cloud_level = key.split("/")
        for cid, condition in conditions.items():
            rate, rtt = float(condition["rate_mbps"]), float(condition["added_rtt_ms"])
            estimate = {}
            for placement, values in medians.items():
                fit = network[placement]
                bytes_ = base.get(placement, values)["request_bytes"]
                estimate[CONFIGURATION[placement]] = round(
                    values["median_ms"]
                    + fit["beta"] * bytes_ * 8 / (rate * 1000)
                    + fit["gamma"] * rtt,
                    1,
                )
            ranked = sorted(estimate, key=estimate.__getitem__)
            cells.append(
                {
                    "condition": cid,
                    "split": condition["split"],
                    "task": task,
                    "edge_load": edge_level,
                    "cloud_load": cloud_level,
                    "winner": ranked[0],
                    "margin_ms": round(estimate[ranked[1]] - estimate[ranked[0]], 1),
                    "winner_lossless": next(
                        c for c in ranked if not c.endswith("int8")
                    ),
                    "estimates_ms": estimate,
                }
            )

    summary: dict[str, Counter[str]] = {}
    for cell in cells:
        label = f"{cell['task']}/{cell['edge_load']}/{cell['cloud_load']}"
        summary.setdefault(label, Counter())[cell["winner"]] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": __doc__.split("\n\n")[1].replace("\n", " "),
                "network_fit": network,
                "profile_medians_ms": {
                    k: {
                        CONFIGURATION[p]: round(v["median_ms"], 1) for p, v in m.items()
                    }
                    for k, m in profile.items()
                },
                "winner_counts": {k: dict(v) for k, v in summary.items()},
                "cells": cells,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for label, counts in summary.items():
        print(f"{label:14} {dict(counts)}")


if __name__ == "__main__":
    main()
