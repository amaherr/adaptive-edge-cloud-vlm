"""Statistics of the G2 v2+ baseline report, on synthetic additive data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/campaign/report_g2v2plus_baseline.py"
_spec = importlib.util.spec_from_file_location("report_g2v2plus_baseline", SCRIPT)
assert _spec and _spec.loader
report: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

EDGE, CLOUD, SYNC, FIRST, INT8 = report.CONFIGURATIONS


def _latency(
    edge_load: str, cloud_load: str, index: int, extra: float
) -> dict[str, float]:
    edge = 800.0 * (1.4 if edge_load == "L1" else 1.0) + index
    cloud = 300.0 + (600.0 if cloud_load == "C2" else 0.0) + index
    split = 900.0 + (300.0 if cloud_load == "C2" else 0.0) + index
    if edge_load == "L1" and cloud_load == "C2":
        cloud += extra
    return {
        EDGE: edge,
        CLOUD: cloud,
        SYNC: split + 400.0,
        FIRST: split + 401.0,
        INT8: split,
    }


def _attempts(extra: float = 0.0) -> list[dict[str, Any]]:
    attempts = []
    for condition in ("T01", "T02"):
        for edge_load, cloud_load in report.LOAD_STATES:
            suffix = "-C2" if cloud_load == "C2" else ""
            cell = f"{condition}-{edge_load}{suffix}"
            for index in range(4):
                for repetition in (0, 1):
                    latency = _latency(edge_load, cloud_load, index, extra)
                    winner = min(latency, key=latency.__getitem__)
                    attempts.append(
                        {
                            "cell_id": cell,
                            "condition_id": condition,
                            "split": "training",
                            "rate_mbps": 5.0,
                            "added_rtt_ms": 20.0,
                            "load_level": edge_load,
                            "task": "short",
                            "repetition": repetition,
                            "request_id": f"q{index}",
                            "latency_ms": latency,
                            "winner": winner,
                        }
                    )
    return attempts


def test_cell_parts() -> None:
    assert report._cell_parts("T01-L0") == ("T01", "L0", "C0", "short")
    assert report._cell_parts("T12-L1-C2") == ("T12", "L1", "C2", "short")
    assert report._cell_parts("T06-L1-long") == ("T06", "L1", "C0", "long")


def test_cluster_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    groups = {f"q{i}": [float(i), float(i) + 0.5] for i in range(20)}
    first = report._cluster_bootstrap(groups, report.median)
    assert first == report._cluster_bootstrap(groups, report.median)
    point, low, high = first
    assert low <= point <= high
    constant = report._cluster_bootstrap({"a": [3.0], "b": [3.0]}, report.median)
    assert constant == (3.0, 3.0, 3.0)


def test_load_effects_recover_additive_penalty_and_factor() -> None:
    rows, summary = report.load_effects(_attempts())
    cloud = summary["cloud"]
    assert cloud["c2_minus_c0_at_L0_ms"]["median"] == 600.0
    assert cloud["c2_minus_c0_at_L1_ms"]["median"] == 600.0
    assert cloud["l1_over_l0_at_C0"]["median"] == 1.0
    assert cloud["interaction_ms"] == {"min": 0.0, "median": 0.0, "max": 0.0}
    edge = summary["edge"]
    assert edge["c2_minus_c0_at_L0_ms"]["median"] == 0.0
    assert edge["l1_over_l0_at_C0"]["min"] == pytest.approx(1.4, abs=0.002)
    assert len(rows) == 2 * len(report.CONFIGURATIONS)


def test_interaction_is_the_extra_cost_of_combined_load() -> None:
    _, summary = report.load_effects(_attempts(extra=25.0))
    assert summary["cloud"]["interaction_ms"]["median"] == 25.0
    assert summary["cloud"]["interaction_ci_covers_zero"] == "0/2"
    assert summary["edge"]["interaction_ms"]["median"] == 0.0


def test_paired_configuration_difference() -> None:
    rows = report.paired_configuration_difference(_attempts(), SYNC, FIRST)
    assert {r["difference_ms_estimate"] for r in rows} == {1.0}
    assert len(rows) == 8


def _summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    cells = []
    for cell in sorted({a["cell_id"] for a in attempts}):
        members = [a for a in attempts if a["cell_id"] == cell]
        medians = {
            c: report.median(a["latency_ms"][c] for a in members)
            for c in report.CONFIGURATIONS
        }
        ranked = sorted(medians, key=medians.__getitem__)
        cells.append(
            {
                "cell_id": cell,
                "rate_mbps": 5.0,
                "added_rtt_ms": 20.0,
                "load_level": members[0]["load_level"],
                "task": "short",
                "winner": ranked[0],
                "runner_up": ranked[1],
                "margin_ms": medians[ranked[1]] - medians[ranked[0]],
                "tie": False,
                "placements": {c: {"median_ms": m} for c, m in medians.items()},
            }
        )
    return {"splits": ["training"], "cells": cells, "files": {}}


def test_static_policies_and_crossover() -> None:
    attempts = _attempts()
    summary = _summary(attempts)
    rows, totals = report.static_policies(summary, attempts)
    assert totals["per_cell"]["cloud"]["cells_not_best"] == 2  # the L0/C2 cells
    assert totals["per_cell"]["cloud"]["max_regret_ms"] == 100.0
    assert totals["cell_oracle_vs_attempt_oracle"]["mean_regret_ms"] == 0.0
    assert len(rows) == 8
    winners = {r["cell_id"]: r["winner"] for r in report.crossover(summary)}
    assert winners["T01-L0"] == "cloud"
    assert winners["T01-L0-C2"] == "edge"
    assert winners["T01-L1-C2"] == "cloud"


def test_load_oracle_refuses_other_splits(tmp_path: Path) -> None:
    (tmp_path / "oracle_cells.json").write_text(
        json.dumps({"splits": ["training", "validation"], "cells": []})
    )
    (tmp_path / "oracle_attempts.jsonl").write_text("")
    with pytest.raises(ValueError, match="training oracle only"):
        report.load_oracle(tmp_path)
