"""G5 analysis on a synthetic trace segment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptive_vlm.prediction.r2.g5 import analyse, requests
from adaptive_vlm.prediction.r2.models import fit_component
from tests.prediction.test_r2 import _world

LATENCY = {
    "edge": 1000.0,
    "cloud": 900.0,
    "distributed": 2000.0,
    "distributed_prompt_first": 2001.0,
    "distributed_int8": 1500.0,
}


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _campaign(tmp_path: Path, *, chosen: str, sensed_cloud: str) -> Path:
    directory = tmp_path / "campaign" / "subblocks" / "net-s0-E01-L0-C2-r0"
    directory.mkdir(parents=True)
    (directory / "plan.json").write_text(
        json.dumps(
            {
                "subblock_id": "net-s0-E01-L0-C2-r0",
                "segment": "net-s0",
                "cell_id": "E01-L0-C2",
                "condition": {"condition_id": "E01"},
                "load_level": "L0",
                "cloud_load": "C2",
                "repetition": 0,
            }
        )
    )
    (directory / "network_pre.json").write_text(
        json.dumps({"edge_to_cloud_receiver_bps": 2e7, "echo_median_rtt_ms": 50.0})
    )
    position = 0
    rows: dict[str, list[dict[str, Any]]] = {p: [] for p in [*LATENCY, "adaptive"]}
    for index in range(4):
        trial = f"r0000-{index:06d}"
        for placement, latency in LATENCY.items():
            position += 1
            rows[placement].append(
                {
                    "warmup": False,
                    "trial_id": trial,
                    "request_id": f"q{index}",
                    "status": "ok",
                    "observed_end_to_end_ms": latency + index,
                    "output_text": "yes",
                    "interleave_position": position,
                }
            )
        position += 1
        configuration = {
            "edge": "edge/full",
            "cloud": "cloud/raw_image",
        }[chosen]
        rows["adaptive"].append(
            {
                "warmup": False,
                "trial_id": trial,
                "request_id": f"q{index}",
                "status": "ok",
                "observed_end_to_end_ms": LATENCY[chosen] + index + 5.0,
                "output_text": "yes",
                "interleave_position": position,
                "metrics": {
                    "adaptive_selected_configuration_id": configuration,
                    "adaptive_sensed_edge_load": "L0",
                    "adaptive_sensed_cloud_load": sensed_cloud,
                    "adaptive_overhead_ms": 4.0,
                    "adaptive_sensed_uplink_bps": 2e7,
                    "adaptive_sensed_rtt_ms": 50.0,
                },
            }
        )
    for placement, values in rows.items():
        _write(directory / "run" / placement / "requests.jsonl", values)
    return tmp_path / "campaign"


def _features() -> dict[str, dict[str, Any]]:
    return {
        f"q{i}": {"image_width": 640, "image_height": 480, "image_bytes": 100_000}
        for i in range(4)
    }


def _calibration() -> dict[str, Any]:
    return fit_component(_world(range(3))) | {
        "provenance": {"training_units_sha256": "a" * 64}
    }


def test_g5_analysis(tmp_path: Path) -> None:
    records = requests(_campaign(tmp_path, chosen="cloud", sensed_cloud="C2"))
    assert len(records) == 4 and records[0]["trace"] == "net"
    summary = analyse(
        records,
        calibration=_calibration(),
        features=_features(),
        best_static="edge/full",
        sla_ms=950.0,
    )
    assert summary["complete_requests"] == 4
    assert summary["regret"]["top1_agreement_percent"] == 100.0
    assert summary["regret"]["mean_ms"]["estimate"] == 5.0
    assert summary["adaptive_minus_fixed_ms"]["edge/full"]["ci_high"] < 0
    assert summary["gate"]["pass"] is True
    assert summary["sensing"]["cloud_accuracy_percent"] == 100.0
    assert summary["latency"]["cloud/raw_image"]["sla_success_percent"] == 100.0
    assert summary["switches_within_segments"] == 0
    assert summary["answer_equals_chosen_configuration"] == "4/4"
    assert summary["label_counterfactual"]["mean_regret_ms"] >= 0


def test_g5_gate_fails_when_slower_than_best_static(tmp_path: Path) -> None:
    records = requests(_campaign(tmp_path, chosen="edge", sensed_cloud="C0"))
    summary = analyse(
        records,
        calibration=_calibration(),
        features=_features(),
        best_static="cloud/raw_image",
        sla_ms=950.0,
    )
    assert summary["gate"]["faster_than_best_static"] is False
    assert summary["gate"]["pass"] is False
    assert summary["sensing"]["cloud_accuracy_percent"] == 0.0
