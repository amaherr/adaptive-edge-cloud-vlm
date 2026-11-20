"""G4 for the r2 selector: selection, oracle, regret and the gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_vlm.prediction.r2.g4 import (
    LOSSLESS,
    best_static,
    decide,
    evaluate_decisions,
    profile_units,
)
from adaptive_vlm.prediction.r2.models import fit_component
from adaptive_vlm.prediction.r2.units import CONFIGURATIONS, Unit
from adaptive_vlm.scheduler.configuration import ALL_CONFIGURATIONS
from adaptive_vlm.scheduler.selector import select
from tests.prediction.test_r2 import CONDITIONS, _world

INT8 = "distributed/projected/int8"


def _fitted(units: list[Unit]) -> dict[str, object]:
    return fit_component(units) | {"provenance": {"training_units_sha256": "x" * 64}}


def test_selector_takes_five_configurations_and_lossless_four() -> None:
    units = _world(range(3))
    calibration = _fitted(units)
    inputs = units[0].inputs
    decision = decide(calibration, inputs)
    assert [c.configuration_id for c in decision.candidates] == list(CONFIGURATIONS)
    assert decision.status == "selected"
    lossless = decide(calibration, inputs, LOSSLESS)
    assert INT8 not in [c.configuration_id for c in lossless.candidates]
    with pytest.raises(ValueError, match="one snapshot per configuration"):
        select((), configurations=ALL_CONFIGURATIONS)


def test_perfect_predictions_have_zero_regret_and_pass() -> None:
    units = _world(range(4))
    calibration = _fitted(units)
    records, summary = evaluate_decisions(
        calibration,
        units,
        best_static_configuration="cloud/raw_image",
        holdout_conditions=list(CONDITIONS),
        fitted_conditions=["X01"],
    )
    selector = summary["policies"]["selector"]
    assert selector["top1_agreement_percent"] == 100.0
    assert selector["max_regret_ms"] == pytest.approx(0.0, abs=1e-6)
    assert summary["complete_blocks"] == 20
    assert summary["gate"]["pass"] is True
    assert summary["safety"]["no_feasible_decisions"] == 0
    always_edge = summary["policies"]["always_edge"]
    assert always_edge["mean_regret_ms"] > 0
    assert summary["selector_minus_policy_ms"]["always_edge"]["estimate"] < 0
    assert len(records) == 20 * 4


def test_gate_needs_a_network_holdout_and_complete_blocks() -> None:
    units = _world(range(2))
    calibration = _fitted(units)
    _, fitted = evaluate_decisions(
        calibration,
        units,
        best_static_configuration="cloud/raw_image",
        holdout_conditions=list(CONDITIONS),
        fitted_conditions=["T01"],
    )
    assert fitted["gate"]["network_holdout"] is False
    assert fitted["gate"]["pass"] is False
    units[0].successes = 1  # one failed repetition makes the unit incomplete
    _, incomplete = evaluate_decisions(
        calibration,
        units,
        best_static_configuration="cloud/raw_image",
        holdout_conditions=list(CONDITIONS),
        fitted_conditions=[],
    )
    assert incomplete["complete_blocks"] == 19
    assert incomplete["complete_request_cells"] == 39


def test_best_static_picks_the_lowest_median() -> None:
    result = best_static(_world(range(3)))
    assert result["configuration"] in CONFIGURATIONS
    assert result["median_ms"][result["configuration"]] == min(
        result["median_ms"].values()
    )


def test_profile_units_reads_one_split(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    profiles = tmp_path / "profiles"
    features = tmp_path / "features"
    features.mkdir()
    task = {"image_width": 640, "image_height": 480, "image_bytes": 1000}
    for name in ("short", "long"):
        (features / f"task_features-{name}.json").write_text(
            json.dumps({"tasks": {"q1": task}})
        )
    for sid, split in (("E01-L0-r0", "evaluation"), ("V01-L0-r0", "validation")):
        directory = campaign / "subblocks" / sid
        directory.mkdir(parents=True)
        (directory / "plan.json").write_text(
            json.dumps(
                {
                    "subblock_id": sid,
                    "cell_id": sid[:6],
                    "condition": {"split": split, "condition_id": sid[:3]},
                    "load_level": "L0",
                    "cloud_load": "C0",
                    "task": "short",
                }
            )
        )
        rows = profiles / sid / "profile"
        rows.mkdir(parents=True)
        row = {
            "warmup": False,
            "status": "ok",
            "request_id": "q1",
            "trial_id": "t1",
            "configuration_id": "cloud/raw_image",
            "edge_load_level": "L0",
            "cloud_load_level": "C0",
            "network_state": {"bucket_id": sid[:3], "uplink_bps": 1e7, "rtt_ms": 20.0},
            "observed_end_to_end_ms": 400.0,
            "fields": {
                "edge_total_ms": None,
                "network_exchange_ms": 399.0,
                "server_processing_ms": 300.0,
                "request_wire_bytes": 1400,
            },
        }
        (rows / "profile.jsonl").write_text(json.dumps(row) + "\n")
    units = profile_units(
        campaign=campaign, profiles=profiles, feature_dir=features, split="evaluation"
    )
    assert [u.inputs.cell_id for u in units] == ["E01-L0"]
    assert units[0].targets["e2e"] == 400.0
