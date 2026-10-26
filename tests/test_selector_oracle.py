from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from adaptive_vlm.benchmark.oracle import evaluate_oracle, run_oracle_evaluation
from adaptive_vlm.benchmark.predictor_eval import _rank_accuracy
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS, CapabilityStatus
from adaptive_vlm.scheduler.predictor import Prediction
from adaptive_vlm.scheduler.selector import (
    CandidateSnapshot,
    SelectionConstraints,
    predicted_oracle,
    select,
)


def _prediction(configuration_id: str, total_ms: float) -> Prediction:
    return Prediction(
        configuration_id=configuration_id,
        valid=True,
        reason=None,
        coverage="in_distribution",
        calibration_id="fit-1",
        estimated_output_tokens=4.0,
        output_estimate_source="training_median",
        components_ms={"compute_ms": total_ms},
        total_ms=total_ms,
        lower_ms=total_ms,
        upper_ms=total_ms,
        request_payload_bytes=1.0,
        request_wire_bytes=1.0,
        response_wire_bytes=1.0,
        edge_memory_requirement_mib=100.0,
    )


def _snapshots(
    latencies: tuple[float, float, float] = (10, 20, 30),
) -> tuple[CandidateSnapshot, ...]:
    return tuple(
        CandidateSnapshot(
            spec.id,
            CapabilityStatus(spec.id, True),
            _prediction(spec.id.value, latency),
        )
        for spec, latency in zip(CONFIGURATIONS, latencies, strict=True)
    )


def _changed(
    snapshots: tuple[CandidateSnapshot, ...], index: int, **changes: object
) -> tuple[CandidateSnapshot, ...]:
    items = list(snapshots)
    items[index] = replace(items[index], **changes)
    return tuple(items)


@pytest.mark.parametrize(
    ("latencies", "expected"),
    [
        ((10, 20, 30), "edge/full"),
        ((20, 10, 30), "cloud/raw_image"),
        ((20, 20, 10), "distributed/projected/synchronous"),
        ((10, 10, 10), "edge/full"),
    ],
)
def test_predicted_oracle_matches_selector(
    latencies: tuple[float, float, float], expected: str
) -> None:
    decision = select(_snapshots(latencies))
    assert decision.selected_configuration_id == expected
    assert predicted_oracle(decision) == expected
    assert len(decision.to_dict()["candidates"]) == 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_numeric_prediction_excluded(bad: float) -> None:
    items = _snapshots()
    items = _changed(items, 0, prediction=_prediction("edge/full", bad))
    decision = select(items)
    assert decision.selected_configuration_id == "cloud/raw_image"
    assert (
        decision.candidates[0].exclusion_reason == "prediction_nonfinite_or_nonpositive"
    )


def test_missing_profile_health_and_oom_exclusions() -> None:
    items = _snapshots()
    items = _changed(items, 0, prediction=None)
    items = _changed(items, 1, healthy=False)
    decision = select(items, SelectionConstraints(edge_memory_budget_mib=50))
    assert decision.status == "no_feasible_candidate"
    assert [item.exclusion_reason for item in decision.candidates] == [
        "prediction_missing",
        "unhealthy",
        "edge_memory_budget_exceeded",
    ]
    assert predicted_oracle(decision) is None


def test_unavailable_capability_and_invalid_fit() -> None:
    items = _snapshots()
    items = _changed(
        items,
        0,
        capability=CapabilityStatus(CONFIGURATIONS[0].id, False, "policy_not_loaded"),
    )
    items = _changed(
        items,
        1,
        prediction=replace(
            items[1].prediction, valid=False, reason="network_out_of_coverage"
        ),
    )
    decision = select(items)
    assert decision.selected_configuration_id == CONFIGURATIONS[2].id.value
    assert [item.exclusion_reason for item in decision.candidates[:2]] == [
        "policy_not_loaded",
        "network_out_of_coverage",
    ]


def test_sla_and_quality_floor() -> None:
    met = select(_snapshots(), SelectionConstraints(sla_ms=15))
    assert met.status == "selected"
    assert met.candidates[0].predicted_sla_met is True
    unmet = select(_snapshots(), SelectionConstraints(sla_ms=5))
    assert unmet.status == "sla_unmet"
    assert unmet.selected_configuration_id == "edge/full"
    unknown_quality = select(_snapshots(), SelectionConstraints(quality_floor=0.8))
    assert unknown_quality.status == "no_feasible_candidate"
    assert all(
        item.exclusion_reason == "quality_unknown"
        for item in unknown_quality.candidates
    )


def _specification() -> dict[str, object]:
    return {
        "schema_version": 1,
        "calibration_id": "fit-1",
        "evidence_scope": "synthetic",
        "safety_evidence_verified": False,
        "sections": ["request_holdout"],
        "holdout_request_ids": ["a", "b"],
        "sla_ms": 25.0,
        "quality_floor": None,
        "edge_memory_budget_mib": None,
        "availability": {
            spec.id.value: {"available": True, "reason": None, "healthy": True}
            for spec in CONFIGURATIONS
        },
        "seed": 42,
        "bootstrap_resamples": 200,
    }


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for request_id, measured in (("a", (12.0, 11.0, 30.0)), ("b", (10.0, 12.0, 30.0))):
        for spec, prediction, observation in zip(
            CONFIGURATIONS, (10.0, 20.0, 30.0), measured, strict=True
        ):
            rows.append(
                {
                    "section": "request_holdout",
                    "request_id": request_id,
                    "network_bucket_id": "fixed",
                    "task_bucket": "fixture",
                    "condition_block_id": "0",
                    "configuration_id": spec.id.value,
                    "prediction": _prediction(spec.id.value, prediction).to_dict(),
                    "baseline_ms": prediction,
                    "observed_ms": observation,
                    "status": "ok",
                }
            )
    return rows


def test_measured_oracle_is_independent_and_reports_regret() -> None:
    report, decisions = evaluate_oracle(_rows(), _specification())
    assert report["paired_blocks"] == 2
    assert report["predicted_oracle_agreement"]
    assert report["top1_agreement"] == 0.5
    assert report["median_regret_ms"] == 0.5
    assert report["median_regret_fraction"] == pytest.approx(0.5 / 11)
    assert report["median_regret_ms_95pct_bootstrap_ci"] == {"lower": 0.0, "upper": 1.0}
    assert report["small_sample_warning"]
    assert not report["automatic_activation_allowed"]
    assert decisions[0]["measured_oracle_configuration_id"] == "cloud/raw_image"
    assert decisions[0]["decision"]["selected_configuration_id"] == "edge/full"


def test_failed_feasible_observation_never_changes_feasibility() -> None:
    rows = _rows()
    rows[0]["status"] = "error"
    report, decisions = evaluate_oracle(rows, _specification())
    assert report["paired_blocks"] == 1
    assert report["incomplete_blocks"] == 1
    assert decisions[0]["decision"]["selected_configuration_id"] == "edge/full"
    assert (
        decisions[0]["evaluation_exclusion_reason"]
        == "missing_or_failed_feasible_observation"
    )


def test_outcome_derived_failed_prediction_rejected() -> None:
    rows = _rows()
    rows[0]["status"] = "error"
    rows[0]["prediction"] = replace(
        _prediction("edge/full", 10), valid=False, reason="failed_trial"
    ).to_dict()
    with pytest.raises(ValueError, match="outcome-derived"):
        evaluate_oracle(rows, _specification())


def test_repeated_condition_oracle_uncertainty() -> None:
    rows = _rows()
    repeated = [
        dict(row, condition_block_id="1") for row in rows if row["request_id"] == "a"
    ]
    for row in repeated:
        row["observed_ms"] = float(row["observed_ms"]) + 2
    report, _ = evaluate_oracle(rows + repeated, _specification())
    assert report["paired_blocks"] == 3
    assert report["repeated_condition_oracle_uncertainty"][0]["blocks"] == 2


def test_predictor_ranking_counts_repetition_blocks() -> None:
    rows = _rows()
    repeated = [
        dict(row, condition_block_id="1")
        for row in rows
        if row["request_id"] == "a"
    ]
    result = _rank_accuracy(rows + repeated)
    assert result["paired_conditions"] == 3


def test_full_gate_requires_physical_blocks_and_safety_evidence() -> None:
    rows = [
        dict(row, section="network_holdout", condition_block_id=str(block))
        for block in range(10)
        for row in _rows()
        if row["request_id"] == "a"
    ]
    specification = dict(
        _specification(),
        evidence_scope="physical-controlled",
        sections=["network_holdout"],
        holdout_request_ids=["a"],
    )
    report, _ = evaluate_oracle(rows, specification)
    assert not report["full_g4_gate_evaluable"]
    specification["safety_evidence_verified"] = True
    report, _ = evaluate_oracle(rows, specification)
    assert report["full_g4_gate_evaluable"]
    assert report["full_g4_gate_passed"]
    assert not report["automatic_activation_allowed"]


def test_frozen_hash_rejects_changed_predictions(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in _rows()))
    specification = tmp_path / "specification.json"
    specification.write_text(
        json.dumps(dict(_specification(), predictions_sha256="0" * 64))
    )
    with pytest.raises(ValueError, match="SHA-256"):
        run_oracle_evaluation(
            predictions_path=predictions,
            specification_path=specification,
            output_dir=tmp_path / "out",
        )
