from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.benchmark.calibrate import canonical_sha256
from adaptive_vlm.benchmark.calibrate_v2 import fit_models, nonnegative_fit
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS, CapabilityStatus
from adaptive_vlm.scheduler.predictor import PredictionTask, transfer_ms
from adaptive_vlm.scheduler.predictor_v2 import (
    CalibrationV2,
    PredictorV2,
    state_from_snapshot,
)
from adaptive_vlm.scheduler.selector import CandidateSnapshot, select

ROOT = Path(__file__).resolve().parents[1]


def synthetic_training() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    split = json.loads(
        (ROOT / "configs/evaluation/physical_vqa_split_v1.json").read_text()
    )
    policy = json.loads((ROOT / "configs/predictor/physical_vqa_v2.json").read_text())
    policy["host_pair_id"] = "synthetic-physical-fixture"
    placements = {s.id.value: {"synthetic": s.id.value} for s in CONFIGURATIONS}
    policy["placement_profile_sha256"] = {
        k: canonical_sha256(v) for k, v in placements.items()
    }
    rows = []
    rates = {"N1": 2e6, "N3": 10e6, "N5": 50e6, "N6": 100e6, "N8": 10e6}
    for i, rid in enumerate(split["training_request_ids"]):
        task = {
            "task_bucket": policy["supported_task_bucket"],
            "image_bytes": 100 + i,
            "image_width": 8 + i % 5,
            "image_height": 8 + i % 3,
            "prompt_tokens": 8 + i % 7,
            "requested_max_new_tokens": 32,
        }
        pixels = task["image_width"] * task["image_height"]
        output_tokens = 4 + i % 5
        for block, rate in rates.items():
            state = {
                "network_bucket_id": block,
                "network_source": policy["network_source"],
                "uplink_bps": rate,
                "downlink_bps": rate,
                "rtt_ms": 50 if block == "N8" else 20,
                "probe_age_seconds": 1,
                "probe_scope": "live",
            }
            for spec in CONFIGURATIONS:
                mode = spec.mode.value
                decode = output_tokens * (2 if mode == "edge" else 1)
                edge = (
                    10 + pixels * 0.03 + task["prompt_tokens"] * 0.2 + decode
                    if mode == "edge"
                    else 8 + pixels * 0.01
                    if mode == "distributed"
                    else None
                )
                server = (
                    20 + pixels * 0.03 + task["prompt_tokens"] * 0.2 + decode
                    if mode == "cloud"
                    else 15 + task["prompt_tokens"] * 0.2 + decode
                    if mode == "distributed"
                    else None
                )
                payload = (
                    task["image_bytes"]
                    if mode == "cloud"
                    else 2985984
                    if mode == "distributed"
                    else 0
                )
                wire = payload + 50 if mode != "edge" else 0
                transport = (
                    5
                    + 0.5 * state["rtt_ms"]
                    + transfer_ms(wire, rate)
                    + transfer_ms(120, rate)
                    if mode != "edge"
                    else 0
                )
                exchange = server + transport if server is not None else None
                total = (
                    edge + 1
                    if mode == "edge"
                    else 1 + exchange + 1
                    if mode == "cloud"
                    else edge + 0.5 + exchange + 1
                )
                fields = {
                    "image_bytes": task["image_bytes"],
                    "image_width": task["image_width"],
                    "image_height": task["image_height"],
                    "prompt_tokens": task["prompt_tokens"],
                    "actual_output_tokens": output_tokens,
                    "decode_ms": decode,
                    "edge_total_ms": edge,
                    "server_processing_ms": server,
                    "transport_residual_ms": transport if mode != "edge" else None,
                    "network_exchange_ms": exchange,
                    "serialization_ms": 0.5 if mode == "distributed" else None,
                    "request_payload_bytes": payload,
                    "request_wire_bytes": wire,
                    "response_wire_bytes": 120 if mode != "edge" else 0,
                    "edge_peak_reserved_mib": 1000 if mode != "cloud" else None,
                    "cloud_peak_reserved_mib": 2000 if mode != "edge" else None,
                    "stage_residual_ms": 1,
                }
                for rep in range(3):
                    before = {
                        "request_id": rid,
                        "block": block,
                        "configuration_id": spec.id.value,
                        "repetition": rep,
                        "task": task,
                        "network": state,
                    }
                    target = {
                        "host_pair_id": policy["host_pair_id"],
                        "model_contract_sha256": policy["model_contract_sha256"],
                        "dataset_manifest_sha256": policy["dataset_manifest_sha256"],
                        "request_id": rid,
                        "warmup": False,
                        "status": "ok",
                        "configuration_id": spec.id.value,
                        "placement_profile": placements[spec.id.value],
                        "task_bucket": policy["supported_task_bucket"],
                        "requested_max_new_tokens": 32,
                        "fields": fields,
                        "raw_metrics": {"image_read_ms": 1},
                        "observed_end_to_end_ms": total,
                        "network_state": {"bucket_id": block},
                    }
                    rows.append({"inputs": before, "target": target})
    return rows, split, policy


@pytest.fixture(scope="module")
def fitted() -> tuple[dict[str, CalibrationV2], dict[str, Any]]:
    rows, split, policy = synthetic_training()
    return fit_models(rows, split, policy), rows[0]["inputs"]


def test_nnls_preserves_units_and_nonnegative_coefficients() -> None:
    assert transfer_ms(1_000_000, 8_000_000) == 1000
    assert nonnegative_fit(
        [[1, 0], [1, 1], [1, 2]], [2, 5, 8], alpha=0
    ) == pytest.approx([2, 3])
    assert all(v >= 0 for v in nonnegative_fit([[1, 1], [1, 2]], [-5, -2], alpha=1e-6))


def test_component_sums_roundtrip_and_determinism(fitted: Any, tmp_path: Path) -> None:
    models, before = fitted
    model = models["v2_a"]
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model.to_dict()))
    loaded = CalibrationV2.load(path)
    assert loaded.to_dict() == model.to_dict()
    predictor = PredictorV2(loaded)
    for spec in CONFIGURATIONS:
        snapshot = {**before, "configuration_id": spec.id.value}
        state = state_from_snapshot(snapshot, model.policy)
        task = PredictionTask(**snapshot["task"])
        p = predictor.predict(spec, state, task)
        assert p.valid and p.selection_eligible
        assert p.total_ms == sum(p.components_ms.values())
        assert p.to_dict() == predictor.predict(spec, state, task).to_dict()
    broken = model.to_dict()
    broken["candidates"][CONFIGURATIONS[0].id.value]["decode_ms_per_token"] = 999
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="hash"):
        CalibrationV2.load(path)


@pytest.mark.parametrize(
    "name,value",
    [
        ("host_pair_id", "aws-unapproved"),
        ("tokenizer_sha256", "wrong"),
        ("model_artifact_sha256", "wrong"),
        ("dataset_manifest_sha256", "wrong"),
        ("precision", "fp32"),
        ("placement_profile_sha256", "wrong"),
    ],
)
def test_identity_mismatches_fail_closed(fitted: Any, name: str, value: Any) -> None:
    models, before = fitted
    model = models["v2_a"]
    spec = CONFIGURATIONS[1]
    state = state_from_snapshot(
        {**before, "configuration_id": spec.id.value}, model.policy
    )
    prediction = PredictorV2(model).predict(
        spec, replace(state, **{name: value}), PredictionTask(**before["task"])
    )
    assert not prediction.valid and prediction.reason and prediction.total_ms is None


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"uplink_bps": 0}, "network_state_missing_or_invalid"),
        ({"uplink_bps": float("nan")}, "network_state_missing_or_invalid"),
        ({"probe_age_seconds": 31}, "network_probe_stale"),
        ({"server_healthy": None}, "cloud_unhealthy_or_unknown"),
        ({"available_configurations": ()}, "configuration_unavailable"),
        ({"rtt_ms": 999}, "network_outside_operational_envelope"),
    ],
)
def test_network_and_executor_restrictions(
    fitted: Any, changes: Any, reason: str
) -> None:
    models, before = fitted
    model = models["v2_a"]
    spec = CONFIGURATIONS[1]
    state = state_from_snapshot(
        {**before, "configuration_id": spec.id.value}, model.policy
    )
    p = PredictorV2(model).predict(
        spec, replace(state, **changes), PredictionTask(**before["task"])
    )
    assert not p.valid and p.reason == reason


def test_extrapolation_is_visible_and_not_live_eligible(fitted: Any) -> None:
    models, before = fitted
    model = models["v2_a"]
    spec = CONFIGURATIONS[1]
    state = state_from_snapshot(
        {**before, "configuration_id": spec.id.value}, model.policy
    )
    low = replace(state, rtt_ms=5)
    high = replace(state, rtt_ms=120)  # Synthetic high RTT; no N9 target read.
    predictor = PredictorV2(model)
    for changed in (low, high):
        p = predictor.predict(spec, changed, PredictionTask(**before["task"]))
        assert p.valid and p.coverage == "physics_extrapolation"
        assert not p.selection_eligible and p.support_distance["rtt_ms"] > 0
    old = PredictorV2(models["v1_recalibrated"]).predict(
        spec, low, PredictionTask(**before["task"])
    )
    assert not old.valid and old.reason == "v1_strict_support_exceeded"


@pytest.mark.parametrize(
    "budget,reason",
    [
        (500, "edge_memory_budget_exceeded"),
        (None, "edge_memory_budget_unknown"),
        (10000, "edge_memory_inventory_mismatch"),
    ],
)
def test_memory_exclusion_keeps_diagnostic_but_cannot_execute(
    fitted: Any, budget: Any, reason: str
) -> None:
    models, before = fitted
    model = models["v2_a"]
    spec = CONFIGURATIONS[0]
    state = state_from_snapshot(before, model.policy)
    p = PredictorV2(model).predict(
        spec,
        replace(state, edge_memory_budget_mib=budget),
        PredictionTask(**before["task"]),
    )
    assert not p.valid and p.reason == reason
    assert p.diagnostic_estimate_available and p.total_ms is not None
    assert not p.execution_feasible and not p.selection_eligible


def test_transfer_monotonicity_and_no_edge_network_cost(fitted: Any) -> None:
    models, before = fitted
    model = models["v2_a"]
    predictor = PredictorV2(model)
    task = PredictionTask(**before["task"])
    spec = CONFIGURATIONS[1]
    state = state_from_snapshot(
        {**before, "configuration_id": spec.id.value}, model.policy
    )
    slow = predictor.predict(spec, state, task)
    faster = predictor.predict(
        spec,
        replace(
            state, uplink_bps=state.uplink_bps * 2, downlink_bps=state.downlink_bps * 2
        ),
        task,
    )
    larger = predictor.predict(
        spec, state, replace(task, image_bytes=task.image_bytes * 2)
    )
    assert (
        faster.components_ms["transport_ms"]
        < slow.components_ms["transport_ms"]
        < larger.components_ms["transport_ms"]
    )
    edge = predictor.predict(
        CONFIGURATIONS[0],
        replace(
            state_from_snapshot(before, model.policy),
            uplink_bps=None,
            downlink_bps=None,
            rtt_ms=None,
        ),
        task,
    )
    assert edge.valid and edge.components_ms["transport_ms"] == 0
    split_spec = CONFIGURATIONS[2]
    split_prediction = predictor.predict(
        split_spec,
        state_from_snapshot(
            {**before, "configuration_id": split_spec.id.value}, model.policy
        ),
        task,
    )
    assert split_prediction.request_payload_bytes == 2985984
    assert split_prediction.request_wire_bytes > split_prediction.request_payload_bytes


def test_no_measured_output_features_and_no_validation_fit_rows(fitted: Any) -> None:
    models, before = fitted
    model = models["v2_a"]
    task = PredictionTask(**before["task"])
    state = state_from_snapshot(before, model.policy)
    assert (
        PredictorV2(model)
        .predict(CONFIGURATIONS[0], state, replace(task, expected_output_tokens=8))
        .reason
        == "unapproved_output_length_estimator"
    )
    rows, split, policy = synthetic_training()
    altered = copy.deepcopy(rows)
    altered[0]["inputs"]["block"] = "N2"
    with pytest.raises(ValueError, match="forbidden"):
        fit_models(altered, split, policy)
    altered = copy.deepcopy(rows)
    altered[0]["inputs"]["request_id"] = split["validation_request_ids"][0]
    with pytest.raises(ValueError, match="forbidden"):
        fit_models(altered, split, policy)
    # A sealed final artifact can change arbitrarily: fitting receives neither
    # its path nor its outcomes, and only training digests enter calibration IDs.
    policy["archive_sha256"] = "changed-final-archive-digest"
    again = fit_models(rows, split, policy)
    assert again["v2_a"].calibration_id == model.calibration_id


def test_selector_honors_v2_research_eligibility(fitted: Any) -> None:
    models, before = fitted
    model = models["v2_a"]
    predictor = PredictorV2(model)
    snapshots = []
    for spec in CONFIGURATIONS:
        state = state_from_snapshot(
            {**before, "configuration_id": spec.id.value}, model.policy
        )
        p = predictor.predict(
            spec,
            replace(state, probe_scope="controlled_block_preflight"),
            PredictionTask(**before["task"]),
        )
        snapshots.append(CandidateSnapshot(spec.id, CapabilityStatus(spec.id, True), p))
    assert select(tuple(snapshots)).status == "no_feasible_candidate"


def test_failures_remain_in_schedule_and_do_not_fit_latency() -> None:
    rows, split, policy = synthetic_training()
    rows[0]["target"]["status"] = "oom"
    rows[0]["target"]["observed_end_to_end_ms"] = 1e200
    models = fit_models(rows, split, policy)
    edge = models["v2_a"].candidates["edge/full"]
    assert edge["scheduled_training_rows"] == 900
    assert edge["successful_training_rows"] == 899
    assert edge["training_status_counts"] == {"ok": 899, "oom": 1}
    assert edge["constant_median_ms"] < 1000


def test_unknown_memory_is_not_zero_and_negative_components_stop(fitted: Any) -> None:
    models, before = fitted
    model = models["v2_a"]
    candidates = copy.deepcopy(model.candidates)
    candidates["edge/full"]["edge_memory_requirement_mib"] = None
    changed = CalibrationV2.create(
        variant=model.variant,
        training_rows_sha256=model.training_rows_sha256,
        split_sha256=model.split_sha256,
        policy=model.policy,
        candidates=candidates,
        request_ids=model.training_request_ids,
        block_ids=model.training_block_ids,
    )
    p = PredictorV2(changed).predict(
        CONFIGURATIONS[0],
        state_from_snapshot(before, model.policy),
        PredictionTask(**before["task"]),
    )
    assert not p.valid and p.reason == "edge_memory_requirement_unknown"
    rows, split, policy = synthetic_training()
    rows[0]["target"]["fields"]["edge_total_ms"] = 0.1
    with pytest.raises(ValueError, match="nondecode"):
        fit_models(rows, split, policy)


def test_grouped_cv_folds_cannot_read_held_request_outcomes() -> None:
    rows, split, policy = synthetic_training()
    held = set(sorted(split["training_request_ids"])[::5])
    keep = [r for r in rows if r["inputs"]["request_id"] not in held]
    first = fit_models(keep, split, policy, require_complete=False)
    for row in rows:
        if row["inputs"]["request_id"] in held:
            row["target"]["observed_end_to_end_ms"] = 1e100
            row["target"]["fields"]["actual_output_tokens"] = 32
            row["target"]["fields"]["decode_ms"] = 1e100
    second = fit_models(
        [r for r in rows if r["inputs"]["request_id"] not in held],
        split,
        policy,
        require_complete=False,
    )
    assert first["v2_a"].calibration_id == second["v2_a"].calibration_id
