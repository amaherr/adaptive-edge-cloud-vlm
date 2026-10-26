"""Deterministic offline network-state replay through the real predictor/selector."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.scheduler.configuration import (
    CONFIGURATIONS,
    CapabilityStatus,
    ConfigurationId,
)
from adaptive_vlm.scheduler.predictor import (
    Calibration,
    PredictionState,
    PredictionTask,
    Predictor,
)
from adaptive_vlm.scheduler.selector import (
    CandidateSnapshot,
    SelectionConstraints,
    predicted_oracle,
    select,
)


def replay_network_trace(
    calibration: Calibration, specification: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if specification.get("schema_version") != 1:
        raise ValueError("unsupported network trace schema")
    if specification.get("calibration_id") != calibration.calibration_id:
        raise ValueError("trace calibration identifier mismatch")
    task = PredictionTask(**specification["task"])
    constraints = SelectionConstraints(**specification.get("constraints", {}))
    predictor = Predictor(calibration)
    samples = specification["samples"]
    if not samples or len({item["sample_id"] for item in samples}) != len(samples):
        raise ValueError("trace requires unique samples")
    records: list[dict[str, Any]] = []
    switches = 0
    noisy_switches = 0
    unsafe: list[str] = []
    expected_mismatches: list[str] = []
    previous: str | None = None
    previous_segment: str | None = None
    for item in samples:
        raw = item["network_state"]
        state = PredictionState(
            host_pair_id=calibration.host_pair_id,
            model_contract_sha256=calibration.model_contract_sha256,
            placement_profile_sha256="pending-per-candidate",
            network_bucket_id=raw["bucket_id"],
            network_source=raw["source"],
            uplink_bps=raw.get("uplink_bps"),
            downlink_bps=raw.get("downlink_bps"),
            rtt_ms=raw.get("rtt_ms"),
            edge_memory_budget_mib=item.get("edge_memory_budget_mib"),
        )
        monitoring = item["monitoring"]
        max_age = specification["max_probe_age_ms"]
        remote_ready = (
            monitoring["health_ready"] is True
            and monitoring.get("error") is None
            and monitoring["age_ms"] <= max_age
        )
        unhealthy = set(item.get("unhealthy_configuration_ids", []))
        if unhealthy - {spec.id.value for spec in CONFIGURATIONS}:
            raise ValueError("unknown unhealthy configuration")
        snapshots_list: list[CandidateSnapshot] = []
        for spec in CONFIGURATIONS:
            is_edge = spec.id is ConfigurationId.EDGE_FULL
            available = item.get("edge_available", True) if is_edge else remote_ready
            status = CapabilityStatus(
                spec.id,
                available,
                None
                if available
                else "edge_unavailable"
                if is_edge
                else "monitor_unavailable",
            )
            fit = calibration.candidates[spec.id.value]
            prediction = predictor.predict(
                spec,
                replace(state, placement_profile_sha256=fit.placement_profile_sha256),
                task,
            )
            snapshots_list.append(
                CandidateSnapshot(
                    spec.id,
                    status,
                    prediction,
                    healthy=spec.id.value not in unhealthy,
                )
            )
        snapshots = tuple(snapshots_list)
        decision = select(snapshots, constraints)
        selected = decision.selected_configuration_id
        oracle = predicted_oracle(decision)
        if selected != oracle:
            raise AssertionError("selector differs from predicted oracle")
        if item.get("expected_configuration_id") != selected:
            expected_mismatches.append(item["sample_id"])
        if selected is not None:
            candidate = next(
                row for row in decision.candidates if row.configuration_id == selected
            )
            if not candidate.eligible or (
                selected != ConfigurationId.EDGE_FULL.value and not remote_ready
            ):
                unsafe.append(item["sample_id"])
        changed = previous is not None and selected != previous
        if changed:
            switches += 1
            if (
                item["segment"] == previous_segment
                and item["segment"] == specification["noisy_segment_id"]
                and not item.get("warmup", False)
            ):
                noisy_switches += 1
        records.append(
            {
                "schema_version": 1,
                "sample_id": item["sample_id"],
                "segment": item["segment"],
                "warmup": item.get("warmup", False),
                "network_state": raw,
                "monitoring": monitoring,
                "decision": decision.to_dict(),
                "predicted_oracle_configuration_id": oracle,
                "switched_from_previous": changed,
            }
        )
        previous = selected
        previous_segment = item["segment"]
    report = {
        "schema_version": 1,
        "calibration_id": calibration.calibration_id,
        "samples": len(records),
        "switches": switches,
        "noisy_segment_switches_after_warmup": noisy_switches,
        "model_load_cost_ms": 0.0,
        "model_load_cost_basis": "all_candidate_models_already_resident",
        "hysteresis_applied": False,
        "expected_choice_mismatches": expected_mismatches,
        "unsafe_selections": unsafe,
        "synthetic_trace_gate_passed": (
            not expected_mismatches and not unsafe and noisy_switches == 0
        ),
    }
    return report, records


def run_network_trace(
    *, calibration_path: Path, trace_path: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"trace output already exists: {output_dir}")
    calibration = Calibration.load(calibration_path)
    trace_bytes = trace_path.read_bytes()
    trace = json.loads(trace_bytes)
    if not isinstance(trace, dict):
        raise ValueError("trace must be a JSON object")
    report, records = replay_network_trace(calibration, trace)
    report["trace_sha256"] = hashlib.sha256(trace_bytes).hexdigest()
    report["calibration_sha256"] = hashlib.sha256(
        calibration_path.read_bytes()
    ).hexdigest()
    output_dir.mkdir(parents=True)
    write_json_atomic(output_dir / "evaluation.json", report)
    (output_dir / "decisions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    return report
