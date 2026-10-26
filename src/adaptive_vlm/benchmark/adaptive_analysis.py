"""Compare a mixed-choice run to fixed-policy controls on the same schedule."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.runner import semantic_config_sha256
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        trial_id = row["trial_id"]
        if trial_id in result:
            raise ValueError(f"duplicate trial: {trial_id}")
        result[trial_id] = row
    return result


def analyze_adaptive_run(
    adaptive_dir: Path, fixed_dirs: list[Path], *, output: Path | None = None
) -> dict[str, Any]:
    plan = _json(adaptive_dir / "plan.json")
    if plan.get("mode") != "adaptive":
        raise ValueError("first run must be adaptive")
    fixed: dict[
        str, tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]
    ] = {}
    for directory in fixed_dirs:
        fixed_plan = _json(directory / "plan.json")
        mode = fixed_plan["mode"]
        if mode in fixed or mode not in {spec.mode.value for spec in CONFIGURATIONS}:
            raise ValueError("fixed control modes must be unique and supported")
        fixed[mode] = (
            fixed_plan,
            _rows(directory / "requests.jsonl"),
            _json(directory / "run_manifest.json"),
        )
    if set(fixed) != {spec.mode.value for spec in CONFIGURATIONS}:
        raise ValueError("analysis requires all three fixed-policy controls")
    expected = {entry["trial_id"]: entry for entry in plan["schedule"]}
    adaptive_rows = _rows(adaptive_dir / "requests.jsonl")
    if set(adaptive_rows) != set(expected):
        raise ValueError("adaptive run has missing or unexpected trials")
    config_hash = semantic_config_sha256(plan["config"])
    for mode, (control, rows, _manifest) in fixed.items():
        if (
            control["dataset_manifest_sha256"] != plan["dataset_manifest_sha256"]
            or control["model_contract_sha256"] != plan["model_contract_sha256"]
            or semantic_config_sha256(control["config"]) != config_hash
            or control["schedule"] != plan["schedule"]
            or set(rows) != set(expected)
        ):
            raise ValueError(f"{mode} control does not match adaptive plan")
        if any(
            row.get("plan_sha256") != control["plan_sha256"] for row in rows.values()
        ):
            raise ValueError(f"{mode} control row has wrong plan hash")
    adaptive_manifest = _json(adaptive_dir / "run_manifest.json")
    adaptive_placement = adaptive_manifest["environment"]["execution_placement"]
    selected_counts: Counter[str] = Counter()
    failures = 0
    token_mismatches: list[str] = []
    placement_mismatches: list[str] = []
    missing_controls: list[str] = []
    wire_errors: list[str] = []
    trace_errors: list[str] = []
    for trial_id, trial in expected.items():
        row = adaptive_rows[trial_id]
        if any(
            row.get(name) != value
            for name, value in {
                "request_id": trial["request_id"],
                "warmup": trial["warmup"],
                "repetition": trial["repetition"],
                "plan_sha256": plan["plan_sha256"],
                "config_sha256": plan["config_sha256"],
                "model_contract_sha256": plan["model_contract_sha256"],
                "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
                "calibration_id": plan["calibration_id"],
                "replay_sha256": plan["replay_sha256"],
            }.items()
        ):
            raise ValueError(f"adaptive trial provenance mismatch: {trial_id}")
        attempts = row.get("attempts", [])
        selected = row.get("selected_configuration_id")
        if len(attempts) not in {1, 2} or (len(attempts) == 2 and (
            attempts[0].get("retry_reason") != "connection_failed_before_send"
            or attempts[0].get("status") != "failed"
        )):
            trace_errors.append(trial_id)
        if (
            attempts
            and selected is not None
            and (attempts[-1]["decision"]["selected_configuration_id"] != selected)
        ):
            trace_errors.append(trial_id)
        if row["status"] != "ok":
            failures += not trial["warmup"]
            continue
        if selected is None or not attempts or attempts[-1]["status"] != "ok":
            trace_errors.append(trial_id)
            continue
        if trial["warmup"]:
            continue
        selected_counts[selected] += 1
        spec = next(
            (item for item in CONFIGURATIONS if item.id.value == selected), None
        )
        if spec is None:
            trace_errors.append(trial_id)
            continue
        _control_plan, control_rows, control_manifest = fixed[spec.mode.value]
        control = control_rows[trial_id]
        if control["status"] != "ok":
            missing_controls.append(trial_id)
            continue
        control_placement = control_manifest["environment"]["execution_placement"]
        same_placement = (
            adaptive_placement["vision_device"] == control_placement["vision_device"]
            and adaptive_placement["remote"] == control_placement["remote"]
            and (
                adaptive_placement["text_device"] == control_placement["text_device"]
                if spec.mode.value == "edge"
                else adaptive_placement["cloud_device"]
                == control_placement["text_device"]
            )
        )
        if not same_placement:
            placement_mismatches.append(trial_id)
        elif row["output_token_ids"] != control["output_token_ids"]:
            token_mismatches.append(trial_id)
        metrics = row["metrics"]
        request_bytes = metrics.get("request_wire_bytes", 0)
        response_bytes = metrics.get("response_wire_bytes", 0)
        if spec.mode.value == "edge":
            if request_bytes or response_bytes:
                wire_errors.append(trial_id)
        elif (
            not isinstance(request_bytes, int)
            or not isinstance(response_bytes, int)
            or request_bytes <= 0
            or response_bytes <= 0
            or (
                "total_wire_bytes" in metrics
                and metrics["total_wire_bytes"] != request_bytes + response_bytes
            )
        ):
            wire_errors.append(trial_id)
    measured_count = sum(not trial["warmup"] for trial in expected.values())
    report = {
        "schema_version": 1,
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
        "model_contract_sha256": plan["model_contract_sha256"],
        "adaptive_plan_sha256": plan["plan_sha256"],
        "calibration_id": plan["calibration_id"],
        "activation": plan["activation"],
        "measured_trials": measured_count,
        "measured_failures": failures,
        "selected_counts": dict(selected_counts),
        "same_placement_token_mismatches": token_mismatches,
        "placement_mismatch_trials": placement_mismatches,
        "unavailable_control_trials": missing_controls,
        "wire_accounting_errors": wire_errors,
        "decision_trace_errors": sorted(set(trace_errors)),
        "g5_local_gate_passed": (
            failures == 0
            and sum(selected_counts.values()) == measured_count
            and len(selected_counts) >= 2
            and not token_mismatches
            and not placement_mismatches
            and not missing_controls
            and not wire_errors
            and not trace_errors
        ),
    }
    if output is not None:
        write_json_atomic(output, report)
    return report
