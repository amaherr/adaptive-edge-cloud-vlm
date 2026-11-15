"""Audit live monitoring provenance, fail-closed decisions and overhead."""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.scheduler.configuration import ConfigurationId


def analyze_monitor_run(run_dir: Path, *, output: Path | None = None) -> dict[str, Any]:
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    monitor_config = plan.get("monitor_config")
    if not isinstance(monitor_config, dict) or monitor_config.get("mode") != "live":
        raise ValueError("run plan has no live monitor")
    rows = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    schedule = {trial["trial_id"]: trial for trial in plan["schedule"]}
    if len(rows) != len(schedule) or {row["trial_id"] for row in rows} != set(schedule):
        raise ValueError("monitor run has missing or duplicate trials")
    failures: list[str] = []
    unsafe: list[str] = []
    overhead_ms: list[float] = []
    fractions: list[float] = []
    probe_errors = 0
    passive_updates = 0
    passive_estimate: float | None = None
    selected: list[str] = []
    timestamps = 0
    for row in rows:
        trial_id = row["trial_id"]
        if row.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError(f"monitor row has wrong plan: {trial_id}")
        attempts = row.get("attempts", [])
        if not attempts:
            failures.append(trial_id)
            continue
        observed = row.get("observed_end_to_end_ms")
        monitor_total = row.get("monitor_overhead_ms")
        if (
            not isinstance(observed, (int, float))
            or not isinstance(monitor_total, (int, float))
            or monitor_total < 0
            or monitor_total > observed
            or abs(
                monitor_total
                - sum(
                    attempt.get("monitoring", {}).get("sampling_overhead_ms", 0)
                    for attempt in attempts
                )
            )
            > 0.001
        ):
            failures.append(trial_id)
        for attempt in attempts:
            monitoring = attempt.get("monitoring")
            if not isinstance(monitoring, dict) or not monitoring.get("sampled_at_utc"):
                failures.append(trial_id)
                continue
            timestamps += 1
            probe_errors += monitoring.get("error") is not None
            choice = attempt["decision"]["selected_configuration_id"]
            if choice in {
                ConfigurationId.CLOUD_RAW_IMAGE.value,
                ConfigurationId.DISTRIBUTED_PROJECTED.value,
            } and (
                monitoring.get("health_ready") is not True
                or not monitoring.get("remote_ready")
                or monitoring.get("age_ms", float("inf")) > monitor_config["max_age_ms"]
            ):
                unsafe.append(trial_id)
            budget = monitoring.get("edge_memory_budget_mib")
            required = monitoring.get("edge_required_peak_mib")
            if (
                choice
                in {
                    ConfigurationId.EDGE_FULL.value,
                    ConfigurationId.DISTRIBUTED_PROJECTED.value,
                }
                and budget is not None
                and (required is None or required > budget)
            ):
                unsafe.append(trial_id)
        if row["warmup"]:
            continue
        if row["status"] != "ok":
            failures.append(trial_id)
        else:
            selected.append(row["selected_configuration_id"])
        if (
            isinstance(monitor_total, (int, float))
            and isinstance(observed, (int, float))
            and observed > 0
        ):
            overhead_ms.append(float(monitor_total))
            fractions.append(float(monitor_total / observed))
        passive_updates += row.get("passive_updated") is True
        if row.get("passive_updated") is True:
            wire_bytes = row.get("metrics", {}).get("request_wire_bytes")
            send_ms = row.get("metrics", {}).get("network_send_ms")
            if (
                isinstance(wire_bytes, int)
                and wire_bytes >= monitor_config["passive_min_bytes"]
                and isinstance(send_ms, (int, float))
                and math.isfinite(send_ms)
                and send_ms > 0
            ):
                raw_bps = 8_000 * wire_bytes / send_ms
                bounded = min(
                    monitor_config["passive_max_bps"],
                    max(monitor_config["passive_min_bps"], raw_bps),
                )
                alpha = monitor_config["passive_alpha"]
                passive_estimate = (
                    bounded
                    if passive_estimate is None
                    else alpha * bounded + (1 - alpha) * passive_estimate
                )
    switches = sum(left != right for left, right in pairwise(selected))
    report = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "monitor_config": monitor_config,
        "measured_trials": sum(not row["warmup"] for row in rows),
        "monitor_samples_recorded": timestamps,
        "probe_errors": probe_errors,
        "passive_updates": passive_updates,
        "monitor_overhead_ms_median": median(overhead_ms) if overhead_ms else None,
        "monitor_overhead_ms_max": max(overhead_ms) if overhead_ms else None,
        "monitor_overhead_fraction_median": median(fractions) if fractions else None,
        "switches_between_successful_measured_trials": switches,
        "final_passive_uplink_application_bps": passive_estimate,
        "passive_measurement_scope": "socket_send_duration_not_link_goodput",
        "model_load_cost_ms": 0.0,
        "model_load_cost_basis": "resident_policies_loaded_before_schedule",
        "unsafe_selections": sorted(set(unsafe)),
        "audit_failures": sorted(set(failures)),
        "local_live_monitor_gate_passed": (
            not failures and not unsafe and timestamps >= len(rows)
        ),
    }
    if output is not None:
        write_json_atomic(output, report)
    return report
