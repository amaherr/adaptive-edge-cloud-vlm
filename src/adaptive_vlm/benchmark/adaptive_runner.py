"""Resumable mixed-configuration benchmark with frozen replay state."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Any

from adaptive_vlm.artifacts import append_jsonl, write_json_atomic
from adaptive_vlm.benchmark.calibrate import canonical_sha256
from adaptive_vlm.benchmark.runner import (
    _quality,
    _read_rows,
    _status,
    make_schedule,
    semantic_config_sha256,
)
from adaptive_vlm.config import ExperimentConfig, model_contract_fingerprint
from adaptive_vlm.contracts import RunManifest
from adaptive_vlm.data.manifest import FrozenManifest, RequestItem
from adaptive_vlm.scheduler.adaptive import RegistrySnapshot, execute_adaptive_request
from adaptive_vlm.scheduler.configuration import (
    CONFIGURATIONS,
    CapabilityRegistry,
    ConfigurationId,
)
from adaptive_vlm.scheduler.predictor import (
    Calibration,
    PredictionState,
    PredictionTask,
    Predictor,
)
from adaptive_vlm.scheduler.selector import SelectionConstraints

TaskBuilder = Callable[[RequestItem], PredictionTask]
RegistryProvider = Callable[[], CapabilityRegistry | RegistrySnapshot]
PassiveObserver = Callable[[Mapping[str, Any]], bool]


def load_adaptive_replay(
    path: Path,
    *,
    manifest: FrozenManifest,
    calibration: Calibration,
    config: ExperimentConfig,
    vision_device: str,
    text_device: str,
    cloud_device: str,
) -> tuple[dict[str, Any], str]:
    content = path.read_bytes()
    replay = json.loads(content)
    if not isinstance(replay, dict) or replay.get("schema_version") != 1:
        raise ValueError("invalid adaptive replay schema")
    expected = {
        "dataset_manifest_sha256": manifest.sha256,
        "calibration_id": calibration.calibration_id,
        "source_profile_sha256": calibration.source_profile_sha256,
        "model_contract_sha256": model_contract_fingerprint(config.model),
        "host_pair_id": calibration.host_pair_id,
    }
    for name, value in expected.items():
        if replay.get(name) != value:
            raise ValueError(f"adaptive replay {name} mismatch")
    placements = replay.get("placement_profiles")
    if not isinstance(placements, dict) or set(placements) != {
        spec.id.value for spec in CONFIGURATIONS
    }:
        raise ValueError("adaptive replay needs all placement profiles")
    for spec in CONFIGURATIONS:
        profile = placements[spec.id.value]
        required = {
            "mode": spec.mode.value,
            "pipeline": "synchronous",
            "remote": f"{config.transport.host}:{config.transport.port}",
            "vision_device": vision_device,
            "text_device": text_device
            if spec.id is ConfigurationId.EDGE_FULL
            else cloud_device,
        }
        if spec.id is ConfigurationId.DISTRIBUTED_PROJECTED:
            required["vision_device"] = vision_device
        if spec.id is ConfigurationId.CLOUD_RAW_IMAGE:
            required["vision_device"] = cloud_device
        if profile != required:
            raise ValueError(f"runtime placement differs for {spec.id.value}")
        fit = calibration.candidates.get(spec.id.value)
        if fit is None or canonical_sha256(profile) != fit.placement_profile_sha256:
            raise ValueError(f"uncalibrated placement for {spec.id.value}")
    if replay.get("task_bucket") not in {
        fit.task_bucket for fit in calibration.candidates.values()
    }:
        raise ValueError("replay task bucket has no calibration")
    task_types = {item.task_type for item in manifest.items}
    if len(task_types) != 1:
        raise ValueError("one adaptive campaign must have one task type")
    if "task_type" in replay and replay["task_type"] not in task_types:
        raise ValueError("replay task type differs from frozen manifest")
    if (
        manifest.source == "vqa-v2-val2014"
        and replay.get("task_type") != "vqa_short_answer"
    ):
        raise ValueError("VQA replay must declare vqa_short_answer task type")
    default = replay.get("default_network_state")
    if not isinstance(default, dict):
        raise ValueError("adaptive replay needs default network state")
    _prediction_state(default, replay, calibration)
    unhealthy = replay.get("unhealthy_by_request", {})
    if not isinstance(unhealthy, dict) or not set(unhealthy) <= {
        item.request_id for item in manifest.items
    }:
        raise ValueError("invalid replay health request IDs")
    ids = {spec.id.value for spec in CONFIGURATIONS}
    if any(
        not isinstance(value, list)
        or len(value) != len(set(value))
        or not set(value) <= ids
        for value in unhealthy.values()
    ):
        raise ValueError("invalid replay unhealthy configuration IDs")
    overrides = replay.get("trial_network_states", {})
    if not isinstance(overrides, dict):
        raise ValueError("invalid trial network states")
    for value in overrides.values():
        _prediction_state(value, replay, calibration)
    SelectionConstraints(
        sla_ms=replay.get("sla_ms"),
        quality_floor=replay.get("quality_floor"),
        edge_memory_budget_mib=replay.get("edge_memory_budget_mib"),
    )
    if replay.get("quality_floor") is not None:
        raise ValueError("Phase 5 has no validated predecision quality scores")
    return replay, hashlib.sha256(content).hexdigest()


def _prediction_state(
    raw: Mapping[str, Any], replay: Mapping[str, Any], calibration: Calibration
) -> PredictionState:
    return PredictionState(
        host_pair_id=calibration.host_pair_id,
        model_contract_sha256=replay["model_contract_sha256"],
        placement_profile_sha256="pending-per-candidate",
        network_bucket_id=raw["bucket_id"],
        network_source=raw["source"],
        uplink_bps=raw.get("uplink_bps"),
        downlink_bps=raw.get("downlink_bps"),
        rtt_ms=raw.get("rtt_ms"),
        edge_memory_budget_mib=replay.get("edge_memory_budget_mib"),
    )


def run_adaptive_benchmark(
    *,
    config: ExperimentConfig,
    manifest: FrozenManifest,
    calibration: Calibration,
    replay: dict[str, Any],
    replay_sha256: str,
    registry_provider: RegistryProvider,
    task_builder: TaskBuilder,
    run_manifest: RunManifest,
    run_dir: Path,
    warmups: int,
    repetitions: int,
    activation: str,
    resume: bool = False,
    monitor_config: Mapping[str, Any] | None = None,
    passive_observer: PassiveObserver | None = None,
) -> dict[str, Any]:
    """Record exactly one terminal row per scheduled trial, including failures."""
    if manifest.revision != config.dataset.revision:
        raise ValueError("request manifest revision differs from config")
    schedule = make_schedule(
        manifest, warmups=warmups, repetitions=repetitions, seed=config.runtime.seed
    )
    unknown = set(replay.get("trial_network_states", {})) - {
        trial.trial_id for trial in schedule
    }
    if unknown:
        raise ValueError(f"replay contains unknown trial IDs: {sorted(unknown)}")
    plan = {
        "schema_version": 1,
        "mode": "adaptive",
        "config": config.to_dict(),
        "config_sha256": semantic_config_sha256(config.to_dict()),
        "model_contract_sha256": model_contract_fingerprint(config.model),
        "dataset_manifest_sha256": manifest.sha256,
        "calibration_id": calibration.calibration_id,
        "replay_sha256": replay_sha256,
        "activation": activation,
        "schedule": [trial.to_dict() for trial in schedule],
    }
    if monitor_config is not None:
        plan["monitor_config"] = dict(monitor_config)
    plan_hash = canonical_sha256(plan)
    plan["plan_sha256"] = plan_hash
    if resume:
        if json.loads((run_dir / "plan.json").read_text()) != plan:
            raise ValueError("cannot resume: adaptive plan differs")
        saved_manifest = json.loads((run_dir / "run_manifest.json").read_text())
        if saved_manifest.get("dataset_manifest_sha256") != manifest.sha256:
            raise ValueError("cannot resume: run manifest differs")
        run_id = saved_manifest["run_id"]
    else:
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        write_json_atomic(run_dir / "plan.json", plan)
        write_json_atomic(run_dir / "run_manifest.json", run_manifest.to_dict())
        run_id = run_manifest.run_id
    raw_path = run_dir / "requests.jsonl"
    rows = _read_rows(raw_path)
    expected = {trial.trial_id: trial for trial in schedule}
    completed: set[str] = set()
    for saved_row in rows:
        saved_trial_id = saved_row.get("trial_id")
        trial = (
            expected.get(saved_trial_id) if isinstance(saved_trial_id, str) else None
        )
        if (
            trial is None
            or trial.trial_id in completed
            or any(
                saved_row.get(name) != value
                for name, value in {
                    "request_id": trial.request_id,
                    "warmup": trial.warmup,
                    "repetition": trial.repetition,
                    "plan_sha256": plan_hash,
                    "run_id": run_id,
                }.items()
            )
        ):
            raise ValueError("saved adaptive trial differs from plan")
        completed.add(trial.trial_id)
    if resume and passive_observer is not None:
        for saved_row in rows:
            if (
                saved_row.get("status") == "ok"
                and saved_row.get("passive_updated") is True
            ):
                passive_observer(saved_row["metrics"])
    items = {item.request_id: item for item in manifest.items}
    predictor = Predictor(calibration)
    placements = {
        spec.id: canonical_sha256(replay["placement_profiles"][spec.id.value])
        for spec in CONFIGURATIONS
    }
    constraints = SelectionConstraints(
        replay.get("sla_ms"),
        replay.get("quality_floor"),
        replay.get("edge_memory_budget_mib"),
    )
    for position, trial in enumerate(schedule, 1):
        if trial.trial_id in completed:
            continue
        item = items[trial.request_id]
        print(
            f"[{position}/{len(schedule)}] adaptive {trial.trial_id} "
            f"{trial.request_id}",
            flush=True,
        )
        started = perf_counter_ns()
        row: dict[str, Any] = {
            **trial.to_dict(),
            "schema_version": 1,
            "started_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "plan_sha256": plan_hash,
            "config_sha256": plan["config_sha256"],
            "model_contract_sha256": plan["model_contract_sha256"],
            "dataset_manifest_sha256": manifest.sha256,
            "calibration_id": calibration.calibration_id,
            "replay_sha256": replay_sha256,
            "run_id": run_id,
            "mode": "adaptive",
            "image_sha256": item.image_sha256,
            "scoring": item.scoring,
            "task_type": item.task_type,
            "activation": activation,
            "attempts": [],
        }
        try:
            task = task_builder(item)
            if (
                task.task_bucket != replay["task_bucket"]
                or task.requested_max_new_tokens != config.model.max_new_tokens
            ):
                raise ValueError("derived task differs from replay/model contract")
            raw_state = replay.get("trial_network_states", {}).get(
                trial.trial_id, replay["default_network_state"]
            )
            state = _prediction_state(raw_state, replay, calibration)
            unhealthy = set(
                replay.get("unhealthy_by_request", {}).get(item.request_id, [])
            )
            healthy = {
                spec.id: spec.id.value not in unhealthy for spec in CONFIGURATIONS
            }
            executed = execute_adaptive_request(
                request=item.to_request(),
                task=task,
                state=state,
                placement_hashes=placements,
                healthy=healthy,
                constraints=constraints,
                predictor=predictor,
                registry_provider=registry_provider,
            )
            row.update(
                task_state=task.__dict__,
                system_state=state.__dict__,
                attempts=list(executed.attempts),
                selected_configuration_id=(
                    executed.selected_configuration_id.value
                    if executed.selected_configuration_id
                    else None
                ),
                selection_overhead_ms=executed.selection_overhead_ms,
                retry_overhead_ms=executed.retry_overhead_ms,
                monitor_overhead_ms=executed.monitor_overhead_ms,
            )
            if executed.result is None:
                error = executed.error or RuntimeError("adaptive request failed")
                raise error
            result = executed.result
            if passive_observer is not None:
                passive_started = perf_counter_ns()
                try:
                    row["passive_updated"] = passive_observer(result.metrics)
                except Exception as exc:
                    row["passive_updated"] = False
                    row["passive_update_error"] = f"{type(exc).__name__}: {exc}"
                row["passive_update_overhead_ms"] = (
                    perf_counter_ns() - passive_started
                ) / 1_000_000
            row.update(
                status="ok",
                output_text=result.output_text,
                output_token_ids=list(result.output_token_ids),
                metrics=dict(result.metrics),
                quality_score=_quality(item, result.output_text),
            )
        except Exception as exc:
            row.update(
                status=_status(exc).value,
                output_text=None,
                output_token_ids=[],
                metrics={},
                quality_score=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        row["observed_end_to_end_ms"] = (perf_counter_ns() - started) / 1_000_000
        append_jsonl(raw_path, row)
        rows.append(row)
    measured = [row for row in rows if not row["warmup"]]
    successful = [row for row in measured if row["status"] == "ok"]
    summary = {
        "schema_version": 1,
        "mode": "adaptive",
        "activation": activation,
        "plan_sha256": plan_hash,
        "dataset_manifest_sha256": manifest.sha256,
        "model_contract_sha256": plan["model_contract_sha256"],
        "calibration_id": calibration.calibration_id,
        "replay_sha256": replay_sha256,
        "warmup_attempts": len(rows) - len(measured),
        "measured_attempts": len(measured),
        "status_counts": dict(Counter(row["status"] for row in measured)),
        "selected_counts": dict(
            Counter(row["selected_configuration_id"] for row in successful)
        ),
        "retry_count": sum(len(row["attempts"]) > 1 for row in measured),
        "monitor_overhead_ms_median": median(
            row["monitor_overhead_ms"]
            for row in measured
            if "monitor_overhead_ms" in row
        )
        if any("monitor_overhead_ms" in row for row in measured)
        else None,
        "passive_updates": sum(row.get("passive_updated") is True for row in measured),
        "selection_overhead_ms_median": median(
            row["selection_overhead_ms"]
            for row in measured
            if "selection_overhead_ms" in row
        )
        if any("selection_overhead_ms" in row for row in measured)
        else None,
        "latency_ms_median_success": median(
            row["observed_end_to_end_ms"] for row in successful
        )
        if successful
        else None,
    }
    write_json_atomic(run_dir / "summary.json", summary)
    return summary
