"""Adaptive benchmark command orchestration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from adaptive_vlm.artifacts import create_run_manifest, write_json_atomic
from adaptive_vlm.benchmark.adaptive_runner import (
    load_adaptive_replay,
    run_adaptive_benchmark,
)
from adaptive_vlm.config import (
    ConfigError,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import InferenceRequest, RequestRecord
from adaptive_vlm.data.manifest import RequestItem, load_frozen_manifest
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.environment import collect_environment
from adaptive_vlm.execution.adaptive import load_phase1_policies
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.execution.remote import check_remote_capabilities
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.progress import ProgressReporter
from adaptive_vlm.scheduler.adaptive import RegistrySnapshot
from adaptive_vlm.scheduler.configuration import CapabilityRegistry, ConfigurationId
from adaptive_vlm.scheduler.dispatch import ConfigurationDispatcher
from adaptive_vlm.scheduler.monitor import LiveMonitor, MonitorConfig
from adaptive_vlm.scheduler.predictor import Calibration, PredictionTask
from adaptive_vlm.scheduler.state import SystemState, TaskState


def run_adaptive_benchmark_command(
    args: argparse.Namespace, *, rss_reader: Callable[[], float | None]
) -> int:
    config = load_experiment_config(args.config)
    manifest = load_frozen_manifest(
        args.manifest, expected_revision=config.dataset.revision
    )
    calibration = Calibration.load(args.calibration)
    replay, replay_hash = load_adaptive_replay(
        args.replay,
        manifest=manifest,
        calibration=calibration,
        config=config,
        vision_device=args.vision_device,
        text_device=args.text_device,
        cloud_device=args.cloud_device,
    )
    if (
        replay.get("edge_memory_budget_mib") is not None
        and args.edge_required_peak_mib is None
    ):
        raise ConfigError("replay memory budget requires --edge-required-peak-mib")
    if config.model.adapter != "moondream2":
        raise ConfigError("adaptive benchmark requires the pinned Moondream2 adapter")
    if args.diagnostic_allow_unpassed_gates:
        activation = "diagnostic_unpassed_gates"
    else:
        if args.predictor_evaluation is None or args.selector_evaluation is None:
            raise ConfigError(
                "adaptive benchmark requires both gate reports or the diagnostic flag"
            )
        predictor_gate = json.loads(args.predictor_evaluation.read_text())
        selector_gate = json.loads(args.selector_evaluation.read_text())
        if (
            not predictor_gate.get("full_g3_gate_passed")
            or not selector_gate.get("full_g4_gate_passed")
            or predictor_gate.get("calibration_id") != calibration.calibration_id
            or selector_gate.get("calibration_id") != calibration.calibration_id
        ):
            raise ConfigError("G3 and G4 reports do not approve this calibration")
        activation = "validated_g3_g4"
    progress = ProgressReporter(args.verbose)
    seeded = seed_everything(config.runtime.seed)
    with progress.phase("Loading resident adaptive policies"):
        policies = load_phase1_policies(
            config,
            artifact_manifest=args.artifact_manifest,
            cache_dir=args.cache_dir,
            offline=args.offline,
            vision_device=args.vision_device,
            text_device=args.text_device,
            edge_components="full",
        )
    edge_policy = policies.get(ConfigurationId.EDGE_FULL)
    if not isinstance(edge_policy, LocalExecutionPolicy):
        raise ConfigError("adaptive benchmark needs a resident full edge adapter")
    model = ModelMetadata(
        config.model.model_id,
        config.model.revision,
        model_contract_fingerprint(config.model),
    )
    monitor = None
    if args.monitor_mode == "live":
        if config.cloud.status == "pending":
            raise ConfigError("live monitor requires declared cloud hardware")
        monitor = LiveMonitor(
            config.transport,
            model,
            MonitorConfig(
                sampling_interval_ms=args.probe_interval_ms,
                max_age_ms=args.probe_max_age_ms,
                probe_timeout_ms=args.probe_timeout_ms,
                passive_alpha=args.passive_alpha,
            ),
        )

    def registry_provider() -> CapabilityRegistry | RegistrySnapshot:
        remote = None
        remote_error = None
        monitoring = None
        if monitor is not None:
            sample = monitor.sample()
            remote = sample.remote
            remote_error = sample.error or (
                "remote_unavailable" if remote is None else None
            )
            monitoring = sample.to_dict()
            monitoring["edge_memory_budget_mib"] = replay.get("edge_memory_budget_mib")
            monitoring["edge_required_peak_mib"] = args.edge_required_peak_mib
            monitoring["edge_memory_source"] = (
                "declared_measured_peak"
                if args.edge_required_peak_mib is not None
                else "unknown"
            )
        elif config.cloud.status == "pending":
            remote_error = "cloud_hardware_pending"
        else:
            try:
                remote = check_remote_capabilities(config.transport, model)
            except (OSError, RuntimeError, ValueError) as exc:
                remote_error = f"{type(exc).__name__}: {exc}"
        registry = CapabilityRegistry(
            policies=policies,
            expected_model_fingerprint=model.fingerprint,
            remote=remote,
            remote_error=remote_error,
            edge_resident_memory_mib=rss_reader(),
            edge_required_peak_mib=args.edge_required_peak_mib,
            edge_memory_budget_mib=replay.get("edge_memory_budget_mib"),
        )
        return (
            RegistrySnapshot(registry, monitoring)
            if monitoring is not None
            else registry
        )

    def task_builder(item: RequestItem) -> PredictionTask:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(item.image_path) as image:
            width, height = image.size
        return PredictionTask(
            task_bucket=replay["task_bucket"],
            image_bytes=item.image_path.stat().st_size,
            image_width=width,
            image_height=height,
            prompt_tokens=edge_policy.prompt_token_count(item.prompt),
            requested_max_new_tokens=config.model.max_new_tokens,
        )

    environment = collect_environment()
    environment["seeded_providers"] = list(seeded)
    environment["execution_placement"] = {
        "mode": "adaptive",
        "vision_device": args.vision_device,
        "text_device": args.text_device,
        "cloud_device": args.cloud_device,
        "remote": f"{config.transport.host}:{config.transport.port}",
    }
    environment["adaptive_replay_sha256"] = replay_hash
    environment["calibration_id"] = calibration.calibration_id
    environment["activation"] = activation
    monitor_config = (
        {"mode": "live", **monitor.config.to_dict()} if monitor is not None else None
    )
    if monitor_config is not None:
        environment["monitoring"] = monitor_config
    run_manifest = create_run_manifest(
        config, environment, dataset_manifest_sha256=manifest.sha256
    )
    with progress.phase("Running adaptive benchmark"):
        summary = run_adaptive_benchmark(
            config=config,
            manifest=manifest,
            calibration=calibration,
            replay=replay,
            replay_sha256=replay_hash,
            registry_provider=registry_provider,
            task_builder=task_builder,
            run_manifest=run_manifest,
            run_dir=args.run_dir,
            warmups=args.warmups,
            repetitions=args.repetitions,
            activation=activation,
            resume=args.resume,
            monitor_config=monitor_config,
            passive_observer=monitor.observe_send if monitor is not None else None,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def current_rss_mib() -> float | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return None


def run_adaptive_infer_command(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    if args.edge_memory_budget_mib is not None and args.edge_required_peak_mib is None:
        raise ConfigError(
            "--edge-memory-budget-mib requires --edge-required-peak-mib "
            "from a measured profile"
        )
    if (
        args.expected_output_tokens is not None
        and args.expected_output_tokens > config.model.max_new_tokens
    ):
        raise ConfigError("expected output exceeds pinned max_new_tokens")
    state = SystemState(
        source="manual",
        uplink_bps=(
            args.uplink_mbps * 1_000_000 if args.uplink_mbps is not None else None
        ),
        rtt_ms=args.rtt_ms,
        edge_memory_budget_mib=args.edge_memory_budget_mib,
    )
    seeded = seed_everything(config.runtime.seed)
    progress = ProgressReporter(args.verbose)
    with progress.phase("Loading resident edge policy components"):
        policies = load_phase1_policies(
            config,
            artifact_manifest=args.artifact_manifest,
            cache_dir=args.cache_dir,
            offline=args.offline,
            vision_device=args.vision_device,
            text_device=args.text_device,
            edge_components=args.edge_components,
        )
    model = ModelMetadata(
        config.model.model_id,
        config.model.revision,
        model_contract_fingerprint(config.model),
    )
    remote = None
    remote_error = None
    if config.cloud.status == "pending":
        remote_error = "cloud_hardware_pending"
    else:
        try:
            remote = check_remote_capabilities(config.transport, model)
        except (OSError, RuntimeError, ValueError) as exc:
            remote_error = f"{type(exc).__name__}: {exc}"
    resident_memory = current_rss_mib()
    registry = CapabilityRegistry(
        policies=policies,
        expected_model_fingerprint=model.fingerprint,
        remote=remote,
        remote_error=remote_error,
        edge_resident_memory_mib=resident_memory,
        edge_required_peak_mib=args.edge_required_peak_mib,
        edge_memory_budget_mib=state.edge_memory_budget_mib,
    )
    request = InferenceRequest(
        request_id=args.request_id,
        image_path=args.image.resolve(),
        prompt=args.prompt,
    )
    task = TaskState.from_request(
        request, expected_output_tokens=args.expected_output_tokens
    )
    configuration_id = ConfigurationId(args.configuration_id)
    with progress.phase(f"Running {configuration_id.value} for {request.request_id}"):
        dispatched = ConfigurationDispatcher(registry).infer(
            request,
            configuration_id=configuration_id,
            system_state=state,
            task_state=task,
        )
    environment = collect_environment()
    environment["seeded_providers"] = list(seeded)
    environment["execution_placement"] = {
        "configuration_id": configuration_id.value,
        "edge_components": args.edge_components,
        "vision_device": args.vision_device,
        "text_device": args.text_device,
        "edge_resident_memory_mib": resident_memory,
        "edge_required_peak_mib": args.edge_required_peak_mib,
        "remote": f"{config.transport.host}:{config.transport.port}",
    }
    manifest = create_run_manifest(config, environment)
    record = RequestRecord.from_result(
        run_id=manifest.run_id,
        mode=registry.policy_for(configuration_id).mode,
        result=dispatched.result,
    )
    output = {
        "decision": {
            **dispatched.decision.to_dict(),
            "selection_overhead_ms": dispatched.selection_overhead_ms,
        },
        "request": record.to_dict(),
    }
    if args.output_dir is not None:
        run_directory = args.output_dir / manifest.run_id
        write_json_atomic(run_directory / "run_manifest.json", manifest.to_dict())
        write_json_atomic(run_directory / "decision.json", output["decision"])
        write_json_atomic(run_directory / "request.json", output["request"])
        progress.info(f"Saved adaptive artifacts to {run_directory}")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0
