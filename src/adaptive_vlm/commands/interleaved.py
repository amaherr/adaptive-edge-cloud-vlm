"""Interleaved benchmark command: G2 v2's three placements or G2 v2+'s five."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.edge_load import (
    NO_LOAD,
    LoadGeneratorFailure,
    LoadLevel,
    level_conditions,
    load_levels,
    make_generator,
)
from adaptive_vlm.benchmark.interleaved import (
    ALL_PLACEMENTS,
    DEVICE_RESTART_EXIT_CODE,
    ENVIRONMENT_RESTART_EXIT_CODE,
    PLACEMENTS,
    DeviceContextFailure,
    run_interleaved_benchmark,
)
from adaptive_vlm.config import (
    ConfigError,
    ExecutionMode,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import RunManifest
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.environment import collect_environment, disable_core_dumps
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.cloud import CloudExecutionPolicy
from adaptive_vlm.execution.distributed import DistributedExecutionPolicy
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.fake import FakeModelAdapter, fake_tensor_payload
from adaptive_vlm.process import exit_with_parent, terminate_on_sigterm
from adaptive_vlm.progress import ProgressReporter
from adaptive_vlm.resource_state import (
    Snapshot,
    WindowsGpuCounterSampler,
    combine,
    gpu_state_source,
    host_memory_snapshot,
)
from adaptive_vlm.scheduler.configuration import SPEC_BY_PLACEMENT
from adaptive_vlm.transport.tensors import TensorPayload


def _request_ids(args: argparse.Namespace, manifest_ids: list[str]) -> list[str]:
    if args.split is None:
        if args.split_name is not None:
            raise ConfigError("--split-name requires --split")
        return manifest_ids
    if args.split_name is None:
        raise ConfigError("--split requires --split-name")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    key = f"{args.split_name}_request_ids"
    if key not in split:
        raise ConfigError(f"split file has no {key}")
    return sorted(str(rid) for rid in split[key])


def run_interleaved_benchmark_command(args: argparse.Namespace) -> int:
    progress = ProgressReporter(args.verbose)
    disable_core_dumps()
    # Die with the supervisor, and let SIGTERM run cleanup (stopping the load
    # generator) instead of killing this process outright.
    exit_with_parent()
    terminate_on_sigterm()
    config = load_experiment_config(args.config)
    manifest = load_frozen_manifest(
        args.manifest or config.dataset.manifest_path,
        expected_revision=config.dataset.revision,
    )
    placements = [p for p in ALL_PLACEMENTS if p in args.placements]
    if len(placements) != len(args.placements):
        raise ConfigError("--placements must not repeat a placement")
    if config.model.adapter not in {"fake", "moondream2"}:
        raise ConfigError(f"unsupported model adapter: {config.model.adapter}")
    specs = {p: SPEC_BY_PLACEMENT[p] for p in placements}
    remote = [p for p in placements if specs[p].mode is not ExecutionMode.EDGE]
    if remote and config.cloud.status == "pending":
        raise ConfigError("remote placements require declared cloud hardware")
    request_ids = _request_ids(args, [item.request_id for item in manifest.items])
    repetitions = (
        list(args.repetition_indices)
        if args.repetition_indices
        else list(range(config.runtime.measured_repetitions))
    )
    warmups = (
        args.warmups if args.warmups is not None else config.runtime.warmup_requests
    )

    seeded = seed_everything(config.runtime.seed)
    environment = collect_environment()
    environment["seeded_providers"] = list(seeded)
    edge_uses_gpu = any(spec.mode is not ExecutionMode.CLOUD for spec in specs.values())
    model = ModelMetadata(
        model_id=config.model.model_id,
        revision=config.model.revision,
        fingerprint=model_contract_fingerprint(config.model),
    )

    adapter: Any = None
    serializer = TensorPayload.from_torch
    if edge_uses_gpu:
        if config.model.adapter == "fake":
            adapter = FakeModelAdapter()
            serializer = fake_tensor_payload
        else:
            if args.artifact_manifest is None:
                raise ConfigError(
                    "Moondream edge placements require --artifact-manifest"
                )
            from adaptive_vlm.models.moondream.adapter import MoondreamAdapter

            with progress.phase("Loading verified shared edge model"):
                adapter = MoondreamAdapter.from_pretrained(
                    config.model,
                    cache_dir=args.cache_dir,
                    artifact_manifest=args.artifact_manifest,
                    vision_device=args.edge_device,
                    text_device=args.edge_device,
                    offline=args.offline,
                    components="full" if "edge" in placements else "vision",
                    weight_residency=(
                        args.edge_residency if "edge" in placements else "resident"
                    ),
                )

    policies: dict[str, ExecutionPolicy] = {}
    run_manifests: dict[str, RunManifest] = {}
    for placement in placements:
        spec = specs[placement]
        policy: ExecutionPolicy
        if spec.mode is ExecutionMode.EDGE:
            policy = LocalExecutionPolicy(adapter)
        elif spec.mode is ExecutionMode.CLOUD:
            policy = CloudExecutionPolicy(transport=config.transport, model=model)
        else:
            policy = DistributedExecutionPolicy(
                encoder=adapter,
                transport=config.transport,
                model=model,
                tensor_serializer=serializer,
                pipeline=spec.pipeline or "synchronous",
                payload_format=spec.payload or "bf16",
            )
        if isinstance(policy, (CloudExecutionPolicy, DistributedExecutionPolicy)):
            with progress.phase(f"Checking remote model contract for {placement}"):
                if not policy.health().ready:
                    raise RuntimeError("remote server is not ready")
                if spec.remote_path not in policy.capabilities().paths:
                    raise RuntimeError(
                        f"remote server does not offer {spec.remote_path}"
                    )
        policies[placement] = policy
        placement_environment = dict(environment)
        is_edge = spec.mode is ExecutionMode.EDGE
        placement_environment["execution_placement"] = {
            "mode": spec.mode.value,
            "placement": placement,
            "configuration_id": spec.id.value,
            "vision_device": (
                "remote" if spec.mode is ExecutionMode.CLOUD else args.edge_device
            ),
            "text_device": args.edge_device if is_edge else "remote",
            "weight_residency": args.edge_residency if is_edge else "resident",
            "shared_edge_adapter": edge_uses_gpu
            and spec.mode is not ExecutionMode.CLOUD,
            "pipeline": spec.pipeline or "synchronous",
            "payload": spec.payload,
            "remote": f"{config.transport.host}:{config.transport.port}",
            "interleaved": True,
        }
        run_manifests[placement] = create_run_manifest(
            config, placement_environment, dataset_manifest_sha256=manifest.sha256
        )

    prepare = getattr(adapter, "activate_stage", None) if adapter is not None else None
    load_level = _load_level(args)
    generator = (
        make_generator(
            load_level,
            directory=args.run_dir.parent / f".{args.run_dir.name}-edge-load",
            device=args.edge_device,
        )
        if load_level is not None
        else None
    )
    sources: list[Callable[[], Snapshot]] = []
    windows: WindowsGpuCounterSampler | None = None
    if args.resource_state:
        index = _device_index(args.edge_device)
        sources += [gpu_state_source(index), host_memory_snapshot]
        try:
            windows = WindowsGpuCounterSampler()
        except FileNotFoundError:
            progress.info("powershell.exe unreachable: no Windows GPU counters")
        else:
            windows.start()
            if not windows.wait_for_sample():
                progress.info("Windows GPU counters produced no sample yet")
            sources.append(windows.snapshot)
    if generator is not None:
        sources.append(generator.snapshot)
    state_probe = combine(*sources, prefix="edge_") if sources else None
    try:
        if generator is not None:
            with progress.phase(
                f"Starting edge load {load_level.name if load_level else ''}"
            ):
                generator.start()
        summary = run_interleaved_benchmark(
            config=config,
            manifest=manifest,
            policies=policies,
            run_manifests=run_manifests,
            run_dir=args.run_dir,
            request_ids=request_ids,
            repetitions=repetitions,
            warmups=warmups,
            seed=args.order_seed,
            prepare_stage=prepare if callable(prepare) else None,
            state_probe=state_probe,
            precondition=generator.check if generator is not None else None,
            conditions=(
                {
                    **level_conditions(load_level),
                    "cloud_load_level": args.cloud_load_level,
                }
                if args.cloud_load_level
                else level_conditions(load_level)
            ),
            resume=args.resume,
            max_consecutive_failures=args.max_consecutive_failures,
        )
    except DeviceContextFailure as exc:
        print(f"error: {exc}", flush=True)
        return DEVICE_RESTART_EXIT_CODE
    except LoadGeneratorFailure as exc:
        print(f"error: {exc}; resume restarts the load generator", flush=True)
        return ENVIRONMENT_RESTART_EXIT_CODE
    finally:
        if generator is not None:
            final = generator.stop()
            progress.info(f"Edge load stopped: {json.dumps(final, sort_keys=True)}")
        if windows is not None:
            windows.stop()
    print(json.dumps(summary["placements"], indent=2, sort_keys=True))
    return 0


def add_interleaved_parser(
    subcommands: Any, add_edge_residency: Any
) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subcommands.add_parser(
        "interleaved-benchmark",
        help="Run every placement per request in a seeded interleaved order",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", type=Path, help="Split file with *_request_ids")
    parser.add_argument("--split-name", help="training, validation or evaluation")
    parser.add_argument(
        "--placements",
        nargs="+",
        choices=ALL_PLACEMENTS,
        default=list(PLACEMENTS),
        help="Run directories to measure (default: G2 v2's three)",
    )
    parser.add_argument("--order-seed", required=True, type=int)
    parser.add_argument(
        "--repetition-indices",
        type=int,
        nargs="+",
        help="Repetition indices to run (default: all configured repetitions)",
    )
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path(".model-cache"))
    parser.add_argument("--edge-device", default="cuda:0")
    add_edge_residency(parser)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-consecutive-failures", type=int, default=0)
    parser.add_argument(
        "--resource-state",
        action="store_true",
        help=(
            "Record edge GPU (NVML), host memory and Windows GPU shared-memory "
            "state before every timed request"
        ),
    )
    parser.add_argument(
        "--edge-load-config", type=Path, help="Edge-load levels JSON (L1, L2, ...)"
    )
    parser.add_argument(
        "--edge-load-level",
        default=NO_LOAD,
        help="L0 (no load, default) or a level from --edge-load-config",
    )
    parser.add_argument(
        "--cloud-load-level",
        help=(
            "Cloud-load level the campaign runs this sub-block under (C0, C1); "
            "recorded in the plan only, the supervisor drives the cloud worker"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _device_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if device.startswith("cuda:") else 0


def _load_level(args: argparse.Namespace) -> LoadLevel | None:
    if args.edge_load_level == NO_LOAD:
        return None
    if args.edge_load_config is None:
        raise ConfigError("--edge-load-level other than L0 needs --edge-load-config")
    levels = load_levels(args.edge_load_config)
    if args.edge_load_level not in levels:
        raise ConfigError(
            f"unknown edge-load level {args.edge_load_level}; "
            f"choose L0 or one of {sorted(levels)}"
        )
    return levels[args.edge_load_level]
