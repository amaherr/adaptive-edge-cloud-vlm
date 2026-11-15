"""Interleaved three-placement benchmark command (G2 v2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.interleaved import (
    DEVICE_RESTART_EXIT_CODE,
    PLACEMENTS,
    DeviceContextFailure,
    run_interleaved_benchmark,
)
from adaptive_vlm.config import (
    ConfigError,
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
from adaptive_vlm.progress import ProgressReporter
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
    config = load_experiment_config(args.config)
    manifest = load_frozen_manifest(
        args.manifest or config.dataset.manifest_path,
        expected_revision=config.dataset.revision,
    )
    placements = [p for p in PLACEMENTS if p in args.placements]
    if len(placements) != len(args.placements):
        raise ConfigError("--placements must not repeat a placement")
    if config.model.adapter not in {"fake", "moondream2"}:
        raise ConfigError(f"unsupported model adapter: {config.model.adapter}")
    remote = [p for p in placements if p != "edge"]
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
    edge_uses_gpu = any(p in placements for p in ("edge", "distributed"))
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
        policy: ExecutionPolicy
        if placement == "edge":
            policy = LocalExecutionPolicy(adapter)
        elif placement == "cloud":
            policy = CloudExecutionPolicy(transport=config.transport, model=model)
        else:
            policy = DistributedExecutionPolicy(
                encoder=adapter,
                transport=config.transport,
                model=model,
                tensor_serializer=serializer,
                pipeline="synchronous",
            )
        if isinstance(policy, (CloudExecutionPolicy, DistributedExecutionPolicy)):
            with progress.phase(f"Checking remote model contract for {placement}"):
                if not policy.health().ready:
                    raise RuntimeError("remote server is not ready")
        policies[placement] = policy
        placement_environment = dict(environment)
        placement_environment["execution_placement"] = {
            "mode": placement,
            "vision_device": "remote" if placement == "cloud" else args.edge_device,
            "text_device": args.edge_device if placement == "edge" else "remote",
            "weight_residency": (
                args.edge_residency if placement == "edge" else "resident"
            ),
            "shared_edge_adapter": edge_uses_gpu and placement != "cloud",
            "pipeline": "synchronous",
            "remote": f"{config.transport.host}:{config.transport.port}",
            "interleaved": True,
        }
        run_manifests[placement] = create_run_manifest(
            config, placement_environment, dataset_manifest_sha256=manifest.sha256
        )

    prepare = getattr(adapter, "activate_stage", None) if adapter is not None else None
    try:
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
            resume=args.resume,
            max_consecutive_failures=args.max_consecutive_failures,
        )
    except DeviceContextFailure as exc:
        print(f"error: {exc}", flush=True)
        return DEVICE_RESTART_EXIT_CODE
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
        "--placements", nargs="+", choices=PLACEMENTS, default=list(PLACEMENTS)
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
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser
