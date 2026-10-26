"""Fixed-policy benchmark command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.runner import run_benchmark
from adaptive_vlm.config import (
    ConfigError,
    ExecutionMode,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.environment import collect_environment
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.cloud import CloudExecutionPolicy
from adaptive_vlm.execution.distributed import DistributedExecutionPolicy
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.models.moondream import MoondreamAdapter
from adaptive_vlm.progress import ProgressReporter


def run_fixed_benchmark_command(args: argparse.Namespace) -> int:
    progress = ProgressReporter(args.verbose)
    config = load_experiment_config(args.config)
    manifest = load_frozen_manifest(
        args.manifest or config.dataset.manifest_path,
        expected_revision=config.dataset.revision,
    )
    mode = ExecutionMode(args.mode)
    if mode is ExecutionMode.DISTRIBUTED and config.model.adapter != "moondream2":
        raise ConfigError("distributed benchmark requires the Moondream2 adapter")
    if mode is not ExecutionMode.EDGE and config.cloud.status == "pending":
        raise ConfigError("remote benchmark requires declared cloud hardware")
    seeded = seed_everything(config.runtime.seed)
    environment = collect_environment()
    environment["seeded_providers"] = list(seeded)
    environment["execution_placement"] = {
        "mode": mode.value,
        "vision_device": args.vision_device,
        "text_device": args.text_device,
        "pipeline": args.pipeline,
        "remote": f"{config.transport.host}:{config.transport.port}",
    }
    lock_path = Path("requirements/model-cu128.lock")
    lock_sha256 = None
    if lock_path.is_file():
        from adaptive_vlm.model_artifact import sha256_file

        lock_sha256 = sha256_file(lock_path)
    run_manifest = create_run_manifest(
        config,
        environment,
        dataset_manifest_sha256=manifest.sha256,
        dependency_lock_sha256=lock_sha256,
    )
    model = ModelMetadata(
        model_id=config.model.model_id,
        revision=config.model.revision,
        fingerprint=model_contract_fingerprint(config.model),
    )
    policy: ExecutionPolicy
    if mode is ExecutionMode.EDGE:
        if config.model.adapter == "fake":
            policy = LocalExecutionPolicy(FakeModelAdapter())
        else:
            if args.artifact_manifest is None:
                raise ConfigError("Moondream benchmark requires --artifact-manifest")
            with progress.phase("Loading verified full model"):
                adapter = MoondreamAdapter.from_pretrained(
                    config.model,
                    cache_dir=args.cache_dir,
                    artifact_manifest=args.artifact_manifest,
                    vision_device=args.vision_device,
                    text_device=args.text_device,
                    offline=args.offline,
                )
            policy = LocalExecutionPolicy(adapter)
    elif mode is ExecutionMode.CLOUD:
        policy = CloudExecutionPolicy(transport=config.transport, model=model)
    else:
        if args.artifact_manifest is None:
            raise ConfigError("distributed benchmark requires --artifact-manifest")
        with progress.phase("Loading verified vision stage"):
            encoder = MoondreamAdapter.from_pretrained(
                config.model,
                cache_dir=args.cache_dir,
                artifact_manifest=args.artifact_manifest,
                vision_device=args.vision_device,
                text_device="cpu",
                offline=args.offline,
                components="vision",
            )
        policy = DistributedExecutionPolicy(
            encoder=encoder,
            transport=config.transport,
            model=model,
            pipeline=args.pipeline,
        )
    if isinstance(policy, (CloudExecutionPolicy, DistributedExecutionPolicy)):
        with progress.phase("Checking remote model contract"):
            if not policy.health().ready:
                raise RuntimeError("remote server is not ready")
    summary = run_benchmark(
        config=config,
        manifest=manifest,
        policy=policy,
        run_manifest=run_manifest,
        run_dir=args.run_dir,
        warmups=(
            args.warmups if args.warmups is not None else config.runtime.warmup_requests
        ),
        repetitions=(
            args.repetitions
            if args.repetitions is not None
            else config.runtime.measured_repetitions
        ),
        pipeline=args.pipeline,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
