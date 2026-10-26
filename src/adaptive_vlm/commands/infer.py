"""Single-request local and remote inference commands."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from adaptive_vlm.artifacts import create_run_manifest, write_json_atomic
from adaptive_vlm.commands.server import (
    validate_cloud_config,
    validate_distributed_config,
)
from adaptive_vlm.config import (
    ConfigError,
    ExecutionMode,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import InferenceRequest, InferenceResult, RequestRecord
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.environment import collect_environment
from adaptive_vlm.execution.cloud import CloudExecutionPolicy
from adaptive_vlm.execution.distributed import DistributedExecutionPolicy
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.models.moondream import MoondreamAdapter
from adaptive_vlm.progress import ProgressReporter


def run_fake_infer_command(
    *,
    config_path: Path,
    image_path: Path,
    prompt: str,
    request_id: str,
    output_dir: Path | None,
    verbose: bool,
) -> int:
    progress = ProgressReporter(verbose)
    progress.detail(f"Loading experiment configuration from {config_path.resolve()}")
    config = load_experiment_config(config_path)
    if config.model.adapter != "fake":
        raise ConfigError(
            "fake-infer requires a configuration whose model adapter is 'fake'"
        )
    seeded = seed_everything(config.runtime.seed)
    environment = collect_environment()
    environment["seeded_providers"] = list(seeded)
    manifest = create_run_manifest(config, environment)

    policy = LocalExecutionPolicy(FakeModelAdapter())
    with progress.phase(f"Running fake inference for {request_id}"):
        result = policy.infer(
            InferenceRequest(
                request_id=request_id,
                image_path=image_path.resolve(),
                prompt=prompt,
            )
        )
    record = RequestRecord.from_result(
        run_id=manifest.run_id,
        mode=config.mode,
        result=result,
    )

    if output_dir is not None:
        run_directory = output_dir / manifest.run_id
        write_json_atomic(run_directory / "run_manifest.json", manifest.to_dict())
        write_json_atomic(run_directory / "request.json", record.to_dict())
        progress.info(f"Saved run artifacts to {run_directory}")
    progress.info(f"Inference complete with {len(result.output_token_ids)} tokens")
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return 0


def run_edge_infer_command(
    *,
    config_path: Path,
    artifact_manifest: Path,
    cache_dir: Path,
    image_path: Path,
    prompt: str,
    request_id: str,
    vision_device: str,
    text_device: str,
    offline: bool,
    output_dir: Path | None,
    verbose: bool,
) -> int:
    progress = ProgressReporter(verbose)
    progress.detail(f"Loading experiment configuration from {config_path.resolve()}")
    config = load_experiment_config(config_path)
    if config.mode is not ExecutionMode.EDGE:
        raise ConfigError("edge-infer requires an edge experiment configuration")
    if config.model.adapter != "moondream2":
        raise ConfigError("edge-infer requires the Moondream2 model adapter")

    seeded = seed_everything(config.runtime.seed)
    environment = collect_environment()
    environment["seeded_providers"] = list(seeded)
    environment["execution_placement"] = {
        "vision_device": vision_device,
        "text_device": text_device,
        "offline_model_cache": offline,
    }
    manifest = create_run_manifest(config, environment)
    progress.detail(
        f"Model {config.model.model_id}@{config.model.revision}; "
        f"vision={vision_device}, text={text_device}, offline={offline}"
    )
    load_started = perf_counter()
    with progress.phase("Loading and verifying the pinned Moondream2 model"):
        adapter = MoondreamAdapter.from_pretrained(
            config.model,
            cache_dir=cache_dir,
            artifact_manifest=artifact_manifest,
            vision_device=vision_device,
            text_device=text_device,
            offline=offline,
        )
    model_load_ms = (perf_counter() - load_started) * 1000
    policy = LocalExecutionPolicy(adapter)
    with progress.phase(f"Running edge inference for {request_id}"):
        result = policy.infer(
            InferenceRequest(
                request_id=request_id,
                image_path=image_path.resolve(),
                prompt=prompt,
            )
        )
    result = InferenceResult(
        request_id=result.request_id,
        output_text=result.output_text,
        output_token_ids=result.output_token_ids,
        metrics={**result.metrics, "model_load_ms": model_load_ms},
    )
    record = RequestRecord.from_result(
        run_id=manifest.run_id,
        mode=policy.mode,
        result=result,
    )

    if output_dir is not None:
        run_directory = output_dir / manifest.run_id
        write_json_atomic(run_directory / "run_manifest.json", manifest.to_dict())
        write_json_atomic(run_directory / "request.json", record.to_dict())
        progress.info(f"Saved run artifacts to {run_directory}")
    progress.info(f"Inference complete with {len(result.output_token_ids)} tokens")
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return 0


def run_cloud_infer_command(
    *,
    config_path: Path,
    image_path: Path,
    prompt: str,
    request_id: str,
    output_dir: Path | None,
    verbose: bool,
) -> int:
    progress = ProgressReporter(verbose)
    progress.detail(f"Loading experiment configuration from {config_path.resolve()}")
    config = load_experiment_config(config_path)
    validate_cloud_config(config.mode, config.cloud.status)
    seeded = seed_everything(config.runtime.seed)
    environment = collect_environment()
    environment["seeded_providers"] = list(seeded)
    environment["execution_placement"] = {
        "client": config.edge.host_id,
        "server": config.cloud.host_id,
        "server_address": config.transport.host,
        "server_port": config.transport.port,
    }
    manifest = create_run_manifest(config, environment)
    model = ModelMetadata(
        model_id=config.model.model_id,
        revision=config.model.revision,
        fingerprint=model_contract_fingerprint(config.model),
    )
    policy = CloudExecutionPolicy(transport=config.transport, model=model)
    progress.detail(f"Expected model contract fingerprint: {model.fingerprint}")
    with progress.phase(
        f"Checking cloud server at {config.transport.host}:{config.transport.port}"
    ):
        health = policy.health()
    if not health.ready:
        raise RuntimeError("cloud server is not ready")
    with progress.phase(f"Running cloud inference for {request_id}"):
        result = policy.infer(
            InferenceRequest(
                request_id=request_id,
                image_path=image_path.resolve(),
                prompt=prompt,
            )
        )
    record = RequestRecord.from_result(
        run_id=manifest.run_id,
        mode=policy.mode,
        result=result,
    )
    if output_dir is not None:
        run_directory = output_dir / manifest.run_id
        write_json_atomic(run_directory / "run_manifest.json", manifest.to_dict())
        write_json_atomic(run_directory / "request.json", record.to_dict())
        progress.info(f"Saved run artifacts to {run_directory}")
    completion = f"Inference complete with {len(result.output_token_ids)} tokens"
    request_bytes = result.metrics.get("request_wire_bytes")
    response_bytes = result.metrics.get("response_wire_bytes")
    if request_bytes is not None and response_bytes is not None:
        completion += (
            f"; sent {request_bytes} bytes and received {response_bytes} bytes"
        )
    progress.info(completion)
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return 0


def run_distributed_infer_command(
    *,
    config_path: Path,
    artifact_manifest: Path,
    cache_dir: Path,
    image_path: Path,
    prompt: str,
    request_id: str,
    vision_device: str,
    pipeline: str,
    offline: bool,
    output_dir: Path | None,
    verbose: bool,
) -> int:
    progress = ProgressReporter(verbose)
    progress.detail(f"Loading experiment configuration from {config_path.resolve()}")
    config = load_experiment_config(config_path)
    validate_distributed_config(config.mode, config.cloud.status)
    if config.model.adapter != "moondream2":
        raise ConfigError("distributed-infer requires the Moondream2 adapter")

    seeded = seed_everything(config.runtime.seed)
    environment = collect_environment()
    environment["seeded_providers"] = list(seeded)
    environment["execution_placement"] = {
        "vision_device": vision_device,
        "language_server": config.cloud.host_id,
        "server_address": config.transport.host,
        "server_port": config.transport.port,
        "offline_model_cache": offline,
        "distributed_pipeline": pipeline,
    }
    manifest = create_run_manifest(config, environment)
    load_started = perf_counter()
    with progress.phase(f"Loading the Moondream vision stage on {vision_device}"):
        adapter = MoondreamAdapter.from_pretrained(
            config.model,
            cache_dir=cache_dir,
            artifact_manifest=artifact_manifest,
            vision_device=vision_device,
            text_device="cpu",
            offline=offline,
            components="vision",
        )
    model_load_ms = (perf_counter() - load_started) * 1000
    model = ModelMetadata(
        model_id=config.model.model_id,
        revision=config.model.revision,
        fingerprint=model_contract_fingerprint(config.model),
    )
    policy = DistributedExecutionPolicy(
        encoder=adapter,
        transport=config.transport,
        model=model,
        pipeline=pipeline,
    )
    with progress.phase(
        f"Checking language server at {config.transport.host}:{config.transport.port}"
    ):
        health = policy.health()
    if not health.ready:
        raise RuntimeError("distributed language server is not ready")
    with progress.phase(f"Running distributed inference for {request_id}"):
        result = policy.infer(
            InferenceRequest(
                request_id=request_id,
                image_path=image_path.resolve(),
                prompt=prompt,
            )
        )
    result = InferenceResult(
        request_id=result.request_id,
        output_text=result.output_text,
        output_token_ids=result.output_token_ids,
        metrics={**result.metrics, "edge_model_load_ms": model_load_ms},
    )
    record = RequestRecord.from_result(
        run_id=manifest.run_id,
        mode=policy.mode,
        result=result,
    )
    if output_dir is not None:
        run_directory = output_dir / manifest.run_id
        write_json_atomic(run_directory / "run_manifest.json", manifest.to_dict())
        write_json_atomic(run_directory / "request.json", record.to_dict())
        progress.info(f"Saved run artifacts to {run_directory}")
    completion = f"Inference complete with {len(result.output_token_ids)} tokens"
    payload_bytes = result.metrics.get("projected_payload_bytes")
    if payload_bytes is not None:
        completion += f"; sent {payload_bytes} projected-tensor bytes"
    progress.info(completion)
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return 0
