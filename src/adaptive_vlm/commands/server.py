"""Cloud and projected-tensor server commands."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from adaptive_vlm.config import (
    ConfigError,
    ExecutionMode,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.models.moondream import MoondreamAdapter
from adaptive_vlm.progress import ProgressReporter
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer


def run_cloud_server_command(
    *,
    config_path: Path,
    artifact_manifest: Path | None,
    cache_dir: Path,
    device: str,
    offline: bool,
    bind_host: str | None,
    verbose: bool,
    dual_paths: bool = False,
) -> int:
    progress = ProgressReporter(verbose)
    progress.detail(f"Loading experiment configuration from {config_path.resolve()}")
    config = load_experiment_config(config_path)
    validate_cloud_config(config.mode, config.cloud.status)
    if dual_paths and config.model.adapter != "moondream2":
        raise ConfigError("adaptive-server requires the Moondream2 adapter")
    seeded = seed_everything(config.runtime.seed)
    load_started = perf_counter()
    inference_function: Callable[[InferenceRequest], InferenceResult]
    with progress.phase(f"Loading {config.model.model_id} on {device}"):
        if config.model.adapter == "fake":
            fake_adapter = FakeModelAdapter()
            model_metadata = fake_adapter.metadata
            inference_function = fake_adapter.generate
        elif config.model.adapter == "moondream2":
            if artifact_manifest is None:
                raise ConfigError(
                    "cloud-server requires --artifact-manifest for Moondream2"
                )
            moondream_adapter = MoondreamAdapter.from_pretrained(
                config.model,
                cache_dir=cache_dir,
                artifact_manifest=artifact_manifest,
                vision_device=device,
                text_device=device,
                offline=offline,
            )
            model_metadata = moondream_adapter.metadata

            def run_monolithic(request: InferenceRequest) -> InferenceResult:
                return moondream_adapter.run_monolithic(request).result

            inference_function = run_monolithic
        else:
            raise ConfigError(
                f"unsupported cloud model adapter: {config.model.adapter}"
            )
    model_load_ms = (perf_counter() - load_started) * 1000
    contract_fingerprint = model_contract_fingerprint(config.model)
    progress.detail(f"Model contract fingerprint: {contract_fingerprint}")

    if dual_paths:
        processor = CloudRequestProcessor(
            protocol_version=config.transport.protocol_version,
            model_fingerprint=contract_fingerprint,
            inference_function=inference_function,
            tensor_inference_function=moondream_adapter.generate_from_payload,
            prompt_prepare_function=moondream_adapter.prepare_prompt,
            prepared_tensor_inference_function=moondream_adapter.generate_from_payload,
            pending_timeout_seconds=config.transport.timeout_seconds,
            event_callback=progress.info,
        )
    else:
        processor = CloudRequestProcessor(
            protocol_version=config.transport.protocol_version,
            model_fingerprint=contract_fingerprint,
            inference_function=inference_function,
            event_callback=progress.info,
        )
    server = CloudServer(
        host=bind_host or config.transport.host,
        port=config.transport.port,
        timeout_seconds=config.transport.timeout_seconds,
        processor=processor,
    )
    server.start()
    progress.info(f"Cloud server ready on {server.host}:{server.port}")
    progress.info("Waiting for requests; press Ctrl+C to stop")
    print(
        json.dumps(
            {
                "ready": True,
                "host": server.host,
                "port": server.port,
                "model_artifact_sha256": model_metadata.fingerprint,
                "model_contract_fingerprint": contract_fingerprint,
                "model_load_ms": model_load_ms,
                "seeded_providers": list(seeded),
                "supported_paths": list(processor.supported_paths),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.shutdown()
        progress.info("Cloud server stopped")
    return 0


def run_distributed_server_command(
    *,
    config_path: Path,
    artifact_manifest: Path,
    cache_dir: Path,
    device: str,
    offline: bool,
    bind_host: str | None,
    verbose: bool,
) -> int:
    progress = ProgressReporter(verbose)
    progress.detail(f"Loading experiment configuration from {config_path.resolve()}")
    config = load_experiment_config(config_path)
    validate_distributed_config(config.mode, config.cloud.status)
    if config.model.adapter != "moondream2":
        raise ConfigError("distributed-server requires the Moondream2 adapter")

    seeded = seed_everything(config.runtime.seed)
    load_started = perf_counter()
    with progress.phase(f"Loading the Moondream language stage on {device}"):
        adapter = MoondreamAdapter.from_pretrained(
            config.model,
            cache_dir=cache_dir,
            artifact_manifest=artifact_manifest,
            vision_device="cpu",
            text_device=device,
            offline=offline,
            components="text",
        )
    model_load_ms = (perf_counter() - load_started) * 1000
    contract_fingerprint = model_contract_fingerprint(config.model)
    processor = CloudRequestProcessor(
        protocol_version=config.transport.protocol_version,
        model_fingerprint=contract_fingerprint,
        tensor_inference_function=adapter.generate_from_payload,
        prompt_prepare_function=adapter.prepare_prompt,
        prepared_tensor_inference_function=adapter.generate_from_payload,
        pending_timeout_seconds=config.transport.timeout_seconds,
        event_callback=progress.info,
    )
    server = CloudServer(
        host=bind_host or config.transport.host,
        port=config.transport.port,
        timeout_seconds=config.transport.timeout_seconds,
        processor=processor,
    )
    server.start()
    progress.info(f"Distributed server ready on {server.host}:{server.port}")
    progress.info("Waiting for projected tensors; press Ctrl+C to stop")
    print(
        json.dumps(
            {
                "ready": True,
                "component_role": adapter.component_role,
                "host": server.host,
                "port": server.port,
                "model_artifact_sha256": adapter.metadata.fingerprint,
                "model_contract_fingerprint": contract_fingerprint,
                "model_load_ms": model_load_ms,
                "seeded_providers": list(seeded),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.shutdown()
        progress.info("Distributed server stopped")
    return 0


def validate_cloud_config(mode: ExecutionMode, cloud_status: str) -> None:
    if mode is not ExecutionMode.CLOUD:
        raise ConfigError("cloud commands require a cloud experiment configuration")
    if cloud_status == "pending":
        raise ConfigError("cloud hardware is pending; declare or verify it before use")


def validate_distributed_config(mode: ExecutionMode, cloud_status: str) -> None:
    if mode is not ExecutionMode.DISTRIBUTED:
        raise ConfigError(
            "distributed commands require a distributed experiment configuration"
        )
    if cloud_status == "pending":
        raise ConfigError(
            "cloud hardware is pending; use a declared or verified split server"
        )
