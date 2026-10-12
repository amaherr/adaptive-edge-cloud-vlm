"""Command-line interface for configuration and infrastructure smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

from adaptive_vlm.artifacts import create_run_manifest, write_json_atomic
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
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-vlm",
        description="Edge-cloud VLM baseline tooling.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    config_parser = subcommands.add_parser("config", help="Configuration tools")
    config_commands = config_parser.add_subparsers(
        dest="config_command", required=True
    )
    validate_parser = config_commands.add_parser(
        "validate", help="Validate and print a resolved experiment config"
    )
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--output", type=Path)

    environment_parser = subcommands.add_parser(
        "environment", help="Collect non-secret environment metadata"
    )
    environment_parser.add_argument("--output", type=Path)
    environment_parser.add_argument("--workdir", type=Path, default=Path.cwd())

    fake_parser = subcommands.add_parser(
        "fake-infer", help="Run the dependency-free fake adapter"
    )
    fake_parser.add_argument("--config", required=True, type=Path)
    fake_parser.add_argument("--image", required=True, type=Path)
    fake_parser.add_argument("--prompt", required=True)
    fake_parser.add_argument("--request-id", default="smoke-0001")
    fake_parser.add_argument("--output-dir", type=Path)
    fake_parser.add_argument("-v", "--verbose", action="store_true")

    edge_parser = subcommands.add_parser(
        "edge-infer", help="Run pinned Moondream inference on the edge host"
    )
    edge_parser.add_argument("--config", required=True, type=Path)
    edge_parser.add_argument("--artifact-manifest", required=True, type=Path)
    edge_parser.add_argument("--cache-dir", default=Path(".model-cache"), type=Path)
    edge_parser.add_argument("--image", required=True, type=Path)
    edge_parser.add_argument("--prompt", required=True)
    edge_parser.add_argument("--request-id", default="edge-0001")
    edge_parser.add_argument("--vision-device", default="cpu")
    edge_parser.add_argument("--text-device", default="cpu")
    edge_parser.add_argument("--offline", action="store_true")
    edge_parser.add_argument("--output-dir", type=Path)
    edge_parser.add_argument("-v", "--verbose", action="store_true")

    server_parser = subcommands.add_parser(
        "cloud-server", help="Serve monolithic inference on the cloud host"
    )
    server_parser.add_argument("--config", required=True, type=Path)
    server_parser.add_argument("--artifact-manifest", type=Path)
    server_parser.add_argument("--cache-dir", default=Path(".model-cache"), type=Path)
    server_parser.add_argument("--device", default="cpu")
    server_parser.add_argument("--offline", action="store_true")
    server_parser.add_argument("--bind-host")
    server_parser.add_argument("-v", "--verbose", action="store_true")

    cloud_parser = subcommands.add_parser(
        "cloud-infer", help="Send one raw image request to a cloud server"
    )
    cloud_parser.add_argument("--config", required=True, type=Path)
    cloud_parser.add_argument("--image", required=True, type=Path)
    cloud_parser.add_argument("--prompt", required=True)
    cloud_parser.add_argument("--request-id", default="cloud-0001")
    cloud_parser.add_argument("--output-dir", type=Path)
    cloud_parser.add_argument("-v", "--verbose", action="store_true")

    distributed_server_parser = subcommands.add_parser(
        "distributed-server",
        help="Serve the language stage of split Moondream inference",
    )
    distributed_server_parser.add_argument("--config", required=True, type=Path)
    distributed_server_parser.add_argument(
        "--artifact-manifest", required=True, type=Path
    )
    distributed_server_parser.add_argument(
        "--cache-dir", default=Path(".model-cache"), type=Path
    )
    distributed_server_parser.add_argument("--device", default="cpu")
    distributed_server_parser.add_argument("--offline", action="store_true")
    distributed_server_parser.add_argument("--bind-host")
    distributed_server_parser.add_argument("-v", "--verbose", action="store_true")

    distributed_parser = subcommands.add_parser(
        "distributed-infer",
        help="Encode an image locally and send projected features to the cloud",
    )
    distributed_parser.add_argument("--config", required=True, type=Path)
    distributed_parser.add_argument("--artifact-manifest", required=True, type=Path)
    distributed_parser.add_argument(
        "--cache-dir", default=Path(".model-cache"), type=Path
    )
    distributed_parser.add_argument("--image", required=True, type=Path)
    distributed_parser.add_argument("--prompt", required=True)
    distributed_parser.add_argument("--request-id", default="distributed-0001")
    distributed_parser.add_argument("--vision-device", default="cpu")
    distributed_parser.add_argument(
        "--pipeline",
        choices=("synchronous", "prompt_first"),
        default="synchronous",
        help="Send the prompt ahead of vision encoding when prompt_first is selected",
    )
    distributed_parser.add_argument("--offline", action="store_true")
    distributed_parser.add_argument("--output-dir", type=Path)
    distributed_parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "config":
            return _validate_config(args.path, args.output)
        if args.command == "environment":
            return _collect_environment(args.workdir, args.output)
        if args.command == "fake-infer":
            return _fake_infer(
                config_path=args.config,
                image_path=args.image,
                prompt=args.prompt,
                request_id=args.request_id,
                output_dir=args.output_dir,
                verbose=args.verbose,
            )
        if args.command == "edge-infer":
            return _edge_infer(
                config_path=args.config,
                artifact_manifest=args.artifact_manifest,
                cache_dir=args.cache_dir,
                image_path=args.image,
                prompt=args.prompt,
                request_id=args.request_id,
                vision_device=args.vision_device,
                text_device=args.text_device,
                offline=args.offline,
                output_dir=args.output_dir,
                verbose=args.verbose,
            )
        if args.command == "cloud-server":
            return _cloud_server(
                config_path=args.config,
                artifact_manifest=args.artifact_manifest,
                cache_dir=args.cache_dir,
                device=args.device,
                offline=args.offline,
                bind_host=args.bind_host,
                verbose=args.verbose,
            )
        if args.command == "cloud-infer":
            return _cloud_infer(
                config_path=args.config,
                image_path=args.image,
                prompt=args.prompt,
                request_id=args.request_id,
                output_dir=args.output_dir,
                verbose=args.verbose,
            )
        if args.command == "distributed-server":
            return _distributed_server(
                config_path=args.config,
                artifact_manifest=args.artifact_manifest,
                cache_dir=args.cache_dir,
                device=args.device,
                offline=args.offline,
                bind_host=args.bind_host,
                verbose=args.verbose,
            )
        if args.command == "distributed-infer":
            return _distributed_infer(
                config_path=args.config,
                artifact_manifest=args.artifact_manifest,
                cache_dir=args.cache_dir,
                image_path=args.image,
                prompt=args.prompt,
                request_id=args.request_id,
                vision_device=args.vision_device,
                pipeline=args.pipeline,
                offline=args.offline,
                output_dir=args.output_dir,
                verbose=args.verbose,
            )
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def _validate_config(path: Path, output: Path | None) -> int:
    config = load_experiment_config(path)
    payload = config.to_dict()
    if output is not None:
        write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _collect_environment(workdir: Path, output: Path | None) -> int:
    payload = collect_environment(workdir)
    if output is not None:
        write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _fake_infer(
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


def _edge_infer(
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


def _cloud_server(
    *,
    config_path: Path,
    artifact_manifest: Path | None,
    cache_dir: Path,
    device: str,
    offline: bool,
    bind_host: str | None,
    verbose: bool,
) -> int:
    progress = ProgressReporter(verbose)
    progress.detail(f"Loading experiment configuration from {config_path.resolve()}")
    config = load_experiment_config(config_path)
    _validate_cloud_config(config.mode, config.cloud.status)
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


def _cloud_infer(
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
    _validate_cloud_config(config.mode, config.cloud.status)
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


def _distributed_server(
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
    _validate_distributed_config(config.mode, config.cloud.status)
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


def _distributed_infer(
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
    _validate_distributed_config(config.mode, config.cloud.status)
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


def _validate_cloud_config(mode: ExecutionMode, cloud_status: str) -> None:
    if mode is not ExecutionMode.CLOUD:
        raise ConfigError("cloud commands require a cloud experiment configuration")
    if cloud_status == "pending":
        raise ConfigError(
            "cloud hardware is pending; declare or verify it before use"
        )


def _validate_distributed_config(mode: ExecutionMode, cloud_status: str) -> None:
    if mode is not ExecutionMode.DISTRIBUTED:
        raise ConfigError(
            "distributed commands require a distributed experiment configuration"
        )
    if cloud_status == "pending":
        raise ConfigError(
            "cloud hardware is pending; use a declared or verified split server"
        )


if __name__ == "__main__":
    raise SystemExit(main())

