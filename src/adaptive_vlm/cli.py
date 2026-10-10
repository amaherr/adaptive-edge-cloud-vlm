"""Command-line interface for configuration and infrastructure smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from adaptive_vlm.artifacts import create_run_manifest, write_json_atomic
from adaptive_vlm.config import ConfigError, ExecutionMode, load_experiment_config
from adaptive_vlm.contracts import InferenceRequest, InferenceResult, RequestRecord
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.environment import collect_environment
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.models.moondream import MoondreamAdapter


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
) -> int:
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
) -> int:
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
    load_started = perf_counter()
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
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

