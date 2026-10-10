"""Command-line interface for configuration and infrastructure smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from adaptive_vlm.artifacts import create_run_manifest, write_json_atomic
from adaptive_vlm.config import ConfigError, load_experiment_config
from adaptive_vlm.contracts import InferenceRequest, RequestRecord
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.environment import collect_environment
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.models.fake import FakeModelAdapter


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
    except (ConfigError, OSError, ValueError) as exc:
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


if __name__ == "__main__":
    raise SystemExit(main())

