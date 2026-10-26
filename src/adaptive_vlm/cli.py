"""Argument parsing and routing for the adaptive VLM command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from adaptive_vlm.commands import (
    adaptive,
    benchmark,
    calibrate,
    evaluate,
    infer,
    profile,
    server,
    vqa,
)
from adaptive_vlm.commands import (
    config as config_commands,
)
from adaptive_vlm.config import ConfigError, ExecutionMode
from adaptive_vlm.scheduler.configuration import ConfigurationId


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-vlm",
        description="Edge-cloud VLM baseline tooling.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    compare_parser = subcommands.add_parser(
        "compare-runs", help="Compare raw token outputs across policies"
    )
    compare_parser.add_argument("run_dirs", nargs="+", type=Path)
    compare_parser.add_argument("--output", type=Path)

    profile_parser = subcommands.add_parser(
        "profile-runs", help="Validate benchmark runs and emit Phase 2 profile rows"
    )
    profile_parser.add_argument("run_dirs", nargs="+", type=Path)
    profile_parser.add_argument("--specification", required=True, type=Path)
    profile_parser.add_argument("--output-dir", required=True, type=Path)

    predictor_parser = subcommands.add_parser(
        "calibrate-predictor",
        help="Fit and evaluate a predictor on frozen profile holdouts",
    )
    predictor_parser.add_argument("--profile", required=True, type=Path)
    predictor_parser.add_argument("--split", required=True, type=Path)
    predictor_parser.add_argument("--output-dir", required=True, type=Path)

    oracle_parser = subcommands.add_parser(
        "evaluate-selector",
        help="Evaluate a frozen selector against matched observations",
    )
    oracle_parser.add_argument("--predictions", required=True, type=Path)
    oracle_parser.add_argument("--specification", required=True, type=Path)
    oracle_parser.add_argument("--output-dir", required=True, type=Path)

    trace_parser = subcommands.add_parser(
        "replay-network-trace", help="Replay frozen network conditions through selector"
    )
    trace_parser.add_argument("--calibration", required=True, type=Path)
    trace_parser.add_argument("--trace", required=True, type=Path)
    trace_parser.add_argument("--output-dir", required=True, type=Path)

    benchmark_parser = subcommands.add_parser(
        "benchmark", help="Run a resumable edge, cloud, or distributed campaign"
    )
    benchmark_parser.add_argument("--config", required=True, type=Path)
    benchmark_parser.add_argument(
        "--mode", required=True, choices=tuple(mode.value for mode in ExecutionMode)
    )
    benchmark_parser.add_argument("--run-dir", required=True, type=Path)
    benchmark_parser.add_argument("--manifest", type=Path)
    benchmark_parser.add_argument("--artifact-manifest", type=Path)
    benchmark_parser.add_argument(
        "--cache-dir", type=Path, default=Path(".model-cache")
    )
    benchmark_parser.add_argument("--vision-device", default="cpu")
    benchmark_parser.add_argument("--text-device", default="cpu")
    benchmark_parser.add_argument("--offline", action="store_true")
    benchmark_parser.add_argument("--warmups", type=int)
    benchmark_parser.add_argument("--repetitions", type=int)
    benchmark_parser.add_argument(
        "--pipeline",
        choices=("synchronous", "prompt_first"),
        default="synchronous",
    )
    benchmark_parser.add_argument("--resume", action="store_true")
    benchmark_parser.add_argument("-v", "--verbose", action="store_true")

    import_parser = subcommands.add_parser(
        "import-vqa-v2", help="Freeze a seeded VQA v2 val2014 subset"
    )
    import_parser.add_argument("--questions", required=True, type=Path)
    import_parser.add_argument("--annotations", required=True, type=Path)
    import_parser.add_argument("--images-dir", required=True, type=Path)
    import_parser.add_argument("--output", required=True, type=Path)
    import_parser.add_argument("--count", type=int, default=100)
    import_parser.add_argument("--seed", type=int, default=42)
    import_parser.add_argument("--revision", required=True)

    config_parser = subcommands.add_parser("config", help="Configuration tools")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
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

    adaptive_server = subcommands.add_parser(
        "adaptive-server", help="Serve raw images and projected tensors together"
    )
    adaptive_server.add_argument("--config", required=True, type=Path)
    adaptive_server.add_argument("--artifact-manifest", required=True, type=Path)
    adaptive_server.add_argument("--cache-dir", default=Path(".model-cache"), type=Path)
    adaptive_server.add_argument("--device", default="cpu")
    adaptive_server.add_argument("--offline", action="store_true")
    adaptive_server.add_argument("--bind-host")
    adaptive_server.add_argument("-v", "--verbose", action="store_true")

    adaptive_parser = subcommands.add_parser(
        "adaptive-infer", help="Manually route one request through a loaded policy"
    )
    adaptive_parser.add_argument("--config", required=True, type=Path)
    adaptive_parser.add_argument(
        "--configuration-id",
        required=True,
        choices=tuple(item.value for item in ConfigurationId),
    )
    adaptive_parser.add_argument("--artifact-manifest", type=Path)
    adaptive_parser.add_argument("--cache-dir", default=Path(".model-cache"), type=Path)
    adaptive_parser.add_argument("--image", required=True, type=Path)
    adaptive_parser.add_argument("--prompt", required=True)
    adaptive_parser.add_argument("--request-id", default="adaptive-0001")
    adaptive_parser.add_argument("--vision-device", default="cpu")
    adaptive_parser.add_argument("--text-device", default="cpu")
    adaptive_parser.add_argument(
        "--edge-components", choices=("full", "vision", "none"), default="full"
    )
    adaptive_parser.add_argument("--uplink-mbps", type=float)
    adaptive_parser.add_argument("--rtt-ms", type=float)
    adaptive_parser.add_argument("--edge-memory-budget-mib", type=float)
    adaptive_parser.add_argument("--edge-required-peak-mib", type=float)
    adaptive_parser.add_argument("--expected-output-tokens", type=int)
    adaptive_parser.add_argument("--offline", action="store_true")
    adaptive_parser.add_argument("--output-dir", type=Path)
    adaptive_parser.add_argument("-v", "--verbose", action="store_true")

    adaptive_benchmark = subcommands.add_parser(
        "adaptive-benchmark", help="Run a frozen mixed-choice adaptive campaign"
    )
    adaptive_benchmark.add_argument("--config", required=True, type=Path)
    adaptive_benchmark.add_argument("--manifest", required=True, type=Path)
    adaptive_benchmark.add_argument("--calibration", required=True, type=Path)
    adaptive_benchmark.add_argument("--replay", required=True, type=Path)
    adaptive_benchmark.add_argument("--run-dir", required=True, type=Path)
    adaptive_benchmark.add_argument("--artifact-manifest", required=True, type=Path)
    adaptive_benchmark.add_argument(
        "--cache-dir", type=Path, default=Path(".model-cache")
    )
    adaptive_benchmark.add_argument("--vision-device", default="cpu")
    adaptive_benchmark.add_argument("--text-device", default="cpu")
    adaptive_benchmark.add_argument("--cloud-device", default="cpu")
    adaptive_benchmark.add_argument("--edge-required-peak-mib", type=float)
    adaptive_benchmark.add_argument(
        "--monitor-mode", choices=("replay", "live"), default="replay"
    )
    adaptive_benchmark.add_argument("--probe-interval-ms", type=float, default=0.0)
    adaptive_benchmark.add_argument("--probe-max-age-ms", type=float, default=5_000.0)
    adaptive_benchmark.add_argument("--probe-timeout-ms", type=float, default=2_000.0)
    adaptive_benchmark.add_argument("--passive-alpha", type=float, default=0.25)
    adaptive_benchmark.add_argument("--warmups", type=int, default=0)
    adaptive_benchmark.add_argument("--repetitions", type=int, default=1)
    adaptive_benchmark.add_argument("--predictor-evaluation", type=Path)
    adaptive_benchmark.add_argument("--selector-evaluation", type=Path)
    adaptive_benchmark.add_argument(
        "--diagnostic-allow-unpassed-gates", action="store_true"
    )
    adaptive_benchmark.add_argument("--resume", action="store_true")
    adaptive_benchmark.add_argument("--offline", action="store_true")
    adaptive_benchmark.add_argument("-v", "--verbose", action="store_true")

    adaptive_analysis = subcommands.add_parser(
        "analyze-adaptive", help="Compare a mixed adaptive run with fixed controls"
    )
    adaptive_analysis.add_argument("adaptive_run_dir", type=Path)
    adaptive_analysis.add_argument("fixed_run_dirs", nargs=3, type=Path)
    adaptive_analysis.add_argument("--output", type=Path)

    thesis_analysis = subcommands.add_parser(
        "analyze-thesis", help="Produce a split-safe paired scheduler evaluation"
    )
    thesis_analysis.add_argument("--specification", required=True, type=Path)
    thesis_analysis.add_argument("--adaptive-run", required=True, type=Path)
    thesis_analysis.add_argument("--fixed-runs", nargs=3, required=True, type=Path)
    thesis_analysis.add_argument("--output-dir", required=True, type=Path)

    vqa_export = subcommands.add_parser(
        "export-vqa-predictions", help="Export complete official-format VQA answers"
    )
    vqa_export.add_argument("--manifest", required=True, type=Path)
    vqa_export.add_argument("--run-dir", required=True, type=Path)
    vqa_export.add_argument("--output", required=True, type=Path)
    vqa_export.add_argument("--repetition", type=int, default=0)

    vqa_subset = subcommands.add_parser(
        "prepare-vqa-eval-subset", help="Match official VQA labels to frozen IDs"
    )
    vqa_subset.add_argument("--manifest", required=True, type=Path)
    vqa_subset.add_argument("--questions", required=True, type=Path)
    vqa_subset.add_argument("--annotations", required=True, type=Path)
    vqa_subset.add_argument("--output-dir", required=True, type=Path)

    monitor_analysis = subcommands.add_parser(
        "analyze-monitor", help="Audit live probe provenance and decision safety"
    )
    monitor_analysis.add_argument("run_dir", type=Path)
    monitor_analysis.add_argument("--output", type=Path)

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
        if args.command == "compare-runs":
            return evaluate.run_compare_command(args)
        if args.command == "profile-runs":
            return profile.run_profile_command(args)
        if args.command == "calibrate-predictor":
            return calibrate.run_calibrate_command(args)
        if args.command == "evaluate-selector":
            return evaluate.run_selector_evaluation_command(args)
        if args.command == "replay-network-trace":
            return evaluate.run_network_trace_command(args)
        if args.command == "benchmark":
            return benchmark.run_fixed_benchmark_command(args)
        if args.command == "import-vqa-v2":
            return vqa.run_vqa_import_command(args)
        if args.command == "config":
            return config_commands.run_validate_config_command(args.path, args.output)
        if args.command == "environment":
            return config_commands.run_collect_environment_command(
                args.workdir, args.output
            )
        if args.command == "fake-infer":
            return infer.run_fake_infer_command(
                config_path=args.config,
                image_path=args.image,
                prompt=args.prompt,
                request_id=args.request_id,
                output_dir=args.output_dir,
                verbose=args.verbose,
            )
        if args.command == "edge-infer":
            return infer.run_edge_infer_command(
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
        if args.command in {"cloud-server", "adaptive-server"}:
            return server.run_cloud_server_command(
                config_path=args.config,
                artifact_manifest=args.artifact_manifest,
                cache_dir=args.cache_dir,
                device=args.device,
                offline=args.offline,
                bind_host=args.bind_host,
                verbose=args.verbose,
                dual_paths=args.command == "adaptive-server",
            )
        if args.command == "adaptive-infer":
            return adaptive.run_adaptive_infer_command(args)
        if args.command == "adaptive-benchmark":
            return adaptive.run_adaptive_benchmark_command(
                args, rss_reader=adaptive.current_rss_mib
            )
        if args.command == "analyze-adaptive":
            return evaluate.run_adaptive_analysis_command(args)
        if args.command == "analyze-thesis":
            return evaluate.run_thesis_analysis_command(args)
        if args.command == "export-vqa-predictions":
            return vqa.run_vqa_export_command(args)
        if args.command == "prepare-vqa-eval-subset":
            return vqa.run_vqa_subset_command(args)
        if args.command == "analyze-monitor":
            return evaluate.run_monitor_analysis_command(args)
        if args.command == "cloud-infer":
            return infer.run_cloud_infer_command(
                config_path=args.config,
                image_path=args.image,
                prompt=args.prompt,
                request_id=args.request_id,
                output_dir=args.output_dir,
                verbose=args.verbose,
            )
        if args.command == "distributed-server":
            return server.run_distributed_server_command(
                config_path=args.config,
                artifact_manifest=args.artifact_manifest,
                cache_dir=args.cache_dir,
                device=args.device,
                offline=args.offline,
                bind_host=args.bind_host,
                verbose=args.verbose,
            )
        if args.command == "distributed-infer":
            return infer.run_distributed_infer_command(
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


if __name__ == "__main__":
    raise SystemExit(main())
