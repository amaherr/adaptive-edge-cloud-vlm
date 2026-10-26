"""Command-line interface for configuration and infrastructure smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

from adaptive_vlm.artifacts import create_run_manifest, write_json_atomic
from adaptive_vlm.benchmark.adaptive_analysis import analyze_adaptive_run
from adaptive_vlm.benchmark.compare import compare_runs
from adaptive_vlm.benchmark.monitor_eval import analyze_monitor_run
from adaptive_vlm.benchmark.network_trace import run_network_trace
from adaptive_vlm.benchmark.oracle import run_oracle_evaluation
from adaptive_vlm.benchmark.predictor_eval import run_predictor_evaluation
from adaptive_vlm.benchmark.profile import build_profile
from adaptive_vlm.benchmark.runner import run_benchmark
from adaptive_vlm.benchmark.thesis_eval import analyze_thesis_campaign
from adaptive_vlm.benchmark.vqa_export import (
    export_vqa_predictions,
    prepare_vqa_eval_subset,
)
from adaptive_vlm.commands.adaptive import run_adaptive_benchmark_command
from adaptive_vlm.config import (
    ConfigError,
    ExecutionMode,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import InferenceRequest, InferenceResult, RequestRecord
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.data.vqa_import import import_vqa_v2
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.environment import collect_environment
from adaptive_vlm.execution.adaptive import load_phase1_policies
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.cloud import CloudExecutionPolicy
from adaptive_vlm.execution.distributed import DistributedExecutionPolicy
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.execution.remote import check_remote_capabilities
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.models.moondream import MoondreamAdapter
from adaptive_vlm.progress import ProgressReporter
from adaptive_vlm.scheduler.configuration import CapabilityRegistry, ConfigurationId
from adaptive_vlm.scheduler.dispatch import ConfigurationDispatcher
from adaptive_vlm.scheduler.state import SystemState, TaskState
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer


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
            print(
                json.dumps(
                    compare_runs(args.run_dirs, output=args.output),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "profile-runs":
            print(
                json.dumps(
                    build_profile(
                        run_dirs=args.run_dirs,
                        specification=args.specification,
                        output_dir=args.output_dir,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "calibrate-predictor":
            print(
                json.dumps(
                    run_predictor_evaluation(
                        profile_path=args.profile,
                        split_path=args.split,
                        output_dir=args.output_dir,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "evaluate-selector":
            print(
                json.dumps(
                    run_oracle_evaluation(
                        predictions_path=args.predictions,
                        specification_path=args.specification,
                        output_dir=args.output_dir,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "replay-network-trace":
            print(
                json.dumps(
                    run_network_trace(
                        calibration_path=args.calibration,
                        trace_path=args.trace,
                        output_dir=args.output_dir,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "import-vqa-v2":
            count = import_vqa_v2(
                args.questions,
                args.annotations,
                args.images_dir,
                args.output,
                count=args.count,
                seed=args.seed,
                revision=args.revision,
            )
            print(f"Frozen {count} VQA requests at {args.output}")
            return 0
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
        if args.command == "adaptive-server":
            return _cloud_server(
                config_path=args.config,
                artifact_manifest=args.artifact_manifest,
                cache_dir=args.cache_dir,
                device=args.device,
                offline=args.offline,
                bind_host=args.bind_host,
                verbose=args.verbose,
                dual_paths=True,
            )
        if args.command == "adaptive-infer":
            return _adaptive_infer(args)
        if args.command == "adaptive-benchmark":
            return run_adaptive_benchmark_command(args, rss_reader=_current_rss_mib)
        if args.command == "analyze-adaptive":
            print(
                json.dumps(
                    analyze_adaptive_run(
                        args.adaptive_run_dir, args.fixed_run_dirs, output=args.output
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "analyze-thesis":
            print(
                json.dumps(
                    analyze_thesis_campaign(
                        args.specification,
                        args.adaptive_run,
                        args.fixed_runs,
                        output_dir=args.output_dir,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "export-vqa-predictions":
            print(
                json.dumps(
                    export_vqa_predictions(
                        args.manifest,
                        args.run_dir,
                        args.output,
                        repetition=args.repetition,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "prepare-vqa-eval-subset":
            print(
                json.dumps(
                    prepare_vqa_eval_subset(
                        args.manifest,
                        args.questions,
                        args.annotations,
                        args.output_dir,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "analyze-monitor":
            print(
                json.dumps(
                    analyze_monitor_run(args.run_dir, output=args.output),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
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


def _benchmark(args: argparse.Namespace) -> int:
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


def _current_rss_mib() -> float | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return None


def _adaptive_infer(args: argparse.Namespace) -> int:
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
    resident_memory = _current_rss_mib()
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
    dual_paths: bool = False,
) -> int:
    progress = ProgressReporter(verbose)
    progress.detail(f"Loading experiment configuration from {config_path.resolve()}")
    config = load_experiment_config(config_path)
    _validate_cloud_config(config.mode, config.cloud.status)
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
        raise ConfigError("cloud hardware is pending; declare or verify it before use")


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
