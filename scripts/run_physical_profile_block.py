"""Run one measured physical VQA block with three fixed GPU placements."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import subprocess
import sys
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.profile import build_profile
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.environment import collect_environment


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_json_atomic(path, value)


def save_plan(path: Path, plan: dict[str, Any]) -> None:
    # Compare the JSON representation so tuples cannot break an identical resume.
    canonical = json.loads(json.dumps(plan))
    if path.exists() and json.loads(path.read_text()) != canonical:
        raise RuntimeError("existing block plan differs; refusing to resume")
    write_json(path, canonical)


def prepare_run(run_dir: Path) -> bool:
    """Recover an interrupted initialization without discarding trial evidence."""
    if not run_dir.exists():
        return False
    if all((run_dir / name).is_file() for name in ("plan.json", "run_manifest.json")):
        return True
    raw = run_dir / "requests.jsonl"
    if raw.exists() and raw.stat().st_size:
        raise RuntimeError(f"missing run metadata with saved attempts: {run_dir}")
    history = run_dir.parent / "initialization_recovery"
    history.mkdir(exist_ok=True)
    run_dir.rename(history / f"{run_dir.name}-{datetime.now(UTC):%Y%m%dT%H%M%S%f}")
    return False


def policy_order(block_id: str, seed: int) -> tuple[str, ...]:
    orders = list(itertools.permutations(("edge", "cloud", "distributed")))
    random.Random(seed).shuffle(orders)
    return orders[(int(block_id[1:]) - 1) % len(orders)]


def check_edge_qdisc(interface: str, rate_mbps: float, delay_ms: float) -> str:
    output = subprocess.check_output(
        ["tc", "qdisc", "show", "dev", interface], text=True
    )
    first = output.splitlines()[0]
    rate = re.search(r"rate\s+([\d.]+)([KMG])bit", first)
    delay = re.search(r"delay\s+([\d.]+)(us|ms|s)", first)
    if "netem" not in first or rate is None or delay is None:
        raise RuntimeError(f"expected netem on {interface}: {first}")
    actual_rate = float(rate[1]) * {"K": 0.001, "M": 1, "G": 1000}[rate[2]]
    actual_delay = float(delay[1]) * {"us": 0.001, "ms": 1, "s": 1000}[delay[2]]
    if abs(actual_rate - rate_mbps) > 0.01 or abs(actual_delay - delay_ms) > 0.01:
        raise RuntimeError(f"wrong {interface} qdisc: {first}")
    return first


def measure_network(host: str, iperf_port: int, echo_port: int) -> dict[str, Any]:
    results = []
    for reverse in (False, True):
        command = [
            "iperf3",
            "-c",
            host,
            "-p",
            str(iperf_port),
            "-t",
            "8",
            "-O",
            "1",
            "-J",
        ]
        if reverse:
            command.append("-R")
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=60
        )
        results.append(json.loads(result.stdout))
    echo = subprocess.run(
        [
            sys.executable,
            "scripts/tcp_rtt_probe.py",
            "client",
            "--host",
            host,
            "--port",
            str(echo_port),
            "--count",
            "10",
            "--warmup",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    forward, reverse = results
    return {
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "forward_iperf": forward,
        "reverse_iperf": reverse,
        "echo": json.loads(echo.stdout),
        "edge_to_cloud_receiver_bps": forward["end"]["sum_received"]["bits_per_second"],
        "cloud_to_edge_receiver_bps": reverse["end"]["sum_received"]["bits_per_second"],
        "edge_sender_retransmissions": forward["end"]["sum_sent"].get("retransmits"),
        "cloud_sender_retransmissions": reverse["end"]["sum_sent"].get("retransmits"),
        "echo_median_rtt_ms": json.loads(echo.stdout)["rtt_ms"]["median"],
    }


def verify_measurement(
    measurement: dict[str, Any],
    rate_mbps: float,
    added_rtt_ms: float,
    baseline_rtt_ms: float,
) -> None:
    target_bps = rate_mbps * 1_000_000
    for name in ("edge_to_cloud_receiver_bps", "cloud_to_edge_receiver_bps"):
        actual = measurement[name]
        if not 0.8 * target_bps <= actual <= 1.2 * target_bps:
            raise RuntimeError(f"{name}={actual} outside target band")
    expected_rtt = baseline_rtt_ms + added_rtt_ms
    actual_rtt = measurement["echo_median_rtt_ms"]
    if abs(actual_rtt - expected_rtt) > max(5.0, 0.2 * added_rtt_ms):
        raise RuntimeError(f"echo RTT {actual_rtt} ms outside target band")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/phase2/physical-v1")
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("configs/profiles/physical_network_blocks_v1.json"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/evaluation/physical_vqa_split_v1.json"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/physical_vqa_v1.yaml")
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=Path("configs/model/moondream2_artifact.yaml"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".model-cache"))
    parser.add_argument(
        "--adaptive-vlm", type=Path, default=Path(".venv-model/bin/adaptive-vlm")
    )
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text())
    if args.block not in grid["blocks"]:
        parser.error(f"unknown block: {args.block}")
    block = grid["blocks"][args.block]
    split = json.loads(args.split.read_text())
    config = load_experiment_config(args.config)
    manifest = load_frozen_manifest(
        config.dataset.manifest_path, expected_revision=config.dataset.revision
    )
    if manifest.sha256 != split["dataset_manifest_sha256"]:
        raise RuntimeError("dataset differs from frozen split")
    if model_contract_fingerprint(config.model) != split["model_contract_sha256"]:
        raise RuntimeError("model contract differs from frozen split")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Git tree must be clean before physical profiling")
    qdisc = check_edge_qdisc(
        grid["edge_interface"], block["rate_mbps"], block["egress_delay_ms"]
    )
    output = args.output_root / args.block
    output.mkdir(parents=True, exist_ok=True)
    order = policy_order(args.block, grid["policy_order_seed"])
    plan = {
        "schema_version": 1,
        "block_id": args.block,
        "policy_order": list(order),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dataset_manifest_sha256": manifest.sha256,
        "split_sha256": digest(args.split),
        "grid_sha256": digest(args.grid),
        "model_contract_sha256": split["model_contract_sha256"],
        "edge_qdisc": qdisc,
        "configured_rate_mbps": block["rate_mbps"],
        "configured_added_rtt_ms": 2 * block["egress_delay_ms"],
    }
    plan_path = output / "plan.json"
    save_plan(plan_path, plan)
    environment_path = output / "environment.json"
    if not environment_path.exists():
        write_json(environment_path, collect_environment())
    # Every restart needs live evidence. Preserve earlier evidence for audit.
    live_pre = measure_network(
        grid["cloud_host"], grid["iperf_port"], grid["echo_port"]
    )
    verify_measurement(
        live_pre,
        block["rate_mbps"],
        2 * block["egress_delay_ms"],
        grid["unshaped_echo_median_ms"],
    )
    history = output / "network_preflights"
    history.mkdir(exist_ok=True)
    write_json(history / f"{datetime.now(UTC):%Y%m%dT%H%M%S%f}.json", live_pre)
    pre_path = output / "network_pre.json"
    if not pre_path.exists():
        write_json(pre_path, live_pre)
    pre = json.loads(pre_path.read_text())
    gpu_log = (output / "edge_gpu_samples.csv").open("a")
    try:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                "--loop-ms=1000",
            ],
            stdout=gpu_log,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        monitor = None
        gpu_log.write("nvidia-smi unavailable\n")
    try:
        for mode in order:
            run_dir = output / mode
            command = [
                str(args.adaptive_vlm),
                "benchmark",
                "--config",
                str(args.config),
                "--manifest",
                str(config.dataset.manifest_path),
                "--mode",
                mode,
                "--run-dir",
                str(run_dir),
                "--artifact-manifest",
                str(args.artifact_manifest),
                "--cache-dir",
                str(args.cache_dir),
                "--offline",
                "--warmups",
                "1",
                "--repetitions",
                "3",
                "--max-consecutive-failures",
                "3",
                "--vision-device",
                "cpu" if mode == "cloud" else "cuda:0",
                "--text-device",
                "cuda:0" if mode == "edge" else "cpu",
            ]
            if prepare_run(run_dir):
                command.append("--resume")
            print(
                f"Starting {args.block} {mode}; log: {output / (mode + '.log')}",
                flush=True,
            )
            with (output / f"{mode}.log").open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{mode} failed; see {output / (mode + '.log')}")
    finally:
        if monitor is not None:
            with suppress(ProcessLookupError):
                monitor.terminate()
            monitor.wait(timeout=5)
        gpu_log.close()
    post_path = output / "network_post.json"
    # Once a profile is published its evidence must remain immutable, including
    # a restart between profile publication and validation-report publication.
    if (output / "profile").exists():
        post = json.loads(post_path.read_text())
    else:
        post = measure_network(
            grid["cloud_host"], grid["iperf_port"], grid["echo_port"]
        )
        write_json(post_path, post)
    verify_measurement(
        post,
        block["rate_mbps"],
        2 * block["egress_delay_ms"],
        grid["unshaped_echo_median_ms"],
    )
    evidence_path = output / "network_evidence.json"
    write_json(evidence_path, {"pre": pre, "post": post})
    profile_spec = {
        "schema_version": 1,
        "campaign_id": f"physical-vqa-v1-{args.block}",
        "host_pair_id": "edge-rtx3050-cloud-rtx2070super-v1",
        "cloud_device": "cuda:0",
        "task_bucket": "vqa-short-answer-32-token-budget",
        "requested_max_new_tokens": 32,
        "network_state": {
            "bucket_id": args.block,
            "source": "physical_wsl_netem_measured",
            "controlled": True,
            "uplink_bps": min(
                pre["edge_to_cloud_receiver_bps"], post["edge_to_cloud_receiver_bps"]
            ),
            "downlink_bps": min(
                pre["cloud_to_edge_receiver_bps"], post["cloud_to_edge_receiver_bps"]
            ),
            "rtt_ms": (pre["echo_median_rtt_ms"] + post["echo_median_rtt_ms"]) / 2,
            "measurement_evidence_path": "network_evidence.json",
            "measurement_evidence_sha256": digest(evidence_path),
        },
    }
    spec_path = output / "profile_specification.json"
    write_json(spec_path, profile_spec)
    profile_dir = output / "profile"
    if not profile_dir.exists():
        with tempfile.TemporaryDirectory(prefix=".profile-", dir=output) as temporary:
            staged = Path(temporary) / "profile"
            summary = build_profile(
                run_dirs=[output / mode for mode in ("edge", "cloud", "distributed")],
                specification=spec_path,
                output_dir=staged,
            )
            staged.rename(profile_dir)
    else:
        summary = json.loads((profile_dir / "summary.json").read_text())
        if summary["specification_sha256"] != digest(spec_path):
            raise RuntimeError("existing profile differs from current network evidence")
    write_json(
        output / "validation_report.json",
        {
            "schema_version": 1,
            "block_id": args.block,
            "profile_summary": summary,
            "network_pre_sha256": digest(pre_path),
            "network_post_sha256": digest(post_path),
            "profile_sha256": digest(profile_dir / "profile.jsonl"),
            "edge_gpu_samples_sha256": digest(output / "edge_gpu_samples.csv"),
            "artifact_hashes": {
                str(path.relative_to(output)): digest(path)
                for path in [
                    output / "plan.json",
                    output / "profile_specification.json",
                    evidence_path,
                    *(
                        output / mode / name
                        for mode in ("edge", "cloud", "distributed")
                        for name in (
                            "plan.json",
                            "run_manifest.json",
                            "requests.jsonl",
                            "summary.json",
                        )
                    ),
                ]
            },
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
