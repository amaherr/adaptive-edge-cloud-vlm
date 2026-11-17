"""Build the same-device correctness reference bundle for the G2 v2 campaign.

Run every step from the repository root, on the commit the campaign will use
(the proofs bind the source hashes). See the runbook's G2 v2 section.

  same-device  on EACH host: monolithic vs decomposed for all manifest
               requests on its GPU (edge ~1 h with resident weights, which
               spill on 4 GiB; cloud ~15 min)
  runtime      on EACH host: record the runtime package versions
  control      on the edge, with the cloud server running: the distributed
               control run (1 warmup + 3 repetitions over the manifest);
               G2 v2+ runs it twice, the second time with --payload int8
  bundle       on the edge, after copying the cloud's same-device folder and
               runtime file across: bind and verify everything; prints the
               index path to put in the campaign settings
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.heterogeneous_correctness import runtime_versions
from adaptive_vlm.campaign.reference import build_bundle, same_device_proof
from adaptive_vlm.environment import disable_core_dumps

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
CONFIG = ROOT / "configs/experiments/physical_vqa_v2.yaml"
ARTIFACT = ROOT / "configs/model/moondream2_artifact.yaml"
CACHE = ROOT / ".model-cache"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("same-device", "runtime", "control", "bundle"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--edge", type=Path, help="bundle: edge same-device folder")
    parser.add_argument("--cloud", type=Path, help="bundle: cloud same-device folder")
    parser.add_argument("--control", type=Path, help="bundle: control run folder")
    parser.add_argument(
        "--payload",
        choices=("bf16", "int8"),
        default="bf16",
        help="control: projected-tensor encoding",
    )
    parser.add_argument(
        "--int8-control", type=Path, help="bundle: int8 control run folder (G2 v2+)"
    )
    parser.add_argument("--edge-runtime", type=Path)
    parser.add_argument("--cloud-runtime", type=Path)
    args = parser.parse_args()
    disable_core_dumps()
    if args.step == "runtime":
        write_json_atomic(args.output, runtime_versions())
    elif args.step == "same-device":
        import torch

        # The cloud host's WSL NVML crashes on device enumeration; PyTorch then
        # falls back to its CUDA runtime count (see run_physical_cloud_gpu.py).
        torch.cuda._device_count_nvml = lambda: -1
        from adaptive_vlm.config import load_experiment_config
        from adaptive_vlm.determinism import seed_everything
        from adaptive_vlm.models.moondream.adapter import MoondreamAdapter

        config = load_experiment_config(args.config)
        seed_everything(config.runtime.seed)
        adapter = MoondreamAdapter.from_pretrained(
            config.model,
            cache_dir=CACHE,
            artifact_manifest=ARTIFACT,
            vision_device=args.device,
            text_device=args.device,
            offline=True,
        )
        report = same_device_proof(
            config_path=args.config,
            output=args.output,
            script_path=SCRIPT,
            adapter=adapter,
            device_name=torch.cuda.get_device_name(args.device),
            log=lambda message: print(message, flush=True),
        )
        print(
            json.dumps(
                {k: report[k] for k in ("device", "completed", "mismatches", "passed")}
            )
        )
        if not report["passed"]:
            raise SystemExit(1)
    elif args.step == "control":
        command = [
            str(ROOT / ".venv-model/bin/adaptive-vlm"),
            "benchmark",
            "--config",
            str(args.config),
            "--mode",
            "distributed",
            "--run-dir",
            str(args.output),
            "--artifact-manifest",
            str(ARTIFACT),
            "--cache-dir",
            str(CACHE),
            "--offline",
            "--warmups",
            "1",
            "--repetitions",
            "3",
            "--vision-device",
            args.device,
            "--text-device",
            "cpu",
            "--max-consecutive-failures",
            "3",
            "--payload",
            args.payload,
        ]
        raise SystemExit(subprocess.run(command, check=False).returncode)
    else:
        missing = [
            name
            for name in ("edge", "cloud", "control", "edge_runtime", "cloud_runtime")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                f"bundle needs --{' --'.join(m.replace('_', '-') for m in missing)}"
            )
        index, proof = build_bundle(
            config_path=args.config,
            edge=args.edge,
            cloud=args.cloud,
            control=args.control,
            edge_runtime=args.edge_runtime,
            cloud_runtime=args.cloud_runtime,
            script_path=SCRIPT,
            output=args.output,
            int8_control=args.int8_control,
        )
        print(json.dumps(proof, indent=2))
        print(f"reference index: {index}", file=sys.stderr)


if __name__ == "__main__":
    main()
