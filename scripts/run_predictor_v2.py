"""Run isolated physical Predictor V2 training (no final-evaluation route)."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.calibrate_v2 import train_bundle
from adaptive_vlm.benchmark.predictor_eval_v2 import (
    evaluate_validation,
    review_training,
    seal_validation_predictions,
)
from adaptive_vlm.benchmark.predictor_v2_data import digest, strict_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "review", "validate"))
    parser.add_argument("--data-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path)
    args = parser.parse_args()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("commit source before the reproducible predictor run")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.action == "fit":
        report = train_bundle(args.data_bundle, args.output_dir)
    else:
        if args.fit_dir is None:
            parser.error("--fit-dir is required for review/validate")
        if args.action == "review":
            report = review_training(args.data_bundle, args.fit_dir, args.output_dir)
        else:
            seal_validation_predictions(args.data_bundle, args.fit_dir, args.output_dir)
            report = evaluate_validation(
                args.data_bundle, args.fit_dir, args.output_dir
            )
    audit = strict_json((args.data_bundle / "partition_audit.json").read_bytes())
    write_json_atomic(
        args.output_dir / "source_manifest.json",
        {
            "code_git_commit": commit,
            "git_dirty": False,
            "source_partition_audit_sha256": digest(
                args.data_bundle / "partition_audit.json"
            ),
            "training_rows_sha256": audit["files"]["training.jsonl"],
            "source_archive_sha256": audit["archive_sha256"],
            "frozen_split_sha256": audit["split_sha256"],
            "data_scope": "Physical G2 only; AWS/Vultr excluded",
            "validation_outcomes_read": args.action == "validate",
            "final_outcomes_read": False,
        },
    )
    write_json_atomic(
        args.output_dir / "run_environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "training_device": "CPU, dependency-light scalar linear algebra",
            "source_commit": commit,
        },
    )
    write_json_atomic(
        args.output_dir / "file_hashes.json",
        {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
