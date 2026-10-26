"""Predictor-calibration CLI command."""

from __future__ import annotations

import argparse
import json

from adaptive_vlm.benchmark.predictor_eval import run_predictor_evaluation


def run_calibrate_command(args: argparse.Namespace) -> int:
    report = run_predictor_evaluation(
        profile_path=args.profile,
        split_path=args.split,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
