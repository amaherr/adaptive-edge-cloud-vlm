"""Comparison and evaluation CLI commands."""

from __future__ import annotations

import argparse
import json

from adaptive_vlm.benchmark.adaptive_analysis import analyze_adaptive_run
from adaptive_vlm.benchmark.compare import compare_runs
from adaptive_vlm.benchmark.monitor_eval import analyze_monitor_run
from adaptive_vlm.benchmark.network_trace import run_network_trace
from adaptive_vlm.benchmark.oracle import run_oracle_evaluation
from adaptive_vlm.benchmark.thesis_eval import analyze_thesis_campaign


def _print_report(report: dict[str, object]) -> int:
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def run_compare_command(args: argparse.Namespace) -> int:
    return _print_report(compare_runs(args.run_dirs, output=args.output))


def run_selector_evaluation_command(args: argparse.Namespace) -> int:
    return _print_report(
        run_oracle_evaluation(
            predictions_path=args.predictions,
            specification_path=args.specification,
            output_dir=args.output_dir,
        )
    )


def run_network_trace_command(args: argparse.Namespace) -> int:
    return _print_report(
        run_network_trace(
            calibration_path=args.calibration,
            trace_path=args.trace,
            output_dir=args.output_dir,
        )
    )


def run_adaptive_analysis_command(args: argparse.Namespace) -> int:
    return _print_report(
        analyze_adaptive_run(
            args.adaptive_run_dir, args.fixed_run_dirs, output=args.output
        )
    )


def run_thesis_analysis_command(args: argparse.Namespace) -> int:
    return _print_report(
        analyze_thesis_campaign(
            args.specification,
            args.adaptive_run,
            args.fixed_runs,
            output_dir=args.output_dir,
        )
    )


def run_monitor_analysis_command(args: argparse.Namespace) -> int:
    return _print_report(analyze_monitor_run(args.run_dir, output=args.output))
