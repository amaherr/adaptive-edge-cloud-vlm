"""VQA subset and official-evaluator input commands."""

from __future__ import annotations

import argparse
import json

from adaptive_vlm.benchmark.vqa_export import (
    export_vqa_predictions,
    prepare_vqa_eval_subset,
)
from adaptive_vlm.data.vqa_import import import_vqa_v2


def run_vqa_import_command(args: argparse.Namespace) -> int:
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


def run_vqa_export_command(args: argparse.Namespace) -> int:
    report = export_vqa_predictions(
        args.manifest,
        args.run_dir,
        args.output,
        repetition=args.repetition,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def run_vqa_subset_command(args: argparse.Namespace) -> int:
    report = prepare_vqa_eval_subset(
        args.manifest,
        args.questions,
        args.annotations,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
