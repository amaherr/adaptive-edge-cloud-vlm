"""Profile-generation CLI command."""

from __future__ import annotations

import argparse
import json

from adaptive_vlm.benchmark.profile import build_profile


def run_profile_command(args: argparse.Namespace) -> int:
    report = build_profile(
        run_dirs=args.run_dirs,
        specification=args.specification,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
