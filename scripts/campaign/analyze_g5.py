"""G5 analysis (amendment A4) over a finished trace campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_vlm.prediction.r2.g5 import analyse, requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--best-static", default="cloud/raw_image")
    parser.add_argument("--sla-ms", type=float, default=1500.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a G5 analysis")
    records = requests(args.campaign)
    summary = analyse(
        records,
        calibration=json.loads(args.calibration.read_text()),
        features=json.loads(args.features.read_text())["tasks"],
        best_static=args.best_static,
        sla_ms=args.sla_ms,
    )
    args.output.mkdir(parents=True)
    (args.output / "g5_requests.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    )
    (args.output / "g5_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps({k: summary[k] for k in ("gate", "regret", "sensing")}, indent=1))


if __name__ == "__main__":
    main()
