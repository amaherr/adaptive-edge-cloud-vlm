"""G4 for the frozen r2 selector (docs/predictor-v2/g4-specification.md).

best-static  validation cells only -> the best single configuration
freeze       write the G4 specification JSON (hashes of every input and of
             the selection code) before the evaluation cells are opened
evaluate     verify the specification, then score the selector on E01-E05
             against the measured oracle; refuses to run twice
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.prediction.r2.g4 import (
    MEDIAN_RELATIVE_REGRET_LIMIT,
    MIN_COMPLETE_BLOCKS,
    RESAMPLES,
    SEED,
    best_static,
    evaluate_decisions,
    profile_units,
)
from adaptive_vlm.prediction.r2.units import scored_units

ROOT = Path(__file__).resolve().parents[2]
CODE = (
    "src/adaptive_vlm/prediction/r2/features.py",
    "src/adaptive_vlm/prediction/r2/models.py",
    "src/adaptive_vlm/prediction/r2/units.py",
    "src/adaptive_vlm/prediction/r2/g4.py",
    "src/adaptive_vlm/scheduler/selector.py",
    "src/adaptive_vlm/scheduler/configuration.py",
    "scripts/prediction/run_g4_r2.py",
)
TRAINING = [f"T{i:02d}" for i in range(1, 13)]
VALIDATION = [f"V{i:02d}" for i in range(1, 6)]
EVALUATION = [f"E{i:02d}" for i in range(1, 6)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    return _sha256(path)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _commit() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty_paths": git("status", "--porcelain").splitlines(),
    }


def best_static_step(args: argparse.Namespace) -> None:
    audit = _read(args.partitions / "partition_audit.json")
    targets = Path(audit["sealed_validation_path"]).read_bytes()
    if (
        hashlib.sha256(targets).hexdigest()
        != audit["files"]["validation_targets.jsonl"]
    ):
        raise ValueError("validation targets differ from their audit hash")
    units = scored_units(
        args.partitions / "validation_inputs.jsonl", targets.splitlines()
    )
    result = best_static(units) | {
        "split": "validation",
        "validation_targets_sha256": audit["files"]["validation_targets.jsonl"],
    }
    _write(args.output, result)
    print(json.dumps(result, indent=1))


def freeze_step(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a frozen G4 specification")
    marker = _read(args.freeze_marker)
    if _sha256(args.calibration) != marker["calibration_sha256"]:
        raise ValueError("calibration is not the frozen r2 model")
    static = _read(args.best_static)
    spec = {
        "schema": "g4-r2-specification",
        "frozen_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": _commit(),
        "model": marker["model"],
        "calibration_sha256": marker["calibration_sha256"],
        "freeze_marker_sha256": _sha256(args.freeze_marker),
        "g3_decision_sha256": marker["decision_sha256"],
        "code_sha256": {path: _sha256(ROOT / path) for path in CODE},
        "best_static": {
            "configuration": static["configuration"],
            "file_sha256": _sha256(args.best_static),
        },
        "selector": {
            "primary": "all five configurations (int8 admitted, A2a rule 4)",
            "secondary": "four lossless configurations",
            "constraints": "none (no SLA, quality parity, resident edge)",
        },
        "unit": "request-cell; observed latency = median of the successful "
        "repetitions; complete only if every configuration succeeded in every "
        "repetition",
        "oracle": "configuration with the lowest observed latency in the "
        "request-cell; configuration order breaks ties",
        "block": "cell (all five configurations on the same request IDs)",
        "holdout_conditions": EVALUATION,
        "fitted_conditions": TRAINING + VALIDATION,
        "gate": {
            "median_relative_regret_percent_max": MEDIAN_RELATIVE_REGRET_LIMIT,
            "complete_blocks_min": MIN_COMPLETE_BLOCKS,
            "network_holdout": True,
            "no_feasible_decisions": 0,
            "unsafe_selections": 0,
            "predicted_oracle_consistent": True,
        },
        "bootstrap": {"seed": SEED, "resamples": RESAMPLES, "cluster": "request ID"},
    }
    digest = _write(args.output, spec)
    print("frozen G4 specification", digest)


def evaluate_step(args: argparse.Namespace) -> None:
    if (args.output / "evaluation.json").exists():
        raise FileExistsError("a G4 evaluation exists; the evaluation pass is single")
    if _sha256(args.specification) != args.specification_sha256:
        raise ValueError("G4 specification differs from the frozen hash")
    spec = _read(args.specification)
    changed = [p for p, d in spec["code_sha256"].items() if _sha256(ROOT / p) != d]
    if changed:
        raise ValueError(f"selection code changed since the freeze: {changed}")
    if _sha256(args.calibration) != spec["calibration_sha256"]:
        raise ValueError("calibration differs from the frozen model")
    calibration = _read(args.calibration)
    units = profile_units(
        campaign=args.campaign,
        profiles=args.profiles,
        feature_dir=args.features,
        split="evaluation",
    )
    records, summary = evaluate_decisions(
        calibration,
        units,
        best_static_configuration=spec["best_static"]["configuration"],
        holdout_conditions=spec["holdout_conditions"],
        fitted_conditions=spec["fitted_conditions"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    decisions = args.output / "decisions.jsonl"
    decisions.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    summary |= {
        "specification_sha256": args.specification_sha256,
        "decisions_sha256": _sha256(decisions),
        "evaluated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": _commit(),
        "g4": "passed" if summary["gate"]["pass"] else "failed",
    }
    _write(args.output / "evaluation.json", summary)
    selector = summary["policies"]["selector"]
    print(
        json.dumps(
            {
                "g4": summary["g4"],
                "gate": summary["gate"],
                "complete_blocks": summary["complete_blocks"],
                "median_relative_regret_percent": selector[
                    "median_relative_regret_percent"
                ],
                "mean_regret_ms": selector["mean_regret_ms"],
                "top1": selector["top1_agreement_percent"],
            },
            indent=1,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    static = sub.add_parser("best-static")
    static.add_argument("--partitions", type=Path, required=True)
    static.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--calibration", type=Path, required=True)
    freeze.add_argument("--freeze-marker", type=Path, required=True)
    freeze.add_argument("--best-static", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--specification", type=Path, required=True)
    evaluate.add_argument("--specification-sha256", required=True)
    evaluate.add_argument("--calibration", type=Path, required=True)
    evaluate.add_argument("--campaign", type=Path, required=True)
    evaluate.add_argument("--profiles", type=Path, required=True)
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"best-static": best_static_step, "freeze": freeze_step, "evaluate": evaluate_step}[
        args.step
    ](args)


if __name__ == "__main__":
    main()
