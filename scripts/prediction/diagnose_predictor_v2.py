"""Attribute saved Predictor V2 training OOF/LOBO errors; no fitting/evaluation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.prediction.v2.dataset.parsing import digest
from adaptive_vlm.prediction.v2.diagnosis.report import diagnose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-bundle", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit diagnostic source before the reproducible run")
    report = diagnose(args.data_bundle, args.fit_dir, args.output_dir)
    write_json_atomic(
        args.output_dir / "source_manifest.json",
        {
            "code_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_dirty": False,
            "training_rows_sha256": digest(args.data_bundle / "training.jsonl"),
            "prediction_file_hashes": {
                name: digest(args.fit_dir / name)
                for name in ["training_oof.jsonl", "training_lobo.jsonl"]
            },
            "validation_outcomes_read": False,
            "final_outcomes_read": False,
            "fits_modified": False,
        },
    )
    write_json_atomic(
        args.output_dir / "file_hashes.json",
        {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()},
    )
    print(
        f"Diagnosed {report['training_attempts']} training attempts; "
        "no fitting or held-out outcome access"
    )


if __name__ == "__main__":
    main()
