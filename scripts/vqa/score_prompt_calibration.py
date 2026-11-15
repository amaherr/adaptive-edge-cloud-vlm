"""Score the G2 v2 prompt calibration runs and apply the frozen selection rule.

Implements docs/protocols/g2-v2-protocol.md sections 3.3 and 3.4: export each
template's saved answers unchanged, score them with the pinned official VQA
evaluator under Python 2.7, and select a template by the predeclared rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.data.vqa_export import export_vqa_predictions, prepare_vqa_eval_subset

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = {"p0": False, "p1": True, "p2": True, "p3": True}
NEAR_BEST_PP = 1.0
FLOOR_PERCENT = 50.0
TOKEN_CAP = 32


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_metrics(run_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text().splitlines()
        if line.strip()
    ]
    measured = [row for row in rows if not row["warmup"]]
    ok = [row for row in measured if row["status"] == "ok"]
    tokens = [len(row["output_token_ids"]) for row in ok]
    return {
        "measured_requests": len(measured),
        "failed_requests": len(measured) - len(ok),
        "mean_output_tokens": mean(tokens) if tokens else None,
        "token_cap_hits": sum(count >= TOKEN_CAP for count in tokens),
        "median_end_to_end_ms": (
            median(row["observed_end_to_end_ms"] for row in ok) if ok else None
        ),
    }


def select(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = {key: value for key, value in results.items() if TEMPLATES[key]}
    best = max(value["accuracy_percent"] for value in eligible.values())
    if best < FLOOR_PERCENT:
        return {
            "selected": None,
            "best_eligible_accuracy_percent": best,
            "reason": f"best eligible accuracy {best}% is below the "
            f"{FLOOR_PERCENT}% floor; amend the protocol",
        }
    near = [
        key
        for key, value in eligible.items()
        if value["accuracy_percent"] >= best - NEAR_BEST_PP
    ]
    chosen = min(
        near,
        key=lambda key: (
            results[key]["token_cap_hits"],
            results[key]["mean_output_tokens"],
            key,
        ),
    )
    return {
        "selected": chosen,
        "best_eligible_accuracy_percent": best,
        "near_best": sorted(near),
        "reason": "fewest token-cap hits, then lowest mean output tokens, then "
        "lowest template ID among templates within "
        f"{NEAR_BEST_PP} pp of the best eligible accuracy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--vqa-root", type=Path, required=True)
    parser.add_argument("--python2", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.work_dir.exists():
        parser.error("refusing to overwrite an existing output or work directory")
    args.work_dir.mkdir(parents=True)
    args.output.mkdir(parents=True)
    question_ids = None
    subset = args.work_dir / "subset"
    results: dict[str, dict[str, Any]] = {}
    for key in TEMPLATES:
        manifest_path = args.manifests / f"{key}.jsonl"
        manifest = load_frozen_manifest(manifest_path)
        ids = sorted(item.request_id for item in manifest.items)
        if question_ids is None:
            question_ids = ids
            prepare_vqa_eval_subset(
                manifest_path, args.questions, args.annotations, subset
            )
        elif ids != question_ids:
            raise ValueError("calibration manifests cover different questions")
        predictions = args.output / f"{key}.predictions.json"
        export = export_vqa_predictions(manifest_path, args.runs / key, predictions)
        official = args.output / f"{key}.official.json"
        subprocess.run(
            [
                str(args.python2),
                str(ROOT / "scripts/vqa/run_official_vqa_eval.py"),
                "--vqa-root",
                str(args.vqa_root),
                "--questions",
                str(subset / "questions.json"),
                "--annotations",
                str(subset / "annotations.json"),
                "--predictions",
                str(predictions),
                "--output",
                str(official),
            ],
            check=True,
        )
        scored = json.loads(official.read_text())
        results[key] = {
            "eligible": TEMPLATES[key],
            "prompt_template": json.loads(
                (args.manifests / f"{key}.selection.json").read_text()
            )["prompt_template"],
            "manifest_sha256": manifest.sha256,
            "predictions_sha256": export["predictions_sha256"],
            "official_result_sha256": sha256(official),
            "accuracy_percent": scored["accuracy_percent"]["overall"],
            "accuracy_by_answer_type_percent": scored["accuracy_percent"][
                "perAnswerType"
            ],
            **run_metrics(args.runs / key),
        }
    decision = select(results)
    record = {
        "schema_version": 1,
        "protocol": "docs/protocols/g2-v2-protocol.md sections 3.3-3.4",
        "question_count": len(question_ids or []),
        "subset_metadata": json.loads((subset / "metadata.json").read_text()),
        "rule": {
            "near_best_pp": NEAR_BEST_PP,
            "floor_percent": FLOOR_PERCENT,
            "token_cap": TOKEN_CAP,
        },
        "templates": results,
        "decision": decision,
    }
    (args.output / "selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2))
    if decision["selected"] is None:
        sys.exit(2)


if __name__ == "__main__":
    main()
