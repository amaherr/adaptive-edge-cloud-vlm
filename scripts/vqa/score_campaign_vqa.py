"""Official VQA accuracy per configuration for chosen splits of a G2 v2 campaign.

Every measured row of the split's short-task sub-blocks contributes its answer;
each request ID must give one identical answer per configuration (failures
count as empty answers). The answers are scored with the pinned official
evaluator (``run_official_vqa_eval.py`` under a Python 2.7 interpreter): the
primary score lowercases answers (protocol section 9), and the unchanged
answers are scored alongside. Placement parity compares per-question accuracy.

The evaluation split is refused unless ``--unseal-evaluation`` is given
(amendment A3: sealed until the selector is frozen for G4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from adaptive_vlm.data.vqa_export import prepare_vqa_eval_subset

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "vqa-v2-val2014-"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _answers(campaign: Path, splits: set[str]) -> dict:
    answers: dict = {s: defaultdict(lambda: defaultdict(set)) for s in splits}
    for subblock in sorted((campaign / "subblocks").iterdir()):
        plan = json.loads((subblock / "plan.json").read_text())
        split = plan["condition"]["split"]
        if split not in splits or plan["task"] != "short":
            continue
        for run in (p for p in (subblock / "run").iterdir() if p.is_dir()):
            configuration = json.loads((run / "plan.json").read_text())[
                "configuration_id"
            ]
            for line in (run / "requests.jsonl").read_text().splitlines():
                row = json.loads(line)
                if not row["warmup"]:
                    answer = row["output_text"] if row["status"] == "ok" else ""
                    answers[split][configuration][row["request_id"]].add(answer)
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--vqa-root", type=Path, required=True)
    parser.add_argument("--python2", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training", "validation"],
        choices=("training", "validation", "evaluation"),
    )
    parser.add_argument("--unseal-evaluation", action="store_true")
    args = parser.parse_args()
    if "evaluation" in args.splits and not args.unseal_evaluation:
        parser.error("evaluation IDs are sealed; add --unseal-evaluation")

    split_file = json.loads(args.split_file.read_text())
    lines = {}
    for line in args.manifest.read_text().splitlines():
        item = json.loads(line)
        item["image_path"] = str((args.manifest.parent / item["image_path"]).resolve())
        lines[item["request_id"]] = json.dumps(item, sort_keys=True)
    answers = _answers(args.campaign, set(args.splits))
    payload: dict = {
        "schema_version": 1,
        "evaluator": "GT-Vision-Lab/VQA a013f0043c1e2cdc995922dfe257f7149aa9af06",
        "questions_sha256": _sha256(args.questions),
        "annotations_sha256": _sha256(args.annotations),
        "manifest_sha256": _sha256(args.manifest),
        "splits": {},
    }
    for split in args.splits:
        ids = split_file[f"{split}_request_ids"]
        work = args.work_dir / split
        work.mkdir(parents=True, exist_ok=True)
        manifest = work / "manifest.jsonl"
        manifest.write_text("".join(lines[i] + "\n" for i in sorted(ids)))
        subset = work / "subset"
        if not (subset / "questions.json").exists():
            prepare_vqa_eval_subset(manifest, args.questions, args.annotations, subset)
        result: dict = {"request_ids": len(ids), "configurations": {}, "parity": {}}
        per_question = {}
        for configuration, by_id in sorted(answers[split].items()):
            if set(by_id) != set(ids):
                raise ValueError(f"{split} {configuration}: IDs differ from the split")
            varying = sorted(i for i, a in by_id.items() if len(a) != 1)
            if varying:
                raise ValueError(f"{configuration}: answers vary for {varying[:3]}")
            result["configurations"][configuration] = {}
            for variant in ("lowercased", "unchanged"):
                predictions = [
                    {
                        "question_id": int(i.removeprefix(PREFIX)),
                        "answer": next(iter(a)).lower()
                        if variant == "lowercased"
                        else next(iter(a)),
                    }
                    for i, a in sorted(by_id.items())
                ]
                tag = f"{configuration.replace('/', '_')}.{variant}"
                prediction_file = work / f"{tag}.predictions.json"
                official = work / f"{tag}.official.json"
                prediction_file.write_text(json.dumps(predictions))
                official.unlink(missing_ok=True)
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
                        str(prediction_file),
                        "--output",
                        str(official),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                scored = json.loads(official.read_text())
                result["configurations"][configuration][variant] = {
                    "accuracy_percent": scored["accuracy_percent"],
                    "predictions_sha256": _sha256(prediction_file),
                    "official_sha256": _sha256(official),
                }
                if variant == "lowercased":
                    per_question[configuration] = scored[
                        "per_question_accuracy_percent"
                    ]
        for a, b in combinations(sorted(per_question), 2):
            questions = sorted(per_question[a])
            difference = sum(
                per_question[a][q] - per_question[b][q] for q in questions
            ) / len(questions)
            result["parity"][f"{a} vs {b}"] = {
                "accuracy_difference_pp": round(difference, 3),
                "questions_differing": sum(
                    per_question[a][q] != per_question[b][q] for q in questions
                ),
                "within_2pp": abs(difference) <= 2.0,
            }
        payload["splits"][split] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    for split, result in payload["splits"].items():
        print(
            split,
            {
                c: {v: round(x["accuracy_percent"]["overall"], 2) for v, x in d.items()}
                for c, d in result["configurations"].items()
            },
        )
        print(
            " parity within 2 pp:",
            all(p["within_2pp"] for p in result["parity"].values()),
        )


if __name__ == "__main__":
    main()
