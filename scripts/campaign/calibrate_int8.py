"""int8 quality calibration (amendment A2a rule 4; AWS roadmap phase E4).

Run on the edge VM, from the repository root, in the model environment.

  run    for one calibration manifest, run every request through the
         decomposed path twice in one process: vision -> bf16 tensor -> text
         (lossless) and vision -> bf16 -> int8 -> bf16 -> text (int8). Writes
         ``<output>/lossless`` and ``<output>/int8`` in the fixed-run format.
  score  export both P1 runs, score them with the pinned official evaluator
         unchanged and lowercased, report the E1 token agreement, and apply
         the rule. Writes ``decision.json``; exits 2 if int8 is not admitted.

Commit ``results/int8-quality-<date>/`` with the decision before the grid is
fixed. If int8 is not admitted, remove it from every grid (the campaign runs
four configurations).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import append_jsonl, write_json_atomic
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.data.vqa_export import export_vqa_predictions, prepare_vqa_eval_subset
from adaptive_vlm.evaluation.int8_quality import (
    identical_share,
    long_report,
    short_decision,
)

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "data/manifests/prompt_calibration"
CONFIG = ROOT / "configs/experiments/aws_vqa_v2.yaml"
ARTIFACT = ROOT / "configs/model/moondream2_artifact.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(manifest: Any, variant: str) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "int8 quality calibration (A2a rule 4)",
        "variant": variant,
        "dataset_manifest_sha256": manifest.sha256,
        "schedule": [
            {
                "trial_id": f"r0000-{index:06d}",
                "request_id": item.request_id,
                "warmup": False,
                "repetition": 0,
            }
            for index, item in enumerate(manifest.items)
        ],
    }
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return plan


def run(args: argparse.Namespace) -> None:
    from adaptive_vlm.config import load_experiment_config
    from adaptive_vlm.determinism import seed_everything
    from adaptive_vlm.models.moondream.adapter import MoondreamAdapter
    from adaptive_vlm.transport.quantization import dequantize, quantize_int8_rows
    from adaptive_vlm.transport.tensors import TensorInferenceRequest, TensorPayload

    config = load_experiment_config(args.config)
    seed_everything(config.runtime.seed)
    manifest = load_frozen_manifest(args.manifest)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    adapter = MoondreamAdapter.from_pretrained(
        config.model,
        cache_dir=args.cache_dir,
        artifact_manifest=ARTIFACT,
        vision_device=args.device,
        text_device=args.device,
        offline=True,
        weight_residency=args.residency,
    )
    plans = {}
    for variant in ("lossless", "int8"):
        (args.output / variant).mkdir(parents=True)
        plans[variant] = _plan(manifest, variant)
        write_json_atomic(args.output / variant / "plan.json", plans[variant])
    for index, item in enumerate(manifest.items):
        print(f"[{index + 1}/{len(manifest.items)}] {item.request_id}", flush=True)
        request = item.to_request()
        encoding = adapter.encode_projected(request)
        payload = TensorPayload.from_torch(encoding.projected.tensor)
        tensors = {
            "lossless": payload,
            "int8": dequantize(quantize_int8_rows(payload)),
        }
        for variant, tensor in tensors.items():
            result = adapter.generate_from_payload(
                TensorInferenceRequest(item.request_id, item.prompt, tensor)
            )
            append_jsonl(
                args.output / variant / "requests.jsonl",
                {
                    **plans[variant]["schedule"][index],
                    "plan_sha256": plans[variant]["plan_sha256"],
                    "dataset_manifest_sha256": manifest.sha256,
                    "status": "ok",
                    "output_text": result.output_text,
                    "output_token_ids": list(result.output_token_ids),
                },
            )


def _official(
    args: argparse.Namespace, subset: Path, predictions: Path, output: Path
) -> float:
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
            str(output),
        ],
        check=True,
    )
    return float(json.loads(output.read_text())["accuracy_percent"]["overall"])


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text().splitlines()
        if line.strip()
    ]


def score(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    subset = args.output / "subset"
    prepare_vqa_eval_subset(
        args.short_manifest, args.questions, args.annotations, subset
    )
    accuracy: dict[str, dict[str, float]] = {"unchanged": {}, "lowercased": {}}
    answers: dict[str, dict[str, str]] = {}
    files: dict[str, str] = {}
    for variant in ("lossless", "int8"):
        predictions = args.output / f"{variant}.predictions.json"
        export_vqa_predictions(args.short_manifest, args.short / variant, predictions)
        lowered = args.output / f"{variant}.lowercased.predictions.json"
        values = json.loads(predictions.read_text())
        lowered.write_text(
            json.dumps([{**v, "answer": v["answer"].lower()} for v in values]) + "\n"
        )
        answers[variant] = {str(v["question_id"]): v["answer"] for v in values}
        for scoring, path in (("unchanged", predictions), ("lowercased", lowered)):
            official = args.output / f"{variant}.{scoring}.official.json"
            accuracy[scoring][variant] = _official(args, subset, path, official)
            files[official.name] = _sha256(official)
        files[predictions.name] = _sha256(predictions)
    decision = short_decision(
        accuracy, identical_share(answers["lossless"], answers["int8"])
    )
    long = (
        long_report(
            *(
                {
                    row["request_id"]: row["output_token_ids"]
                    for row in _rows(args.long / variant)
                }
                for variant in ("lossless", "int8")
            )
        )
        if args.long
        else None
    )
    record = {
        "schema_version": 1,
        "protocol": "docs/protocols/g2-v2-protocol.md amendment A2a rule 4",
        "short_manifest_sha256": _sha256(args.short_manifest),
        "accuracy_percent": accuracy,
        "short": decision,
        "long": long,
        "files_sha256": files,
    }
    (args.output / "decision.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"admitted": decision["admitted"], **decision["checks"]}))
    if not decision["admitted"]:
        sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, default=CONFIG)
    run_parser.add_argument("--cache-dir", type=Path, default=ROOT / ".model-cache")
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument(
        "--residency", choices=("resident", "swap"), default="resident"
    )
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--short", type=Path, required=True)
    score_parser.add_argument(
        "--short-manifest", type=Path, default=CALIBRATION / "p1.jsonl"
    )
    score_parser.add_argument("--long", type=Path)
    score_parser.add_argument("--questions", type=Path, required=True)
    score_parser.add_argument("--annotations", type=Path, required=True)
    score_parser.add_argument("--vqa-root", type=Path, required=True)
    score_parser.add_argument("--python2", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "run":
        run(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
