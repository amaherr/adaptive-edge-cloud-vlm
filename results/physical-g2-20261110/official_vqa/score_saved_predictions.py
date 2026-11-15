"""Export and officially score all saved G2 predictions without inference.

The upstream code runs unchanged under Python 2.7. This Python 3 orchestrator
uses the repository's existing frozen-manifest exporter and evaluation adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
from adaptive_vlm.data.vqa_export import (  # noqa: E402
    export_vqa_predictions,
    prepare_vqa_eval_subset,
)

PIN = "a013f0043c1e2cdc995922dfe257f7149aa9af06"
PREFIX = "artifacts/unattended/site-preflight-20261110/campaign-heterogeneous-v2"
MANIFEST_SHA = "4e62826280e49359e8cf98d85154000c1583d63474be83a2e9a429da9bde72c9"
ANNOTATIONS_SHA = "0564650e712b34c246e4ad142894692bf2e45dee8e9cc6ebb2cedbe85731c4bf"
QUESTIONS_SHA = "f34d9c9909dea76700d361baddd63dd47b4494945097459b542d9fe840244811"
MODES = ("edge", "cloud", "distributed")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def leading_yes_no(answer):
    """Post-hoc, reference-free diagnostic; never replaces raw primary answers."""
    match = re.match(r"^\s*(yes|no)\b", answer, flags=re.IGNORECASE)
    return match.group(1).lower() if match else answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python2", type=Path, required=True)
    parser.add_argument("--vqa-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("python2", "vqa_root", "annotations", "questions", "work_dir"):
        setattr(args, name, getattr(args, name).resolve())
    commit = subprocess.check_output(
        ["git", "-C", str(args.vqa_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == PIN, "Upstream evaluator commit differs"
    tracked_diff = subprocess.check_output(
        ["git", "-C", str(args.vqa_root), "diff", "HEAD", "--"], text=True
    )
    assert not tracked_diff, "Official evaluator must be unmodified"
    manifest = REPO / "data/manifests/physical_vqa_v1.jsonl"
    assert sha(manifest) == MANIFEST_SHA
    assert sha(args.annotations) == ANNOTATIONS_SHA
    assert sha(args.questions) == QUESTIONS_SHA
    archive = RESULTS / "evidence.tar.gz"
    archive_hash = sha(archive)
    assert archive_hash == (RESULTS / "SHA256SUMS").read_text().split()[0]
    frozen = {
        int(r["request_id"].split("-")[-1]): r
        for r in map(json.loads, manifest.read_text().splitlines())
    }
    subset = args.work_dir / "subset"
    if not subset.exists():
        prepare_vqa_eval_subset(manifest, args.questions, args.annotations, subset)
    subset_metadata = read(subset / "metadata.json")
    for name in ("questions", "annotations"):
        assert sha(subset / f"{name}.json") == subset_metadata[f"{name}_subset_sha256"]
    annotations = read(subset / "annotations.json")["annotations"]
    questions = read(subset / "questions.json")["questions"]
    for q in questions:
        assert q["question"] == frozen[q["question_id"]]["prompt"]
    for a in annotations:
        f = frozen[a["question_id"]]
        assert Counter(x["answer"] for x in a["answers"]) == Counter(f["references"])
        assert f"{a['image_id']:012d}" in f["image_path"]
    dump(HERE / "subset_provenance.json", subset_metadata)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [
        args.vqa_root / "PythonHelperTools/vqaTools/vqa.py",
        args.vqa_root / "PythonEvaluationTools/vqaEvaluation/vqaEval.py",
    ]
    source_hashes = {str(p.relative_to(args.vqa_root)): sha(p) for p in source_paths}
    records = []

    def evaluate(predictions, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            saved = read(output)
            assert saved["input_sha256"]["predictions"] == sha(predictions)
            assert saved["input_sha256"]["annotations"] == sha(
                subset / "annotations.json"
            )
            assert saved["input_sha256"]["questions"] == sha(subset / "questions.json")
            assert saved["evaluator_source_sha256"] == {
                p.name: sha(p) for p in source_paths
            }
            return saved
        cmd = [
            str(args.python2),
            str(REPO / "scripts/vqa/run_official_vqa_eval.py"),
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
        ]
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output.with_suffix(".log").write_text(completed.stdout + completed.stderr)
        return read(output)

    inventory = read(RESULTS / "archive_inventory.json")
    with tarfile.open(archive, "r:gz") as tar:
        for index in range(1, 10):
            block = f"N{index}"
            for mode in MODES:
                run = args.work_dir / "runs" / block / mode
                run.mkdir(parents=True, exist_ok=True)
                for name in ("plan.json", "requests.jsonl"):
                    member = f"{PREFIX}/{block}/{mode}/{name}"
                    payload = tar.extractfile(member).read()
                    assert (
                        hashlib.sha256(payload).hexdigest()
                        == inventory[member]["sha256"]
                    )
                    (run / name).write_bytes(payload)
                for rep in range(3):
                    stem = f"{block}/{mode}-r{rep}"
                    predictions = HERE / "predictions" / f"{stem}.json"
                    metadata_path = predictions.with_suffix(".meta.json")
                    if not predictions.exists():
                        metadata = export_vqa_predictions(
                            manifest, run, predictions, repetition=rep
                        )
                    else:
                        metadata = read(metadata_path)
                        assert (
                            metadata["plan_sha256"]
                            == read(run / "plan.json")["plan_sha256"]
                        )
                        assert metadata["repetition"] == rep
                        assert metadata["predictions_sha256"] == sha(predictions)
                    result_path = HERE / "official_outputs" / f"{stem}.json"
                    result = evaluate(predictions, result_path)
                    records.append(
                        {
                            "block": block,
                            "mode": mode,
                            "repetition": rep,
                            "question_count": result["question_count"],
                            "accuracy_percent": result["accuracy_percent"],
                            "failed_answers_exported_empty": metadata[
                                "failed_answers_exported_empty"
                            ],
                            "predictions_sha256": sha(predictions),
                            "official_output_sha256": sha(result_path),
                        }
                    )
                    print(
                        f"{stem}: {result['accuracy_percent']['overall']}%", flush=True
                    )
    # A clearly separate formatting diagnostic, not the official raw baseline.
    diagnostic = {}
    for mode in MODES:
        raw_predictions = read(HERE / "predictions" / "N1" / f"{mode}-r0.json")
        transformed = [
            {**r, "answer": leading_yes_no(r["answer"])} for r in raw_predictions
        ]
        predictions = HERE / "diagnostic_predictions" / f"{mode}-leading-yes-no.json"
        dump(predictions, transformed)
        result = evaluate(
            predictions, HERE / "diagnostic_outputs" / f"{mode}-leading-yes-no.json"
        )
        diagnostic[mode] = {
            "changed_answers": sum(
                a["answer"] != b["answer"]
                for a, b in zip(raw_predictions, transformed, strict=True)
            ),
            "accuracy_percent": result["accuracy_percent"],
            "scope": (
                "Post-hoc text transform on N1 repetition0, scored by official "
                "evaluator. Not the raw G2 score or a new inference result."
            ),
        }
    for mode in MODES:
        raw_predictions = read(HERE / "predictions" / "N1" / f"{mode}-r0.json")
        lowered = [{**r, "answer": r["answer"].lower()} for r in raw_predictions]
        predictions = HERE / "diagnostic_predictions" / f"{mode}-lowercase-only.json"
        dump(predictions, lowered)
        result = evaluate(
            predictions, HERE / "diagnostic_outputs" / f"{mode}-lowercase-only.json"
        )
        diagnostic[mode]["lowercase_only_diagnostic"] = {
            "accuracy_percent": result["accuracy_percent"],
            "changed_answers": sum(
                a["answer"] != b["answer"]
                for a, b in zip(raw_predictions, lowered, strict=True)
            ),
            "scope": "Lowercase-only post-hoc diagnostic, not raw baseline accuracy",
        }
        subprocess.run(
            [
                str(args.python2),
                str(HERE / "inspect_answer_forms.py"),
                "--vqa-root",
                str(args.vqa_root),
                "--annotations",
                str(subset / "annotations.json"),
                "--predictions",
                str(HERE / "predictions" / "N1" / f"{mode}-r0.json"),
                "--output",
                str(HERE / f"{mode}-answer-forms.json"),
            ],
            check=True,
        )
    provenance = {
        "schema_version": 1,
        "upstream_repository": "https://github.com/GT-Vision-Lab/VQA",
        "upstream_commit": PIN,
        "upstream_tracked_sources_modified": False,
        "source_sha256": source_hashes,
        "adapter_sha256": sha(REPO / "scripts/vqa/run_official_vqa_eval.py"),
        "archive_sha256": archive_hash,
        "manifest_sha256": MANIFEST_SHA,
        "python2_version": subprocess.check_output(
            [str(args.python2), "--version"], stderr=subprocess.STDOUT, text=True
        ).strip(),
        "python2_binary_sha256": sha(args.python2),
        "predictions_postprocessed_for_primary_score": False,
        "official_evaluations": len(records),
        "attempts_evaluated": sum(r["question_count"] for r in records),
        "unique_question_ids": 100,
        "input_alignment_verified": True,
        "exporter_sha256": sha(REPO / "src/adaptive_vlm/data/vqa_export.py"),
        "orchestrator_sha256": sha(Path(__file__)),
        "pypy_download_url": (
            "https://downloads.python.org/pypy/pypy2.7-v7.3.15-linux64.tar.bz2"
        ),
        "pypy_download_sha256": (
            "e857553bdc4f25ba9670a5c173a057a9ff71262d5c5da73a6ddef9d7dc5d4f5e"
        ),
    }
    dump(HERE / "evaluation_index.json", records)
    dump(HERE / "evaluator_provenance.json", provenance)
    dump(HERE / "format_diagnostic.json", diagnostic)


if __name__ == "__main__":
    main()
