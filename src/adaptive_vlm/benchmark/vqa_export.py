"""Export complete VQA predictions for the external official VQA evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.data.manifest import load_frozen_manifest


def export_vqa_predictions(
    manifest_path: Path, run_dir: Path, output: Path, *, repetition: int = 0
) -> dict[str, Any]:
    """Emit official-format answers, counting failures as empty answers."""

    if repetition < 0:
        raise ValueError("repetition must be non-negative")
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise FileExistsError("VQA predictions or metadata already exist")
    manifest = load_frozen_manifest(manifest_path)
    if manifest.source != "vqa-v2-val2014" or any(
        item.scoring != "vqa_consensus" for item in manifest.items
    ):
        raise ValueError("manifest must contain official VQA v2 val2014 requests")
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    if plan["dataset_manifest_sha256"] != manifest.sha256:
        raise ValueError("run plan and VQA manifest do not match")
    expected = {
        entry["trial_id"]: entry
        for entry in plan["schedule"]
        if not entry["warmup"] and entry["repetition"] == repetition
    }
    if len(expected) != len(manifest.items) or {
        entry["request_id"] for entry in expected.values()
    } != {item.request_id for item in manifest.items}:
        raise ValueError("selected repetition does not cover the VQA manifest once")
    rows: dict[str, dict[str, Any]] = {}
    for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["trial_id"] in expected:
            if row["trial_id"] in rows:
                raise ValueError("duplicate VQA trial")
            rows[row["trial_id"]] = row
    if set(rows) != set(expected):
        raise ValueError("missing VQA trial rows")
    by_request: dict[str, dict[str, Any]] = {}
    failures = 0
    for trial_id, entry in expected.items():
        row = rows[trial_id]
        if (
            row.get("plan_sha256") != plan["plan_sha256"]
            or row.get("request_id") != entry["request_id"]
            or row.get("repetition") != repetition
            or row.get("warmup") is not False
            or row.get("dataset_manifest_sha256") != manifest.sha256
        ):
            raise ValueError(f"VQA trial provenance mismatch: {trial_id}")
        if row["status"] != "ok":
            failures += 1
        by_request[entry["request_id"]] = row
    predictions = []
    for item in manifest.items:
        row = by_request[item.request_id]
        predictions.append(
            {
                "question_id": int(item.request_id.removeprefix("vqa-v2-val2014-")),
                "answer": row["output_text"] if row["status"] == "ok" else "",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    metadata = {
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "plan_sha256": plan["plan_sha256"],
        "repetition": repetition,
        "question_count": len(predictions),
        "failed_answers_exported_empty": failures,
        "predictions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "accuracy_metric": "pending_official_vqa_evaluator",
    }
    write_json_atomic(output.with_suffix(".meta.json"), metadata)
    return metadata


def prepare_vqa_eval_subset(
    manifest_path: Path,
    questions_path: Path,
    annotations_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Subset official source JSON so official loadRes sees exactly our IDs."""

    if output_dir.exists():
        raise FileExistsError(f"VQA subset output exists: {output_dir}")
    manifest = load_frozen_manifest(manifest_path)
    if manifest.source != "vqa-v2-val2014":
        raise ValueError("manifest must be a VQA v2 val2014 subset")
    ids = {
        int(item.request_id.removeprefix("vqa-v2-val2014-")) for item in manifest.items
    }
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    selected_questions = [
        question
        for question in questions["questions"]
        if question["question_id"] in ids
    ]
    selected_annotations = [
        annotation
        for annotation in annotations["annotations"]
        if annotation["question_id"] in ids
    ]
    if (
        len(selected_questions) != len(ids)
        or len(selected_annotations) != len(ids)
        or {row["question_id"] for row in selected_questions} != ids
        or {row["question_id"] for row in selected_annotations} != ids
        or any(
            question["image_id"] != annotation["image_id"]
            for question, annotation in zip(
                sorted(selected_questions, key=lambda row: row["question_id"]),
                sorted(selected_annotations, key=lambda row: row["question_id"]),
                strict=True,
            )
        )
    ):
        raise ValueError("official VQA source files do not match frozen subset")
    output_dir.mkdir(parents=True)
    write_json_atomic(
        output_dir / "questions.json",
        {**questions, "questions": selected_questions},
    )
    write_json_atomic(
        output_dir / "annotations.json",
        {**annotations, "annotations": selected_annotations},
    )
    metadata = {
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "questions_source_sha256": hashlib.sha256(
            questions_path.read_bytes()
        ).hexdigest(),
        "annotations_source_sha256": hashlib.sha256(
            annotations_path.read_bytes()
        ).hexdigest(),
        "questions_subset_sha256": hashlib.sha256(
            (output_dir / "questions.json").read_bytes()
        ).hexdigest(),
        "annotations_subset_sha256": hashlib.sha256(
            (output_dir / "annotations.json").read_bytes()
        ).hexdigest(),
        "question_count": len(ids),
    }
    write_json_atomic(output_dir / "metadata.json", metadata)
    return metadata
