"""Deterministic importer for the official VQA v2 validation JSON files."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_jsonl_atomic
from adaptive_vlm.data.manifest import _sha256_file, load_frozen_manifest


def import_vqa_v2(
    questions_path: Path,
    annotations_path: Path,
    images_dir: Path,
    output_path: Path,
    *,
    count: int,
    seed: int,
    revision: str,
) -> int:
    """Freeze a seeded subset of val2014 questions and ten human answers each."""

    if count < 1 or not revision.strip():
        raise ValueError("count must be positive and revision must be non-empty")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output_path}")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))[
        "annotations"
    ]
    if not isinstance(questions, list) or not isinstance(annotations, list):
        raise ValueError("VQA questions and annotations must be lists")
    answer_by_id: dict[int, dict[str, Any]] = {}
    for annotation in annotations:
        question_id = annotation["question_id"]
        if question_id in answer_by_id:
            raise ValueError(f"duplicate VQA annotation: {question_id}")
        answer_by_id[question_id] = annotation
    question_by_id: dict[int, dict[str, Any]] = {}
    for question in questions:
        question_id = question["question_id"]
        if question_id in question_by_id:
            raise ValueError(f"duplicate VQA question: {question_id}")
        question_by_id[question_id] = question
    if set(question_by_id) != set(answer_by_id):
        raise ValueError("VQA questions and annotations contain different IDs")
    candidates = sorted(question_by_id)
    if count > len(candidates):
        raise ValueError(f"requested {count} questions; only {len(candidates)} paired")
    selected = sorted(random.Random(seed).sample(candidates, count))
    rows: list[dict[str, Any]] = []
    for question_id in selected:
        question = question_by_id[question_id]
        annotation = answer_by_id[question_id]
        image_id = question["image_id"]
        if image_id != annotation["image_id"]:
            raise ValueError(f"VQA image ID mismatch for question {question_id}")
        answers = [entry["answer"] for entry in annotation["answers"]]
        if len(answers) != 10:
            raise ValueError(
                f"question {question_id} has {len(answers)} answers, not 10"
            )
        image = images_dir / f"COCO_val2014_{image_id:012d}.jpg"
        rows.append(
            {
                "request_id": f"vqa-v2-val2014-{question_id}",
                "image_path": str(image.resolve(strict=True)),
                "image_sha256": _sha256_file(image),
                "prompt": question["question"],
                "references": answers,
                "scoring": "vqa_consensus",
                "source": "vqa-v2-val2014",
                "revision": revision,
            }
        )
    write_jsonl_atomic(output_path, rows)
    load_frozen_manifest(output_path, expected_revision=revision)
    return len(rows)
