"""Deterministic importer for the official VQA v2 validation JSON files."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic, write_jsonl_atomic
from adaptive_vlm.data.manifest import _sha256_file, load_frozen_manifest
from adaptive_vlm.data.vqa_selection import (
    apply_prompt_template,
    image_crop_count,
    load_candidates,
    select_stratified,
)


def import_vqa_v2(
    questions_path: Path,
    annotations_path: Path,
    images_dir: Path,
    output_path: Path,
    *,
    count: int,
    seed: int,
    revision: str,
    candidates_path: Path | None = None,
    two_crop_count: int | None = None,
    prompt_template: str | None = None,
) -> int:
    """Freeze a seeded subset of val2014 questions and ten human answers each.

    Without ``candidates_path`` the subset is ``random.Random(seed).sample`` over
    all paired questions (the G2 v1 method). With it, questions are taken in
    the frozen candidate order, optionally filling a two-crop stratum, and a
    ``.selection.json`` record is written next to the manifest.
    """

    if count < 1 or not revision.strip():
        raise ValueError("count must be positive and revision must be non-empty")
    if two_crop_count is not None and candidates_path is None:
        raise ValueError("crop stratification requires a candidate file")
    if prompt_template is not None:
        apply_prompt_template(prompt_template, "?")
    selection_path = output_path.with_name(f"{output_path.stem}.selection.json")
    if candidates_path is not None and selection_path.exists():
        raise FileExistsError(f"refusing to overwrite selection: {selection_path}")
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
    crops: dict[int, int] = {}
    if candidates_path is None:
        candidates = sorted(question_by_id)
        if count > len(candidates):
            raise ValueError(
                f"requested {count} questions; only {len(candidates)} paired"
            )
        selected = sorted(random.Random(seed).sample(candidates, count))
    else:
        ordered = load_candidates(candidates_path)
        for row in ordered:
            question = question_by_id.get(row["question_id"])
            if question is None or question["image_id"] != row["image_id"]:
                raise ValueError(f"candidate {row['question_id']} not in VQA source")
            crops[row["question_id"]] = image_crop_count(
                images_dir / f"COCO_val2014_{row['image_id']:012d}.jpg"
            )
        selected = sorted(
            select_stratified(
                ordered, crops, count=count, two_crop_count=two_crop_count
            )
        )
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
                "image_path": os.path.relpath(
                    image.resolve(strict=True), output_path.parent.resolve()
                ),
                "image_sha256": _sha256_file(image),
                "prompt": (
                    question["question"]
                    if prompt_template is None
                    else apply_prompt_template(prompt_template, question["question"])
                ),
                "references": answers,
                "scoring": "vqa_consensus",
                "source": "vqa-v2-val2014",
                "revision": revision,
            }
        )
    write_jsonl_atomic(output_path, rows)
    manifest = load_frozen_manifest(output_path, expected_revision=revision)
    if candidates_path is not None:
        write_json_atomic(
            selection_path,
            {
                "schema_version": 1,
                "revision": revision,
                "manifest_sha256": manifest.sha256,
                "candidates_sha256": _sha256_file(candidates_path),
                "questions_sha256": _sha256_file(questions_path),
                "annotations_sha256": _sha256_file(annotations_path),
                "prompt_template": prompt_template,
                "count": count,
                "two_crop_count": two_crop_count,
                "crop_counts": {
                    f"vqa-v2-val2014-{question_id}": crops[question_id]
                    for question_id in selected
                },
            },
        )
    return len(rows)
