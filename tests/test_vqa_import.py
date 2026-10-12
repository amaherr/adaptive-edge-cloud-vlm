"""Small official-schema fixtures for the VQA importer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.data.vqa_import import import_vqa_v2


def test_import_vqa_subset_is_deterministic_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    questions = tmp_path / "questions.json"
    annotations = tmp_path / "annotations.json"
    question_rows = []
    annotation_rows = []
    for question_id in (3, 1, 2):
        (images / f"COCO_val2014_{question_id:012d}.jpg").write_bytes(
            bytes([question_id])
        )
        question_rows.append(
            {"question_id": question_id, "image_id": question_id, "question": "What?"}
        )
        annotation_rows.append(
            {
                "question_id": question_id,
                "image_id": question_id,
                "answers": [{"answer": "cat"}] * 10,
            }
        )
    questions.write_text(json.dumps({"questions": question_rows}), encoding="utf-8")
    annotations.write_text(
        json.dumps({"annotations": annotation_rows}), encoding="utf-8"
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    for output in (first, second):
        assert (
            import_vqa_v2(
                questions,
                annotations,
                images,
                output,
                count=2,
                seed=42,
                revision="vqa-v2-val2014-test",
            )
            == 2
        )
    assert first.read_bytes() == second.read_bytes()
    manifest = load_frozen_manifest(first, expected_revision="vqa-v2-val2014-test")
    assert len(manifest.items) == 2
    assert all(len(item.references) == 10 for item in manifest.items)
    with pytest.raises(FileExistsError):
        import_vqa_v2(
            questions,
            annotations,
            images,
            first,
            count=2,
            seed=42,
            revision="vqa-v2-val2014-test",
        )


def test_import_vqa_rejects_missing_annotation(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    annotations = tmp_path / "annotations.json"
    questions.write_text(
        json.dumps({"questions": [{"question_id": 1, "image_id": 1,
                                    "question": "What?"}]}), encoding="utf-8"
    )
    annotations.write_text(json.dumps({"annotations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="different IDs"):
        import_vqa_v2(
            questions, annotations, tmp_path, tmp_path / "out.jsonl",
            count=1, seed=42, revision="test",
        )
