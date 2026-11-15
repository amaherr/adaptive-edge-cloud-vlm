"""Candidate ordering, crop strata, prompt templates and request splits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.data.vqa_import import import_vqa_v2
from adaptive_vlm.data.vqa_selection import (
    apply_prompt_template,
    build_request_split,
    image_crop_count,
    jpeg_dimensions,
    load_candidates,
    manifest_exclusions,
    ordered_candidates,
    select_stratified,
    write_candidates,
)


def jpeg(height: int, width: int, *, progressive: bool = False) -> bytes:
    app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + bytes(9)
    marker = b"\xff\xc2" if progressive else b"\xff\xc0"
    frame = (
        marker
        + (17).to_bytes(2, "big")
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03"
        + bytes(9)
    )
    return b"\xff\xd8" + app0 + b"\xff\xff" + frame + b"\xff\xd9"


def write_source(
    root: Path,
    sizes: dict[int, tuple[int, int]],
    *,
    short: frozenset[int] = frozenset(),
) -> tuple[Path, Path, Path]:
    """Questions 10*k on image k; ``short`` questions have nine answers."""

    images = root / "images"
    images.mkdir()
    questions = []
    annotations = []
    for image_id, (height, width) in sizes.items():
        (images / f"COCO_val2014_{image_id:012d}.jpg").write_bytes(jpeg(height, width))
        for offset in range(2):
            question_id = image_id * 10 + offset
            questions.append(
                {
                    "question_id": question_id,
                    "image_id": image_id,
                    "question": f"What is {question_id}?",
                }
            )
            annotations.append(
                {
                    "question_id": question_id,
                    "image_id": image_id,
                    "answers": [{"answer": "cat"}]
                    * (9 if question_id in short else 10),
                }
            )
    questions_path = root / "questions.json"
    annotations_path = root / "annotations.json"
    questions_path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    annotations_path.write_text(
        json.dumps({"annotations": annotations}), encoding="utf-8"
    )
    return questions_path, annotations_path, images


def test_jpeg_dimensions_and_crop_strata(tmp_path: Path) -> None:
    baseline = tmp_path / "a.jpg"
    baseline.write_bytes(jpeg(480, 640))
    progressive = tmp_path / "b.jpg"
    progressive.write_bytes(jpeg(375, 500, progressive=True))
    assert jpeg_dimensions(baseline) == (480, 640)
    assert jpeg_dimensions(progressive) == (375, 500)
    assert image_crop_count(baseline) > 2
    assert image_crop_count(progressive) == 2
    broken = tmp_path / "c.jpg"
    broken.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="not a JPEG"):
        jpeg_dimensions(broken)
    headless = tmp_path / "d.jpg"
    headless.write_bytes(b"\xff\xd8\xff\xd9")
    with pytest.raises(ValueError, match="no frame header"):
        jpeg_dimensions(headless)


def test_prompt_template_inserts_question_verbatim() -> None:
    template = "{question} Answer with a single word or phrase."
    assert (
        apply_prompt_template(template, "Is it {red}?")
        == "Is it {red}? Answer with a single word or phrase."
    )
    for bad in ("No placeholder.", "{question} {question}"):
        with pytest.raises(ValueError, match="exactly once"):
            apply_prompt_template(bad, "Why?")


def test_candidates_exclude_ids_images_repeats_and_short_annotations() -> None:
    questions = [
        {"question_id": q, "image_id": q // 10, "question": "?"}
        for q in (10, 11, 20, 21, 30, 40, 50)
    ]
    annotations = [
        {
            "question_id": row["question_id"],
            "image_id": row["image_id"],
            "answers": [{"answer": "x"}] * (9 if row["question_id"] == 40 else 10),
        }
        for row in questions
    ]
    kwargs = {
        "seed": 7,
        "excluded_question_ids": {50},
        "excluded_image_ids": {3},
    }
    # Image 3 and question 50 are excluded and question 40 (image 4) has nine
    # answers, so only images 1 and 2 remain, each at most once.
    result = ordered_candidates(questions, annotations, count=2, **kwargs)
    assert {row["image_id"] for row in result} == {1, 2}
    assert result == ordered_candidates(questions, annotations, count=2, **kwargs)
    with pytest.raises(ValueError, match="only 2 eligible"):
        ordered_candidates(questions, annotations, count=3, **kwargs)
    with pytest.raises(ValueError, match="positive"):
        ordered_candidates(questions, annotations, count=0, **kwargs)


def test_select_stratified_fills_quotas_in_candidate_order() -> None:
    candidates = [{"question_id": q, "image_id": q} for q in range(1, 9)]
    crops = {1: 9, 2: 2, 3: 9, 4: 9, 5: 2, 6: 10, 7: 2, 8: 9}
    assert select_stratified(candidates, crops, count=4, two_crop_count=2) == [
        1,
        2,
        3,
        5,
    ]
    assert select_stratified(candidates, crops, count=3, two_crop_count=None) == [
        1,
        2,
        3,
    ]
    with pytest.raises(ValueError, match="1 two-crop"):
        select_stratified(candidates, crops, count=5, two_crop_count=4)
    with pytest.raises(ValueError, match="between 0"):
        select_stratified(candidates, crops, count=2, two_crop_count=3)


def test_stratified_import_and_split_end_to_end(tmp_path: Path) -> None:
    sizes = {
        image_id: ((375, 500) if image_id % 3 == 0 else (480, 640))
        for image_id in range(1, 31)
    }
    questions, annotations, images = write_source(tmp_path, sizes)
    excluded = tmp_path / "excluded.jsonl"
    import_vqa_v2(
        questions,
        annotations,
        images,
        excluded,
        count=4,
        seed=1,
        revision="vqa-old",
    )
    old_questions, old_images = manifest_exclusions([excluded])
    assert len(old_questions) == 4 and old_images
    candidates_path = tmp_path / "candidates.json"
    record = write_candidates(
        candidates_path,
        questions_path=questions,
        annotations_path=annotations,
        exclusion_manifests=[excluded],
        seed=3,
        count=20,
    )
    assert record["excluded_question_count"] == 4
    candidates = load_candidates(candidates_path)
    assert not {row["image_id"] for row in candidates} & old_images
    with pytest.raises(FileExistsError):
        write_candidates(
            candidates_path,
            questions_path=questions,
            annotations_path=annotations,
            exclusion_manifests=[],
            seed=3,
            count=20,
        )
    manifest_path = tmp_path / "new.jsonl"
    template = "{question} Answer with a single word or phrase."
    assert (
        import_vqa_v2(
            questions,
            annotations,
            images,
            manifest_path,
            count=12,
            seed=3,
            revision="vqa-new",
            candidates_path=candidates_path,
            two_crop_count=3,
            prompt_template=template,
        )
        == 12
    )
    manifest = load_frozen_manifest(manifest_path, expected_revision="vqa-new")
    assert all(item.prompt.endswith("word or phrase.") for item in manifest.items)
    selection = json.loads((tmp_path / "new.selection.json").read_text())
    assert selection["manifest_sha256"] == manifest.sha256
    assert sorted(selection["crop_counts"].values()).count(2) == 3
    assert selection["prompt_template"] == template
    split = build_request_split(
        manifest_path,
        counts={"training": 6, "validation": 3, "evaluation": 3},
        two_crop_counts={"training": 1, "validation": 1, "evaluation": 1},
        seed=5,
    )
    assert sorted(rid for ids in split.values() for rid in ids) == sorted(
        item.request_id for item in manifest.items
    )
    for name, size in (("training", 6), ("validation", 3), ("evaluation", 3)):
        assert len(split[name]) == size
        assert sum(selection["crop_counts"][rid] == 2 for rid in split[name]) == 1
    with pytest.raises(ValueError, match="has 3 two-crop"):
        build_request_split(
            manifest_path,
            counts={"training": 6, "validation": 3, "evaluation": 3},
            two_crop_counts={"training": 2, "validation": 1, "evaluation": 1},
            seed=5,
        )


def test_import_rejects_strata_without_candidates(tmp_path: Path) -> None:
    questions, annotations, images = write_source(tmp_path, {1: (480, 640)})
    with pytest.raises(ValueError, match="requires a candidate file"):
        import_vqa_v2(
            questions,
            annotations,
            images,
            tmp_path / "x.jsonl",
            count=1,
            seed=1,
            revision="r",
            two_crop_count=1,
        )
