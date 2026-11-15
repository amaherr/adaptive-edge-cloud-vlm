"""Exclusion-aware VQA candidate order, crop strata and request splits.

Implements dataset construction for the G2 v2 protocol
(docs/protocols/g2-v2-protocol.md, sections 3.2 and 4).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.models.moondream.geometry import crop_count

QUESTION_PLACEHOLDER = "{question}"
SPLIT_NAMES = ("training", "validation", "evaluation")
_IMAGE_NAME = re.compile(r"COCO_val2014_(\d{12})\.jpg\Z")
_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_id_of(request_id: str) -> int:
    prefix = "vqa-v2-val2014-"
    if not request_id.startswith(prefix):
        raise ValueError(f"not a VQA v2 val2014 request ID: {request_id}")
    return int(request_id.removeprefix(prefix))


def image_id_of(image_path: Path) -> int:
    match = _IMAGE_NAME.fullmatch(image_path.name)
    if match is None:
        raise ValueError(f"not a COCO val2014 image name: {image_path.name}")
    return int(match.group(1))


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` from a baseline or progressive JPEG header."""

    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        raise ValueError(f"not a JPEG file: {path}")
    index = 2
    while index < len(data):
        if data[index] != 0xFF:
            raise ValueError(f"corrupt JPEG marker stream: {path}")
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            continue
        if index + 2 > len(data):
            break
        length = int.from_bytes(data[index : index + 2], "big")
        if length < 2:
            raise ValueError(f"corrupt JPEG segment length: {path}")
        if marker in _SOF_MARKERS:
            if index + 7 > len(data):
                break
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            if height < 1 or width < 1:
                raise ValueError(f"JPEG has no frame size: {path}")
            return height, width
        index += length
    raise ValueError(f"JPEG has no frame header: {path}")


def image_crop_count(path: Path) -> int:
    height, width = jpeg_dimensions(path)
    return crop_count(height, width)


def apply_prompt_template(template: str, question: str) -> str:
    """Insert the verbatim question; no other formatting is interpreted."""

    if template.count(QUESTION_PLACEHOLDER) != 1:
        raise ValueError("prompt template must contain {question} exactly once")
    prompt = template.replace(QUESTION_PLACEHOLDER, question)
    if not prompt.strip():
        raise ValueError("prompt template produced an empty prompt")
    return prompt


def manifest_exclusions(paths: Iterable[Path]) -> tuple[set[int], set[int]]:
    """Question and image IDs used by already frozen VQA manifests."""

    question_ids: set[int] = set()
    image_ids: set[int] = set()
    for path in paths:
        manifest = load_frozen_manifest(path)
        if manifest.source != "vqa-v2-val2014":
            raise ValueError(f"exclusion manifest is not VQA v2 val2014: {path}")
        for item in manifest.items:
            question_ids.add(question_id_of(item.request_id))
            image_ids.add(image_id_of(item.image_path))
    return question_ids, image_ids


def ordered_candidates(
    questions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    count: int,
    excluded_question_ids: set[int],
    excluded_image_ids: set[int],
) -> list[dict[str, int]]:
    """Seeded walk over question IDs with exclusions and one question per image."""

    if count < 1:
        raise ValueError("candidate count must be positive")
    question_by_id = {int(row["question_id"]): row for row in questions}
    if len(question_by_id) != len(questions):
        raise ValueError("duplicate VQA question IDs")
    annotation_by_id = {int(row["question_id"]): row for row in annotations}
    if len(annotation_by_id) != len(annotations):
        raise ValueError("duplicate VQA annotation IDs")
    order = sorted(question_by_id)
    random.Random(seed).shuffle(order)
    used_images: set[int] = set()
    selected: list[dict[str, int]] = []
    for question_id in order:
        image_id = int(question_by_id[question_id]["image_id"])
        if (
            question_id in excluded_question_ids
            or image_id in excluded_image_ids
            or image_id in used_images
        ):
            continue
        annotation = annotation_by_id.get(question_id)
        if annotation is None or len(annotation["answers"]) != 10:
            continue
        if int(annotation["image_id"]) != image_id:
            raise ValueError(f"VQA image ID mismatch for question {question_id}")
        used_images.add(image_id)
        selected.append({"question_id": question_id, "image_id": image_id})
        if len(selected) == count:
            return selected
    raise ValueError(f"only {len(selected)} eligible candidates; requested {count}")


def write_candidates(
    output: Path,
    *,
    questions_path: Path,
    annotations_path: Path,
    exclusion_manifests: Sequence[Path],
    seed: int,
    count: int,
) -> dict[str, Any]:
    """Freeze the candidate order and its provenance as JSON."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidates: {output}")
    excluded_questions, excluded_images = manifest_exclusions(exclusion_manifests)
    candidates = ordered_candidates(
        json.loads(questions_path.read_text(encoding="utf-8"))["questions"],
        json.loads(annotations_path.read_text(encoding="utf-8"))["annotations"],
        seed=seed,
        count=count,
        excluded_question_ids=excluded_questions,
        excluded_image_ids=excluded_images,
    )
    record = {
        "schema_version": 1,
        "seed": seed,
        "count": count,
        "questions_sha256": _sha256(questions_path),
        "annotations_sha256": _sha256(annotations_path),
        "exclusion_manifest_sha256": sorted(
            _sha256(path) for path in exclusion_manifests
        ),
        "excluded_question_count": len(excluded_questions),
        "excluded_image_count": len(excluded_images),
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_candidates(path: Path) -> list[dict[str, int]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("unsupported candidate file schema")
    candidates = [
        {"question_id": int(row["question_id"]), "image_id": int(row["image_id"])}
        for row in record["candidates"]
    ]
    if len({row["question_id"] for row in candidates}) != len(candidates) or len(
        {row["image_id"] for row in candidates}
    ) != len(candidates):
        raise ValueError("candidate file repeats a question or image")
    return candidates


def select_stratified(
    candidates: Sequence[Mapping[str, int]],
    crops: Mapping[int, int],
    *,
    count: int,
    two_crop_count: int | None,
) -> list[int]:
    """First candidates in order filling the two-crop and other strata."""

    if two_crop_count is None:
        if count > len(candidates):
            raise ValueError(f"{len(candidates)} candidates cannot fill {count}")
        return [int(row["question_id"]) for row in candidates[:count]]
    if not 0 <= two_crop_count <= count:
        raise ValueError("two-crop count must be between 0 and the total count")
    quotas = {True: two_crop_count, False: count - two_crop_count}
    selected: list[int] = []
    for row in candidates:
        stratum = crops[int(row["question_id"])] == 2
        if quotas[stratum]:
            quotas[stratum] -= 1
            selected.append(int(row["question_id"]))
        if len(selected) == count:
            return selected
    raise ValueError(
        f"candidates cannot fill the crop strata: {quotas[True]} two-crop and "
        f"{quotas[False]} other requests missing"
    )


def build_request_split(
    manifest_path: Path,
    *,
    counts: Mapping[str, int],
    two_crop_counts: Mapping[str, int],
    seed: int,
) -> dict[str, list[str]]:
    """Assign request IDs to splits within crop strata (protocol section 4.3)."""

    if set(counts) != set(SPLIT_NAMES) or set(two_crop_counts) != set(SPLIT_NAMES):
        raise ValueError(f"split counts must name exactly {SPLIT_NAMES}")
    manifest = load_frozen_manifest(manifest_path)
    if sum(counts.values()) != len(manifest.items):
        raise ValueError("split counts do not cover the manifest exactly once")
    strata: dict[bool, list[str]] = {True: [], False: []}
    for item in manifest.items:
        strata[image_crop_count(item.image_path) == 2].append(item.request_id)
    if sum(two_crop_counts.values()) != len(strata[True]):
        raise ValueError(
            f"manifest has {len(strata[True])} two-crop requests; split asks for "
            f"{sum(two_crop_counts.values())}"
        )
    result: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for two_crop, ids in strata.items():
        ordered = sorted(ids)
        random.Random(seed).shuffle(ordered)
        start = 0
        for name in SPLIT_NAMES:
            size = (
                two_crop_counts[name]
                if two_crop
                else counts[name] - two_crop_counts[name]
            )
            if size < 0:
                raise ValueError(f"{name} two-crop count exceeds its total")
            result[name].extend(ordered[start : start + size])
            start += size
    return {name: sorted(ids) for name, ids in result.items()}
