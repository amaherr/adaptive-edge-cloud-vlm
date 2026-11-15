from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adaptive_vlm.data.manifest import load_frozen_manifest


def _row(image: Path, *, request_id: str = "one") -> dict[str, object]:
    return {
        "request_id": request_id,
        "image_path": image.name,
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "prompt": "What is shown?",
        "references": ["red", "crimson"],
        "scoring": "exact_match",
        "source": "fixture",
        "revision": "v1",
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_loads_ordered_manifest_and_checks_revision(tmp_path: Path) -> None:
    image = tmp_path / "image.ppm"
    image.write_bytes(b"image")
    path = tmp_path / "manifest.jsonl"
    _write_rows(path, [_row(image, request_id="first"), _row(image, request_id="last")])

    manifest = load_frozen_manifest(path, expected_revision="v1")

    assert [item.request_id for item in manifest.items] == ["first", "last"]
    assert manifest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest.items[0].to_request().image_path == image


def test_detects_changed_image_bytes(tmp_path: Path) -> None:
    image = tmp_path / "image.ppm"
    image.write_bytes(b"original")
    path = tmp_path / "manifest.jsonl"
    _write_rows(path, [_row(image)])
    image.write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_manifest(path)


def test_rejects_duplicate_request_ids(tmp_path: Path) -> None:
    image = tmp_path / "image.ppm"
    image.write_bytes(b"image")
    path = tmp_path / "manifest.jsonl"
    _write_rows(path, [_row(image), _row(image)])

    with pytest.raises(ValueError, match="duplicate request_id"):
        load_frozen_manifest(path)


def test_requires_ten_answers_for_vqa_consensus(tmp_path: Path) -> None:
    image = tmp_path / "image.ppm"
    image.write_bytes(b"image")
    path = tmp_path / "manifest.jsonl"
    row = _row(image)
    row["scoring"] = "vqa_consensus"
    _write_rows(path, [row])

    with pytest.raises(ValueError, match="ten annotator"):
        load_frozen_manifest(path)
