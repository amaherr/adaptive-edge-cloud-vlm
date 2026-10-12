"""Content-verified JSONL request manifests shared by every policy."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_vlm.contracts import InferenceRequest

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCORING = {"none", "exact_match", "vqa_consensus"}
_FIELDS = {
    "request_id",
    "image_path",
    "image_sha256",
    "prompt",
    "references",
    "scoring",
    "source",
    "revision",
}


@dataclass(frozen=True)
class RequestItem:
    request_id: str
    image_path: Path
    image_sha256: str
    prompt: str
    references: tuple[str, ...]
    scoring: str
    source: str
    revision: str

    def to_request(self) -> InferenceRequest:
        return InferenceRequest(self.request_id, self.image_path, self.prompt)


@dataclass(frozen=True)
class FrozenManifest:
    path: Path
    sha256: str
    source: str
    revision: str
    items: tuple[RequestItem, ...]


def load_frozen_manifest(
    path: str | Path, *, expected_revision: str | None = None
) -> FrozenManifest:
    """Load ordered requests and verify every referenced image byte hash."""

    resolved = Path(path).expanduser().resolve(strict=True)
    content = resolved.read_bytes()
    rows = content.decode("utf-8").splitlines()
    if not rows:
        raise ValueError("request manifest must contain at least one row")

    items: list[RequestItem] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(rows, start=1):
        if not line.strip():
            raise ValueError(f"manifest line {line_number} is blank")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest line {line_number} is invalid JSON") from exc
        item = _parse_item(raw, resolved.parent, line_number)
        if item.request_id in seen_ids:
            raise ValueError(f"duplicate request_id: {item.request_id}")
        seen_ids.add(item.request_id)
        if items and (item.source, item.revision) != (
            items[0].source,
            items[0].revision,
        ):
            raise ValueError("manifest mixes dataset sources or revisions")
        items.append(item)

    if expected_revision is not None and items[0].revision != expected_revision:
        raise ValueError(
            f"manifest revision {items[0].revision!r} differs from "
            f"configured revision {expected_revision!r}"
        )
    return FrozenManifest(
        path=resolved,
        sha256=hashlib.sha256(content).hexdigest(),
        source=items[0].source,
        revision=items[0].revision,
        items=tuple(items),
    )


def _parse_item(raw: Any, base: Path, line_number: int) -> RequestItem:
    if not isinstance(raw, Mapping) or set(raw) != _FIELDS:
        raise ValueError(f"manifest line {line_number} has invalid fields")
    fields = raw
    for name in ("request_id", "image_path", "prompt", "source", "revision"):
        if not isinstance(fields[name], str) or not fields[name].strip():
            raise ValueError(f"manifest line {line_number} has invalid {name}")
    digest = fields["image_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"manifest line {line_number} has invalid image_sha256")
    scoring = fields["scoring"]
    if not isinstance(scoring, str) or scoring not in _SCORING:
        raise ValueError(f"manifest line {line_number} has invalid scoring")
    references = fields["references"]
    if not isinstance(references, list) or any(
        not isinstance(answer, str) or not answer.strip() for answer in references
    ):
        raise ValueError(f"manifest line {line_number} has invalid references")
    if scoring == "none" and references:
        raise ValueError("unscored requests must not contain references")
    if scoring == "exact_match" and not references:
        raise ValueError("exact-match requests require references")
    if scoring == "vqa_consensus" and len(references) != 10:
        raise ValueError("VQA consensus requests require ten annotator answers")

    image_path = Path(fields["image_path"])
    if not image_path.is_absolute():
        image_path = base / image_path
    image_path = image_path.resolve(strict=True)
    if not image_path.is_file():
        raise ValueError(f"manifest image is not a file: {image_path}")
    actual_digest = _sha256_file(image_path)
    if actual_digest != digest:
        raise ValueError(
            f"image SHA-256 mismatch for {fields['request_id']}: "
            f"expected {digest}, got {actual_digest}"
        )
    return RequestItem(
        request_id=fields["request_id"],
        image_path=image_path,
        image_sha256=digest,
        prompt=fields["prompt"],
        references=tuple(references),
        scoring=scoring,
        source=fields["source"],
        revision=fields["revision"],
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
