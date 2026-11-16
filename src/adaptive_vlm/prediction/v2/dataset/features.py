"""Pre-execution task features for Predictor V2 (no measured outcomes).

``pinned_task_features`` uses the frozen input bytes, image headers and the
pinned Moondream tokenizer and query template. ``fake_task_features`` applies
the fake adapter's stand-in rules, for rehearsals with the fake model.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path
from typing import Any

from adaptive_vlm.data.manifest import FrozenManifest
from adaptive_vlm.models.fake import fake_image_geometry, fake_prompt_tokens

PINNED_QUERY_TEMPLATE = {"prefix": [1, 15381, 2], "suffix": [3]}


def _frozen_bytes(item: Any) -> bytes:
    raw = item.image_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item.image_sha256:
        raise ValueError(f"frozen image bytes changed: {item.request_id}")
    return bytes(raw)


def query_template(template_source: Path) -> dict[str, list[int]]:
    """The ``query`` template dict literal in Moondream's pinned config.py."""

    template = None
    for node in ast.walk(ast.parse(template_source.read_text())):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "query":
                    template = ast.literal_eval(value)
    if template != PINNED_QUERY_TEMPLATE:
        raise ValueError("unsupported pinned query template")
    return template


def pinned_task_features(
    manifest: FrozenManifest, *, tokenizer: Path, template_source: Path, bucket: str
) -> dict[str, dict[str, Any]]:
    template = query_template(template_source)
    encoder = importlib.import_module("tokenizers").Tokenizer.from_file(str(tokenizer))
    image_module = importlib.import_module("PIL.Image")
    tasks = {}
    for item in manifest.items:
        raw = _frozen_bytes(item)
        with image_module.open(item.image_path) as image:
            width, height = image.size
        # The pinned non-reasoning query appends the suffix a second time.
        prompt_tokens = (
            len(template["prefix"])
            + len(encoder.encode(item.prompt).ids)
            + 2 * len(template["suffix"])
        )
        tasks[item.request_id] = {
            "task_bucket": bucket,
            "image_bytes": len(raw),
            "image_width": width,
            "image_height": height,
            "prompt_tokens": prompt_tokens,
            "requested_max_new_tokens": 32,
        }
    return tasks


def fake_task_features(
    manifest: FrozenManifest, *, bucket: str
) -> dict[str, dict[str, Any]]:
    tasks = {}
    for item in manifest.items:
        raw = _frozen_bytes(item)
        width, height = fake_image_geometry(raw)
        tasks[item.request_id] = {
            "task_bucket": bucket,
            "image_bytes": len(raw),
            "image_width": width,
            "image_height": height,
            "prompt_tokens": fake_prompt_tokens(item.prompt),
            "requested_max_new_tokens": 32,
        }
    return tasks
