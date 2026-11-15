"""Strict JSON parsing and routing-metadata extraction for V2 data."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROUTING_KEYS = frozenset(
    {
        "schema_version",
        "request_id",
        "configuration_id",
        "network_state",
        "warmup",
        "repetition",
        "trial_id",
        "measured_at_utc",
        "host_pair_id",
        "model_contract_sha256",
        "dataset_manifest_sha256",
        "placement_profile",
        "plan_sha256",
        "requested_max_new_tokens",
        "task_bucket",
        "git_commit",
        "git_dirty",
    }
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return result


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(data: bytes | str) -> dict[str, Any]:
    value = json.loads(data, object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _skip_value(text: str, position: int) -> int:
    """Skip opaque outcome values without decoding their numbers/objects."""
    start = position
    if text[position] in "{[":
        stack: list[str] = []
        quoted = escaped = False
        while position < len(text):
            char = text[position]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack or (stack.pop(), char) not in {("{", "}"), ("[", "]")}:
                    raise ValueError("malformed opaque JSON value")
                if not stack:
                    return position + 1
            position += 1
    elif text[position] == '"':
        position += 1
        escaped = False
        while position < len(text):
            char = text[position]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return position + 1
            position += 1
    else:
        while position < len(text) and text[position] not in ",} \t\r\n":
            position += 1
        if position > start:
            return position
    raise ValueError("unterminated JSON value")


def routing_metadata(line: bytes) -> dict[str, Any]:
    """Decode only allowlisted envelope keys, never fields/raw_metrics/outcomes."""
    text = line.decode("utf-8").strip()
    if not text.startswith("{"):
        raise ValueError("profile row must be an object")
    position = 1
    result: dict[str, Any] = {}
    seen: set[str] = set()
    decoder = json.JSONDecoder(object_pairs_hook=_pairs)
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            raise ValueError("unterminated profile envelope")
        if text[position] == "}":
            if text[position + 1 :].strip():
                raise ValueError("trailing profile bytes")
            return result
        key, position = decoder.raw_decode(text, position)
        if not isinstance(key, str) or key in seen:
            raise ValueError("invalid or duplicate envelope key")
        seen.add(key)
        while position < len(text) and text[position].isspace():
            position += 1
        if text[position] != ":":
            raise ValueError("missing JSON colon")
        position += 1
        while text[position].isspace():
            position += 1
        if key in ROUTING_KEYS:
            result[key], position = decoder.raw_decode(text, position)
        else:
            position = _skip_value(text, position)
        while position < len(text) and text[position].isspace():
            position += 1
        if text[position] == ",":
            position += 1
        elif text[position] != "}":
            raise ValueError("missing JSON delimiter")
