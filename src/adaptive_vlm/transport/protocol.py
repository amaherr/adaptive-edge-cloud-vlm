"""Versioned, bounded, checksummed binary framing over stream sockets."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

DEFAULT_MAX_HEADER_BYTES: Final = 64 * 1024
DEFAULT_MAX_PAYLOAD_BYTES: Final = 32 * 1024 * 1024
MAX_REQUEST_ID_BYTES: Final = 128
MAX_PROMPT_CHARACTERS: Final = 16_384
MAX_OUTPUT_TOKENS: Final = 4_096
_LENGTH = struct.Struct("!I")
_HEADER_KEYS = {
    "version",
    "type",
    "request_id",
    "payload_length",
    "payload_sha256",
    "metadata",
}


class MessageType(StrEnum):
    HEALTH_REQUEST = "health_request"
    HEALTH_RESPONSE = "health_response"
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    PROMPT_REQUEST = "prompt_request"
    PROMPT_RESPONSE = "prompt_response"
    TENSOR_REQUEST = "tensor_request"
    ERROR = "error"


class ProtocolError(ValueError):
    """Base class for rejected or malformed protocol data."""


class FrameTooLarge(ProtocolError):
    pass


class ChecksumMismatch(ProtocolError):
    pass


class VersionMismatch(ProtocolError):
    pass


class PeerDisconnected(ConnectionError):
    pass


@dataclass(frozen=True)
class Frame:
    version: int
    message_type: MessageType
    request_id: str
    metadata: Mapping[str, Any]
    payload: bytes = b""

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ProtocolError("protocol version must be positive")
        if not self.request_id or len(self.request_id.encode()) > MAX_REQUEST_ID_BYTES:
            raise ProtocolError("request_id must contain 1 to 128 UTF-8 bytes")
        if not isinstance(self.payload, bytes):
            raise ProtocolError("payload must be bytes")
        _validate_json_mapping(self.metadata, "metadata")
        _validate_message(self)


@dataclass(frozen=True)
class ReceivedFrame:
    frame: Frame
    wire_bytes: int


def encode_frame(
    frame: Frame,
    *,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> bytes:
    """Serialize one safe frame with explicit size and checksum bounds."""

    if len(frame.payload) > max_payload_bytes:
        raise FrameTooLarge("payload exceeds configured maximum")
    header = {
        "version": frame.version,
        "type": frame.message_type.value,
        "request_id": frame.request_id,
        "payload_length": len(frame.payload),
        "payload_sha256": hashlib.sha256(frame.payload).hexdigest(),
        "metadata": dict(frame.metadata),
    }
    encoded_header = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if not encoded_header or len(encoded_header) > max_header_bytes:
        raise FrameTooLarge("header exceeds configured maximum")
    return _LENGTH.pack(len(encoded_header)) + encoded_header + frame.payload


def send_frame(
    connection: Any,
    frame: Frame,
    *,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> int:
    """Send one complete frame and return its exact wire byte count."""

    encoded = encode_frame(
        frame,
        max_header_bytes=max_header_bytes,
        max_payload_bytes=max_payload_bytes,
    )
    connection.sendall(encoded)
    return len(encoded)


def receive_frame(
    connection: Any,
    *,
    expected_version: int,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> ReceivedFrame:
    """Receive exactly one frame from a fragmented stream."""

    prefix = _receive_exact(connection, _LENGTH.size)
    (header_length,) = _LENGTH.unpack(prefix)
    if header_length == 0 or header_length > max_header_bytes:
        raise FrameTooLarge("invalid or oversized header length")

    encoded_header = _receive_exact(connection, header_length)
    try:
        header = json.loads(encoded_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("header is not valid UTF-8 JSON") from exc
    if not isinstance(header, dict) or set(header) != _HEADER_KEYS:
        raise ProtocolError("header fields are invalid")

    version = header["version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ProtocolError("version must be an integer")
    if version != expected_version:
        raise VersionMismatch(
            f"expected protocol version {expected_version}, received {version}"
        )

    payload_length = header["payload_length"]
    if (
        not isinstance(payload_length, int)
        or isinstance(payload_length, bool)
        or payload_length < 0
    ):
        raise ProtocolError("payload_length must be a non-negative integer")
    if payload_length > max_payload_bytes:
        raise FrameTooLarge("payload exceeds configured maximum")

    payload = _receive_exact(connection, payload_length)
    expected_checksum = header["payload_sha256"]
    if not _is_sha256(expected_checksum):
        raise ProtocolError("payload_sha256 must be a lowercase SHA-256 digest")
    actual_checksum = hashlib.sha256(payload).hexdigest()
    if actual_checksum != expected_checksum:
        raise ChecksumMismatch("payload checksum does not match")

    try:
        message_type = MessageType(header["type"])
    except (TypeError, ValueError) as exc:
        raise ProtocolError("unknown message type") from exc
    metadata = header["metadata"]
    if not isinstance(metadata, Mapping):
        raise ProtocolError("metadata must be an object")
    request_id = header["request_id"]
    if not isinstance(request_id, str):
        raise ProtocolError("request_id must be a string")

    frame = Frame(
        version=version,
        message_type=message_type,
        request_id=request_id,
        metadata=metadata,
        payload=payload,
    )
    return ReceivedFrame(
        frame=frame,
        wire_bytes=_LENGTH.size + header_length + payload_length,
    )


def _receive_exact(connection: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise PeerDisconnected(
                f"peer disconnected with {remaining} frame bytes outstanding"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_message(frame: Frame) -> None:
    metadata = frame.metadata
    if frame.message_type is MessageType.HEALTH_REQUEST:
        _require_keys(metadata, set())
        _require_empty_payload(frame)
    elif frame.message_type is MessageType.HEALTH_RESPONSE:
        _require_keys(metadata, {"ready", "model_fingerprint"})
        _require_type(metadata, "ready", bool)
        _require_fingerprint(metadata)
        _require_empty_payload(frame)
    elif frame.message_type is MessageType.INFERENCE_REQUEST:
        _require_keys(metadata, {"prompt", "model_fingerprint", "image_suffix"})
        prompt = _require_type(metadata, "prompt", str)
        if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ProtocolError("prompt is empty or too long")
        suffix = _require_type(metadata, "image_suffix", str)
        if suffix and re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix) is None:
            raise ProtocolError("image_suffix is invalid")
        _require_fingerprint(metadata)
        if not frame.payload:
            raise ProtocolError("inference request image payload is empty")
    elif frame.message_type is MessageType.INFERENCE_RESPONSE:
        _require_keys(
            metadata,
            {"model_fingerprint", "output_text", "output_token_ids", "metrics"},
        )
        _require_fingerprint(metadata)
        _require_type(metadata, "output_text", str)
        token_ids = _require_type(metadata, "output_token_ids", list)
        if len(token_ids) > MAX_OUTPUT_TOKENS or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in token_ids
        ):
            raise ProtocolError("output_token_ids are invalid")
        metrics = _require_type(metadata, "metrics", Mapping)
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, (str, int, float, bool, type(None)))
            for key, value in metrics.items()
        ):
            raise ProtocolError("response metrics are invalid")
        _require_empty_payload(frame)
    elif frame.message_type is MessageType.PROMPT_REQUEST:
        _require_keys(metadata, {"prompt", "model_fingerprint"})
        prompt = _require_type(metadata, "prompt", str)
        if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ProtocolError("prompt is empty or too long")
        _require_fingerprint(metadata)
        _require_empty_payload(frame)
    elif frame.message_type is MessageType.PROMPT_RESPONSE:
        _require_keys(metadata, {"accepted", "model_fingerprint"})
        _require_type(metadata, "accepted", bool)
        _require_fingerprint(metadata)
        _require_empty_payload(frame)
    elif frame.message_type is MessageType.TENSOR_REQUEST:
        _validate_tensor_request(frame)
    elif frame.message_type is MessageType.ERROR:
        _require_keys(metadata, {"code", "message", "retryable"})
        _require_type(metadata, "code", str)
        message = _require_type(metadata, "message", str)
        if not message or len(message) > 2_048:
            raise ProtocolError("error message is empty or too long")
        _require_type(metadata, "retryable", bool)
        _require_empty_payload(frame)


def _validate_tensor_request(frame: Frame) -> None:
    metadata = frame.metadata
    _require_keys(
        metadata,
        {
            "prompt",
            "model_fingerprint",
            "shape",
            "dtype",
            "byte_order",
            "layout",
            "nbytes",
            "pipeline",
        },
    )
    prompt = _require_type(metadata, "prompt", str)
    if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARACTERS:
        raise ProtocolError("prompt is empty or too long")
    _require_fingerprint(metadata)
    pipeline = _require_type(metadata, "pipeline", str)
    if pipeline not in {"synchronous", "prompt_first"}:
        raise ProtocolError("tensor pipeline is unsupported")
    shape = _require_type(metadata, "shape", list)
    if not 1 <= len(shape) <= 8 or any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in shape
    ):
        raise ProtocolError("tensor shape is invalid")
    dtype = _require_type(metadata, "dtype", str)
    dtype_sizes = {"bfloat16": 2, "float16": 2, "float32": 4}
    if dtype not in dtype_sizes:
        raise ProtocolError("tensor dtype is unsupported")
    byte_order = _require_type(metadata, "byte_order", str)
    if byte_order not in {"little", "big"}:
        raise ProtocolError("tensor byte_order is unsupported")
    layout = _require_type(metadata, "layout", str)
    if layout != "contiguous":
        raise ProtocolError("tensor layout is unsupported")
    nbytes = _require_type(metadata, "nbytes", int)
    if isinstance(nbytes, bool) or nbytes != len(frame.payload):
        raise ProtocolError("tensor nbytes does not match payload")
    if nbytes != math.prod(shape) * dtype_sizes[dtype]:
        raise ProtocolError("tensor shape, dtype, and nbytes are inconsistent")
    if not frame.payload:
        raise ProtocolError("tensor payload is empty")


def _require_keys(metadata: Mapping[str, Any], expected: set[str]) -> None:
    if set(metadata) != expected:
        raise ProtocolError("message metadata fields are invalid")


def _require_type(
    metadata: Mapping[str, Any], key: str, expected_type: type[Any]
) -> Any:
    value = metadata.get(key)
    if not isinstance(value, expected_type):
        raise ProtocolError(f"metadata field {key!r} has an invalid type")
    return value


def _require_fingerprint(metadata: Mapping[str, Any]) -> None:
    if not _is_sha256(metadata.get("model_fingerprint")):
        raise ProtocolError("model_fingerprint must be a lowercase SHA-256 digest")


def _require_empty_payload(frame: Frame) -> None:
    if frame.payload:
        raise ProtocolError(f"{frame.message_type.value} payload must be empty")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_json_mapping(value: Mapping[str, Any], field: str) -> None:
    if not all(isinstance(key, str) for key in value):
        raise ProtocolError(f"{field} keys must be strings")
    _validate_json_value(value, field, depth=0)


def _validate_json_value(value: Any, field: str, *, depth: int) -> None:
    if depth > 8:
        raise ProtocolError(f"{field} nesting exceeds maximum depth")
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError(f"{field} contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError(f"{field} keys must be strings")
            _validate_json_value(item, field, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item, field, depth=depth + 1)
        return
    raise ProtocolError(f"{field} contains a non-JSON value")
