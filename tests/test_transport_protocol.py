from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import pytest

from adaptive_vlm.transport.protocol import (
    ChecksumMismatch,
    Frame,
    FrameTooLarge,
    MessageType,
    PeerDisconnected,
    ProtocolError,
    VersionMismatch,
    encode_frame,
    receive_frame,
    send_frame,
)

FINGERPRINT = "a" * 64


@dataclass
class MemoryConnection:
    """Minimal stream double supporting fragmented reads and injected errors."""

    incoming: bytearray = field(default_factory=bytearray)
    fragment_size: int | None = None
    receive_error: Exception | None = None
    sent: bytearray = field(default_factory=bytearray)

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    def recv(self, size: int) -> bytes:
        if self.receive_error is not None:
            raise self.receive_error
        if not self.incoming:
            return b""
        count = min(size, len(self.incoming))
        if self.fragment_size is not None:
            count = min(count, self.fragment_size)
        result = bytes(self.incoming[:count])
        del self.incoming[:count]
        return result


def _connection_for(
    data: bytes, *, fragment_size: int | None = None
) -> MemoryConnection:
    return MemoryConnection(bytearray(data), fragment_size=fragment_size)


def _inference_frame(payload: bytes = b"image") -> Frame:
    return Frame(
        version=1,
        message_type=MessageType.INFERENCE_REQUEST,
        request_id="request-1",
        metadata={
            "prompt": "Describe the image.",
            "model_fingerprint": FINGERPRINT,
            "image_suffix": ".png",
        },
        payload=payload,
    )


def test_fragmented_frame_round_trip() -> None:
    encoded = encode_frame(_inference_frame())
    received = receive_frame(
        _connection_for(encoded, fragment_size=1), expected_version=1
    )

    assert received.frame == _inference_frame()
    assert received.wire_bytes == len(encoded)


def test_send_frame_reports_exact_wire_bytes() -> None:
    sender = MemoryConnection()
    sent = send_frame(sender, _inference_frame())
    received = receive_frame(_connection_for(sender.sent), expected_version=1)

    assert sent == received.wire_bytes


def test_checksum_mismatch_is_rejected() -> None:
    encoded = bytearray(encode_frame(_inference_frame()))
    encoded[-1] ^= 1

    with pytest.raises(ChecksumMismatch):
        receive_frame(_connection_for(encoded), expected_version=1)


def test_oversized_header_is_rejected_before_allocation() -> None:
    with pytest.raises(FrameTooLarge, match="header"):
        receive_frame(_connection_for(struct.pack("!I", 65_537)), expected_version=1)


def test_oversized_declared_payload_is_rejected() -> None:
    header = {
        "version": 1,
        "type": "inference_request",
        "request_id": "request-1",
        "payload_length": 11,
        "payload_sha256": "0" * 64,
        "metadata": {
            "prompt": "prompt",
            "model_fingerprint": FINGERPRINT,
            "image_suffix": ".png",
        },
    }
    encoded_header = json.dumps(header).encode()
    connection = _connection_for(
        struct.pack("!I", len(encoded_header)) + encoded_header
    )

    with pytest.raises(FrameTooLarge, match="payload"):
        receive_frame(connection, expected_version=1, max_payload_bytes=10)


def test_timeout_is_propagated() -> None:
    connection = MemoryConnection(receive_error=TimeoutError())
    with pytest.raises(TimeoutError):
        receive_frame(connection, expected_version=1)


def test_disconnect_mid_frame_is_rejected() -> None:
    with pytest.raises(PeerDisconnected, match="outstanding"):
        receive_frame(
            _connection_for(struct.pack("!I", 10) + b"short"), expected_version=1
        )


def test_version_mismatch_is_rejected() -> None:
    with pytest.raises(VersionMismatch):
        receive_frame(
            _connection_for(encode_frame(_inference_frame())), expected_version=2
        )


def test_tensor_payload_round_trips_bit_exactly() -> None:
    payload = bytes(range(256)) * 4
    frame = Frame(
        version=1,
        message_type=MessageType.TENSOR_REQUEST,
        request_id="tensor-1",
        metadata={
            "prompt": "Describe.",
            "model_fingerprint": FINGERPRINT,
            "shape": [512],
            "dtype": "bfloat16",
            "byte_order": "little",
            "layout": "contiguous",
            "nbytes": len(payload),
        },
        payload=payload,
    )
    received = receive_frame(_connection_for(encode_frame(frame)), expected_version=1)

    assert received.frame.payload == payload


def test_message_schema_rejects_unexpected_metadata() -> None:
    with pytest.raises(ProtocolError, match="metadata"):
        Frame(
            version=1,
            message_type=MessageType.HEALTH_REQUEST,
            request_id="health-1",
            metadata={"unexpected": True},
        )


def test_image_suffix_cannot_escape_temporary_directory() -> None:
    with pytest.raises(ProtocolError, match="image_suffix"):
        Frame(
            version=1,
            message_type=MessageType.INFERENCE_REQUEST,
            request_id="request-1",
            metadata={
                "prompt": "Describe.",
                "model_fingerprint": FINGERPRINT,
                "image_suffix": "../image.png",
            },
            payload=b"image",
        )


def test_non_finite_metric_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="non-finite"):
        Frame(
            version=1,
            message_type=MessageType.INFERENCE_RESPONSE,
            request_id="response-1",
            metadata={
                "model_fingerprint": FINGERPRINT,
                "output_text": "text",
                "output_token_ids": [1],
                "metrics": {"latency_ms": float("nan")},
            },
        )
