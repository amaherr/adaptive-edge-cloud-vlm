from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.config import TransportConfig
from adaptive_vlm.contracts import InferenceRequest
from adaptive_vlm.execution.cloud import CloudExecutionPolicy, RemoteInferenceError
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.transport.protocol import (
    Frame,
    MessageType,
    ReceivedFrame,
    encode_frame,
    receive_frame,
)
from adaptive_vlm.transport.server import CloudRequestProcessor


class _Reader:
    def __init__(self, data: bytes, fragment_size: int = 7) -> None:
        self.data = bytearray(data)
        self.fragment_size = fragment_size

    def recv(self, size: int) -> bytes:
        count = min(size, self.fragment_size, len(self.data))
        result = bytes(self.data[:count])
        del self.data[:count]
        return result


class LoopbackConnection:
    def __init__(self, responder: Callable[[ReceivedFrame], Frame]) -> None:
        self._responder = responder
        self._request = bytearray()
        self._response: _Reader | None = None
        self.closed = False
        self.timeout: float | None = None

    def sendall(self, data: bytes) -> None:
        self._request.extend(data)

    def recv(self, size: int) -> bytes:
        if self._response is None:
            request = receive_frame(_Reader(self._request), expected_version=1)
            self._response = _Reader(encode_frame(self._responder(request)))
        return self._response.recv(size)

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def close(self) -> None:
        self.closed = True


def _transport() -> TransportConfig:
    return TransportConfig(
        host="127.0.0.1",
        port=5050,
        timeout_seconds=2.0,
        protocol_version=1,
    )


def _install_loopback(
    monkeypatch: pytest.MonkeyPatch,
    responder: Callable[[ReceivedFrame], Frame],
) -> list[LoopbackConnection]:
    connections: list[LoopbackConnection] = []

    def connect(*_: Any, **__: Any) -> LoopbackConnection:
        connection = LoopbackConnection(responder)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "adaptive_vlm.transport.client.socket.create_connection", connect
    )
    return connections


def test_health_validates_server_model(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeModelAdapter()
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=adapter.metadata.fingerprint,
        inference_function=adapter.generate,
    )
    connections = _install_loopback(monkeypatch, processor.handle)
    policy = CloudExecutionPolicy(transport=_transport(), model=adapter.metadata)

    health = policy.health()

    assert health.ready is True
    assert health.model_fingerprint == adapter.metadata.fingerprint
    assert connections[0].timeout == 2.0
    assert connections[0].closed is True


def test_cloud_policy_returns_output_and_communication_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = FakeModelAdapter()
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=adapter.metadata.fingerprint,
        inference_function=adapter.generate,
    )
    _install_loopback(monkeypatch, processor.handle)
    image = tmp_path / "image.png"
    image.write_bytes(b"image bytes")
    request = InferenceRequest("request-1", image, "Describe the image.")
    expected = adapter.generate(request)

    result = CloudExecutionPolicy(transport=_transport(), model=adapter.metadata).infer(
        request
    )

    assert result.output_text == expected.output_text
    assert result.output_token_ids == expected.output_token_ids
    assert result.metrics["request_wire_bytes"] > len(b"image bytes")
    assert result.metrics["response_wire_bytes"] > 0
    assert result.metrics["image_bytes"] == len(b"image bytes")
    assert result.metrics["server_model_output_tokens"] == 8


def test_cloud_policy_surfaces_structured_remote_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = FakeModelAdapter()

    def reject(request: ReceivedFrame) -> Frame:
        return Frame(
            version=1,
            message_type=MessageType.ERROR,
            request_id=request.frame.request_id,
            metadata={
                "code": "model_error",
                "message": "deliberate",
                "retryable": False,
            },
        )

    _install_loopback(monkeypatch, reject)
    image = tmp_path / "image.png"
    image.write_bytes(b"image bytes")
    policy = CloudExecutionPolicy(transport=_transport(), model=adapter.metadata)

    with pytest.raises(RemoteInferenceError, match="deliberate") as error:
        policy.infer(InferenceRequest("request-1", image, "Prompt"))

    assert error.value.code == "model_error"
    assert error.value.retryable is False
