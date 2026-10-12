from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.transport.protocol import (
    Frame,
    MessageType,
    ReceivedFrame,
    encode_frame,
    receive_frame,
)
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer
from adaptive_vlm.transport.tensors import TensorInferenceRequest

FINGERPRINT = FakeModelAdapter().metadata.fingerprint


class _Connection:
    def __init__(self, incoming: bytes) -> None:
        self.incoming = bytearray(incoming)
        self.sent = bytearray()
        self.timeout: float | None = None

    def recv(self, size: int) -> bytes:
        count = min(size, 11, len(self.incoming))
        result = bytes(self.incoming[:count])
        del self.incoming[:count]
        return result

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *_: Any) -> None:
        return None


class _Listener:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection
        self.server: CloudServer | None = None
        self.accept_count = 0
        self.closed = False
        self.bound_address: tuple[str, int] | None = None

    def setsockopt(self, *_: Any) -> None:
        return None

    def bind(self, address: tuple[str, int]) -> None:
        self.bound_address = address

    def listen(self) -> None:
        return None

    def settimeout(self, _: float) -> None:
        return None

    def getsockname(self) -> tuple[str, int]:
        return ("127.0.0.1", 45123)

    def accept(self) -> tuple[_Connection, tuple[str, int]]:
        self.accept_count += 1
        if self.accept_count == 1:
            return self.connection, ("127.0.0.1", 50000)
        assert self.server is not None
        self.server.shutdown()
        raise OSError("listener closed")

    def close(self) -> None:
        self.closed = True


def _received(frame: Frame) -> ReceivedFrame:
    return ReceivedFrame(frame=frame, wire_bytes=len(encode_frame(frame)))


def _request(
    *, fingerprint: str = FINGERPRINT, request_id: str = "request-1"
) -> ReceivedFrame:
    return _received(
        Frame(
            version=1,
            message_type=MessageType.INFERENCE_REQUEST,
            request_id=request_id,
            metadata={
                "prompt": "Describe the image.",
                "model_fingerprint": fingerprint,
                "image_suffix": ".png",
            },
            payload=b"image bytes",
        )
    )


def _processor() -> CloudRequestProcessor:
    adapter = FakeModelAdapter()
    return CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=adapter.metadata.fingerprint,
        inference_function=adapter.generate,
    )


def test_health_reports_readiness_and_loaded_fingerprint() -> None:
    processor = _processor()
    response = processor.handle(
        _received(
            Frame(
                version=1,
                message_type=MessageType.HEALTH_REQUEST,
                request_id="health-1",
                metadata={},
            )
        )
    )

    assert response.message_type is MessageType.HEALTH_RESPONSE
    assert response.metadata == {
        "ready": True,
        "model_fingerprint": FINGERPRINT,
    }


def test_inference_returns_model_output_and_server_metrics(tmp_path: Path) -> None:
    adapter = FakeModelAdapter()
    local_image = tmp_path / "input.png"
    local_image.write_bytes(b"image bytes")
    expected = adapter.generate(
        InferenceRequest("request-1", local_image, "Describe the image.")
    )
    processor = _processor()

    response = processor.handle(_request())

    assert response.message_type is MessageType.INFERENCE_RESPONSE
    assert response.metadata["output_text"] == expected.output_text
    assert response.metadata["output_token_ids"] == list(expected.output_token_ids)
    metrics = response.metadata["metrics"]
    assert metrics["server_model_output_tokens"] == 8
    assert metrics["server_request_wire_bytes"] > len(b"image bytes")
    assert metrics["server_total_ms"] >= 0
    assert processor.stats.completed_requests == 1


def test_wrong_model_fingerprint_is_rejected_without_model_call() -> None:
    calls = 0

    def infer(_: InferenceRequest) -> InferenceResult:
        nonlocal calls
        calls += 1
        raise AssertionError("model should not run")

    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        inference_function=infer,
    )

    response = processor.handle(_request(fingerprint="b" * 64))

    assert response.message_type is MessageType.ERROR
    assert response.metadata["code"] == "model_fingerprint_mismatch"
    assert calls == 0
    assert processor.stats.rejected_requests == 1


def test_tensor_request_uses_only_tensor_inference_function() -> None:
    received_data: list[bytes] = []

    def tensor_infer(request: TensorInferenceRequest) -> InferenceResult:
        received_data.append(request.tensor.data)
        return InferenceResult(
            request_id=request.request_id,
            output_text="tensor-result",
            output_token_ids=(1, 2),
            metrics={"tensor_bytes": len(request.tensor.data)},
        )

    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        tensor_inference_function=tensor_infer,
    )
    data = bytes(range(16))
    request = Frame(
        version=1,
        message_type=MessageType.TENSOR_REQUEST,
        request_id="tensor-1",
        metadata={
            "prompt": "Describe.",
            "model_fingerprint": FINGERPRINT,
            "shape": [2, 4],
            "dtype": "bfloat16",
            "byte_order": sys.byteorder,
            "layout": "contiguous",
            "nbytes": len(data),
            "pipeline": "synchronous",
        },
        payload=data,
    )

    response = processor.handle(_received(request))

    assert response.message_type is MessageType.INFERENCE_RESPONSE
    assert response.metadata["output_text"] == "tensor-result"
    assert received_data == [data]
    assert processor.stats.completed_requests == 1


def test_tensor_request_is_rejected_by_image_only_server() -> None:
    processor = _processor()
    data = bytes(16)
    request = Frame(
        version=1,
        message_type=MessageType.TENSOR_REQUEST,
        request_id="tensor-1",
        metadata={
            "prompt": "Describe.",
            "model_fingerprint": FINGERPRINT,
            "shape": [2, 4],
            "dtype": "bfloat16",
            "byte_order": sys.byteorder,
            "layout": "contiguous",
            "nbytes": len(data),
            "pipeline": "synchronous",
        },
        payload=data,
    )

    response = processor.handle(_received(request))

    assert response.message_type is MessageType.ERROR
    assert response.metadata["code"] == "unsupported_execution_path"


def _prompt_request(
    *, request_id: str = "prompt-first-1", prompt: str = "Describe."
) -> ReceivedFrame:
    return _received(
        Frame(
            version=1,
            message_type=MessageType.PROMPT_REQUEST,
            request_id=request_id,
            metadata={"prompt": prompt, "model_fingerprint": FINGERPRINT},
        )
    )


def _tensor_request(
    *, request_id: str = "prompt-first-1", prompt: str = "Describe."
) -> ReceivedFrame:
    data = bytes(16)
    return _received(
        Frame(
            version=1,
            message_type=MessageType.TENSOR_REQUEST,
            request_id=request_id,
            metadata={
                "prompt": prompt,
                "model_fingerprint": FINGERPRINT,
                "shape": [2, 4],
                "dtype": "bfloat16",
                "byte_order": sys.byteorder,
                "layout": "contiguous",
                "nbytes": len(data),
                "pipeline": "prompt_first",
            },
            payload=data,
        )
    )


def test_prompt_first_state_is_consumed_by_matching_tensor() -> None:
    prepared_calls: list[tuple[str, object]] = []

    def prepared_infer(
        request: TensorInferenceRequest, prepared: object
    ) -> InferenceResult:
        prepared_calls.append((request.prompt, prepared))
        return InferenceResult(request.request_id, "prepared", (7,), {})

    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        tensor_inference_function=lambda request: InferenceResult(
            request.request_id, "sync", (), {}
        ),
        prompt_prepare_function=lambda prompt: tuple(prompt.upper()),
        prepared_tensor_inference_function=prepared_infer,
    )

    prompt_response = processor.handle(_prompt_request())
    tensor_response = processor.handle(_tensor_request())

    assert prompt_response.message_type is MessageType.PROMPT_RESPONSE
    assert prompt_response.metadata["accepted"] is True
    assert tensor_response.message_type is MessageType.INFERENCE_RESPONSE
    assert prepared_calls == [("Describe.", tuple("DESCRIBE."))]
    assert processor.pending_requests == 0


def test_prompt_first_state_expires_and_is_cleaned_up() -> None:
    now = [10.0]
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        tensor_inference_function=lambda request: InferenceResult(
            request.request_id, "sync", (), {}
        ),
        prompt_prepare_function=lambda prompt: prompt,
        prepared_tensor_inference_function=lambda request, _: InferenceResult(
            request.request_id, "prepared", (), {}
        ),
        pending_timeout_seconds=5.0,
        clock=lambda: now[0],
    )
    processor.handle(_prompt_request())
    now[0] = 16.0

    removed = processor.cleanup_pending()
    response = processor.handle(_tensor_request())

    assert removed == 1
    assert processor.pending_requests == 0
    assert response.message_type is MessageType.ERROR
    assert response.metadata["code"] == "pending_prompt_not_found"


def test_prompt_first_rejects_mismatched_prompt_and_discards_state() -> None:
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        tensor_inference_function=lambda request: InferenceResult(
            request.request_id, "sync", (), {}
        ),
        prompt_prepare_function=lambda prompt: prompt,
        prepared_tensor_inference_function=lambda request, _: InferenceResult(
            request.request_id, "prepared", (), {}
        ),
    )
    processor.handle(_prompt_request(prompt="Original"))

    response = processor.handle(_tensor_request(prompt="Changed"))

    assert response.message_type is MessageType.ERROR
    assert response.metadata["code"] == "prompt_mismatch"
    assert processor.pending_requests == 0


def test_prompt_first_rejects_duplicate_and_full_pending_state() -> None:
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        tensor_inference_function=lambda request: InferenceResult(
            request.request_id, "sync", (), {}
        ),
        prompt_prepare_function=lambda prompt: prompt,
        prepared_tensor_inference_function=lambda request, _: InferenceResult(
            request.request_id, "prepared", (), {}
        ),
        max_pending_requests=1,
    )
    processor.handle(_prompt_request(request_id="first"))

    duplicate = processor.handle(_prompt_request(request_id="first"))
    full = processor.handle(_prompt_request(request_id="second"))

    assert duplicate.metadata["code"] == "duplicate_request_id"
    assert full.metadata["code"] == "pending_capacity_exceeded"
    assert processor.pending_requests == 1


def test_model_failure_becomes_safe_error_and_server_remains_usable() -> None:
    calls = 0

    def infer(request: InferenceRequest) -> InferenceResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("deliberate failure")
        return FakeModelAdapter().generate(request)

    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        inference_function=infer,
    )

    failed = processor.handle(_request(request_id="failure"))
    recovered = processor.handle(_request(request_id="recovered"))

    assert failed.message_type is MessageType.ERROR
    assert failed.metadata["code"] == "model_error"
    assert recovered.message_type is MessageType.INFERENCE_RESPONSE
    assert processor.stats.failed_requests == 1
    assert processor.stats.completed_requests == 1


def test_one_hundred_sequential_requests_complete() -> None:
    processor = _processor()

    outputs = [
        processor.handle(_request(request_id=f"request-{index}"))
        for index in range(100)
    ]

    assert all(item.message_type is MessageType.INFERENCE_RESPONSE for item in outputs)
    assert len({item.metadata["output_text"] for item in outputs}) == 1
    assert processor.stats.completed_requests == 100


def test_server_lifecycle_serves_request_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_request = encode_frame(_request().frame)
    connection = _Connection(encoded_request)
    listener = _Listener(connection)
    monkeypatch.setattr(
        "adaptive_vlm.transport.server.socket.socket", lambda *_: listener
    )
    server = CloudServer(
        host="127.0.0.1",
        port=0,
        timeout_seconds=2,
        processor=_processor(),
    )
    listener.server = server

    server.start()
    assert server.is_ready
    assert server.port == 45123
    with pytest.raises(RuntimeError, match="already started"):
        server.start()
    server.serve_forever()

    response = receive_frame(_Connection(connection.sent), expected_version=1).frame
    assert response.message_type is MessageType.INFERENCE_RESPONSE
    assert connection.timeout == 2
    assert listener.closed
    assert not server.is_ready


@pytest.mark.parametrize(
    ("port", "timeout", "message"),
    [(-1, 1.0, "port"), (65536, 1.0, "port"), (5050, 0.0, "timeout")],
)
def test_server_rejects_invalid_network_settings(
    port: int, timeout: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CloudServer(
            host="127.0.0.1",
            port=port,
            timeout_seconds=timeout,
            processor=_processor(),
        )
