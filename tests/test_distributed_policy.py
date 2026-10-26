from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Any

import pytest

from adaptive_vlm.config import ExecutionMode, TransportConfig
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.execution.distributed import DistributedExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.features import ProjectedVisionFeatures, VisionEncoding
from adaptive_vlm.transport.protocol import (
    Frame,
    ReceivedFrame,
    encode_frame,
    receive_frame,
)
from adaptive_vlm.transport.server import CloudRequestProcessor
from adaptive_vlm.transport.tensors import TensorInferenceRequest, TensorPayload

FINGERPRINT = "a" * 64
MODEL = ModelMetadata("test/model", "revision", FINGERPRINT)


@dataclass
class _FakeTensor:
    shape: tuple[int, int] = (729, 2048)
    dtype: str = "bfloat16"
    device: str = "cpu"

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def element_size(self) -> int:
        return 2


class _FakeEncoder:
    component_role = "vision"

    def __init__(self) -> None:
        self.calls = 0

    def encode_projected(self, _: InferenceRequest) -> VisionEncoding:
        self.calls += 1
        return VisionEncoding(
            projected=ProjectedVisionFeatures.from_tensor(_FakeTensor()),
            metrics={"edge_vision_ms": 12.5},
        )


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = bytearray(data)

    def recv(self, size: int) -> bytes:
        count = min(size, 13, len(self.data))
        result = bytes(self.data[:count])
        del self.data[:count]
        return result


class _LoopbackConnection:
    def __init__(self, responder: Callable[[ReceivedFrame], Frame]) -> None:
        self.responder = responder
        self.request = bytearray()
        self.response: _Reader | None = None

    def sendall(self, data: bytes) -> None:
        self.request.extend(data)

    def recv(self, size: int) -> bytes:
        if self.response is None:
            received = receive_frame(_Reader(self.request), expected_version=1)
            self.response = _Reader(encode_frame(self.responder(received)))
        return self.response.recv(size)

    def settimeout(self, _: float) -> None:
        pass

    def close(self) -> None:
        pass


def _transport() -> TransportConfig:
    return TransportConfig("127.0.0.1", 5050, 2.0, 1)


def _serializer(_: object) -> TensorPayload:
    return TensorPayload(
        shape=(729, 2048),
        dtype="bfloat16",
        byte_order=sys.byteorder,
        layout="contiguous",
        data=bytes(2_985_984),
    )


def _install_loopback(
    monkeypatch: pytest.MonkeyPatch,
    responder: Callable[[ReceivedFrame], Frame],
) -> None:
    monkeypatch.setattr(
        "adaptive_vlm.transport.client.socket.create_connection",
        lambda *_args, **_kwargs: _LoopbackConnection(responder),
    )


def test_synchronous_distributed_policy_sends_projected_tensor_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    received_tensors: list[TensorInferenceRequest] = []

    def tensor_infer(request: TensorInferenceRequest) -> InferenceResult:
        received_tensors.append(request)
        return InferenceResult(
            request.request_id,
            "remote text",
            (10, 11),
            {"language_generation_ms": 4.0},
        )

    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        tensor_inference_function=tensor_infer,
    )
    _install_loopback(monkeypatch, processor.handle)
    encoder = _FakeEncoder()
    image = tmp_path / "image.png"
    image.write_bytes(b"image is consumed only by edge encoder")
    policy = DistributedExecutionPolicy(
        encoder=encoder,
        transport=_transport(),
        model=MODEL,
        tensor_serializer=_serializer,
    )

    result = policy.infer(InferenceRequest("split-1", image, "Describe."))

    assert policy.mode is ExecutionMode.DISTRIBUTED
    assert encoder.calls == 1
    assert len(received_tensors) == 1
    assert received_tensors[0].tensor.data == bytes(2_985_984)
    assert result.output_token_ids == (10, 11)
    assert result.metrics["edge_vision_ms"] == 12.5
    assert result.metrics["server_model_language_generation_ms"] == 4.0
    assert result.metrics["projected_payload_bytes"] == 2_985_984
    assert result.metrics["request_metadata_and_framing_bytes"] > 0
    assert result.metrics["total_request_wire_bytes"] == result.metrics[
        "request_wire_bytes"
    ]


def test_distributed_policy_accepts_resident_full_or_vision_adapter() -> None:
    encoder = _FakeEncoder()
    encoder.component_role = "full"
    policy = DistributedExecutionPolicy(
        encoder=encoder,
        transport=_transport(),
        model=MODEL,
        tensor_serializer=_serializer,
    )
    assert policy.mode is ExecutionMode.DISTRIBUTED

    encoder.component_role = "text"
    with pytest.raises(ValueError, match="retain vision"):
        DistributedExecutionPolicy(
            encoder=encoder,
            transport=_transport(),
            model=MODEL,
            tensor_serializer=_serializer,
        )


def test_distributed_health_uses_shared_contract_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        tensor_inference_function=lambda request: InferenceResult(
            request.request_id, "", (), {}
        ),
    )
    _install_loopback(monkeypatch, processor.handle)
    policy = DistributedExecutionPolicy(
        encoder=_FakeEncoder(),
        transport=_transport(),
        model=MODEL,
        tensor_serializer=_serializer,
    )

    health = policy.health()

    assert health.ready
    assert health.model_fingerprint == FINGERPRINT


def test_prompt_first_overlaps_prepare_and_vision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    prompt_received = Event()
    prepared_calls: list[tuple[str, object]] = []

    def prepare(prompt: str) -> object:
        prompt_received.set()
        return tuple(prompt)

    def prepared_infer(
        request: TensorInferenceRequest, prepared: object
    ) -> InferenceResult:
        prepared_calls.append((request.prompt, prepared))
        return InferenceResult(request.request_id, "remote text", (10, 11), {})

    class WaitingEncoder(_FakeEncoder):
        def encode_projected(self, request: InferenceRequest) -> VisionEncoding:
            assert prompt_received.wait(2), "prompt was not sent during vision encoding"
            return super().encode_projected(request)

    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        tensor_inference_function=lambda request: InferenceResult(
            request.request_id, "sync", (), {}
        ),
        prompt_prepare_function=prepare,
        prepared_tensor_inference_function=prepared_infer,
    )
    _install_loopback(monkeypatch, processor.handle)
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    policy = DistributedExecutionPolicy(
        encoder=WaitingEncoder(),
        transport=_transport(),
        model=MODEL,
        tensor_serializer=_serializer,
        pipeline="prompt_first",
    )

    result = policy.infer(InferenceRequest("overlap-1", image, "Describe."))

    assert result.output_token_ids == (10, 11)
    assert result.metrics["distributed_pipeline"] == "prompt_first"
    assert result.metrics["prompt_request_wire_bytes"] > 0
    assert result.metrics["server_prompt_prepare_ms"] >= 0
    assert result.metrics["total_request_wire_bytes"] == (
        result.metrics["request_wire_bytes"]
        + result.metrics["prompt_request_wire_bytes"]
    )
    assert prepared_calls == [("Describe.", tuple("Describe."))]
    assert processor.pending_requests == 0
