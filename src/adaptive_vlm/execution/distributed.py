"""Synchronous edge-vision/cloud-language execution policy."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter_ns
from typing import Protocol

from adaptive_vlm.config import ExecutionMode, TransportConfig
from adaptive_vlm.contracts import InferenceRequest, InferenceResult, MetricValue
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.remote import (
    HealthStatus,
    check_remote_health,
    parse_inference_response,
)
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.features import VisionEncoding
from adaptive_vlm.transport.client import exchange_frame
from adaptive_vlm.transport.protocol import Frame, MessageType
from adaptive_vlm.transport.tensors import TensorPayload

TensorSerializer = Callable[[object], TensorPayload]


class EdgeVisionAdapter(Protocol):
    @property
    def component_role(self) -> str: ...

    def encode_projected(self, request: InferenceRequest) -> VisionEncoding: ...


class DistributedExecutionPolicy(ExecutionPolicy):
    def __init__(
        self,
        *,
        encoder: EdgeVisionAdapter,
        transport: TransportConfig,
        model: ModelMetadata,
        tensor_serializer: TensorSerializer = TensorPayload.from_torch,
    ) -> None:
        if encoder.component_role != "vision":
            raise ValueError("distributed edge adapter must retain only vision")
        self._encoder = encoder
        self._transport = transport
        self._model = model
        self._tensor_serializer = tensor_serializer

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.DISTRIBUTED

    def health(self) -> HealthStatus:
        return check_remote_health(self._transport, self._model)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        started_ns = perf_counter_ns()
        encoding = self._encoder.encode_projected(request)

        serialization_started_ns = perf_counter_ns()
        tensor = self._tensor_serializer(encoding.projected.tensor)
        serialization_ms = (
            perf_counter_ns() - serialization_started_ns
        ) / 1_000_000
        if tensor.shape != encoding.projected.shape:
            raise ValueError("serialized tensor shape differs from projected features")
        if tensor.dtype != encoding.projected.dtype:
            raise ValueError("serialized tensor dtype differs from projected features")

        metadata = {
            "prompt": request.prompt,
            "model_fingerprint": self._model.fingerprint,
            **tensor.metadata(),
        }
        network_started_ns = perf_counter_ns()
        exchange = exchange_frame(
            self._transport,
            Frame(
                version=self._transport.protocol_version,
                message_type=MessageType.TENSOR_REQUEST,
                request_id=request.request_id,
                metadata=metadata,
                payload=tensor.data,
            ),
        )
        network_round_trip_ms = (
            perf_counter_ns() - network_started_ns
        ) / 1_000_000
        remote = parse_inference_response(
            exchange,
            request_id=request.request_id,
            model_fingerprint=self._model.fingerprint,
        )

        metrics: dict[str, MetricValue] = dict(encoding.metrics)
        metrics.update(remote.metrics)
        metrics.update(
            {
                "tensor_serialization_ms": serialization_ms,
                "network_round_trip_ms": network_round_trip_ms,
                "client_end_to_end_ms": (
                    perf_counter_ns() - started_ns
                )
                / 1_000_000,
                "request_wire_bytes": exchange.request_wire_bytes,
                "response_wire_bytes": exchange.response_wire_bytes,
                "projected_payload_bytes": len(tensor.data),
                "request_metadata_and_framing_bytes": (
                    exchange.request_wire_bytes - len(tensor.data)
                ),
            }
        )
        return InferenceResult(
            request_id=remote.request_id,
            output_text=remote.output_text,
            output_token_ids=remote.output_token_ids,
            metrics=metrics,
        )
