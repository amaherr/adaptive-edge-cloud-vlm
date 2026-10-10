"""All-cloud execution policy using the bounded TCP protocol."""

from __future__ import annotations

import socket
import uuid
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, cast

from adaptive_vlm.config import ExecutionMode, TransportConfig
from adaptive_vlm.contracts import InferenceRequest, InferenceResult, MetricValue
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.transport.protocol import (
    Frame,
    MessageType,
    ProtocolError,
    receive_frame,
    send_frame,
)


class RemoteInferenceError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class HealthStatus:
    ready: bool
    model_fingerprint: str


class CloudExecutionPolicy(ExecutionPolicy):
    def __init__(
        self,
        *,
        transport: TransportConfig,
        model: ModelMetadata,
    ) -> None:
        self._transport = transport
        self._model = model

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.CLOUD

    def health(self) -> HealthStatus:
        request_id = f"health-{uuid.uuid4().hex}"
        response, _, _ = self._exchange(
            Frame(
                version=self._transport.protocol_version,
                message_type=MessageType.HEALTH_REQUEST,
                request_id=request_id,
                metadata={},
            )
        )
        self._validate_response(response, request_id)
        if response.message_type is not MessageType.HEALTH_RESPONSE:
            raise ProtocolError("server returned a non-health response")
        fingerprint = cast(str, response.metadata["model_fingerprint"])
        if fingerprint != self._model.fingerprint:
            raise ProtocolError("server health reported a different model fingerprint")
        return HealthStatus(
            ready=cast(bool, response.metadata["ready"]),
            model_fingerprint=fingerprint,
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        image = request.image_path.read_bytes()
        started_ns = perf_counter_ns()
        response, request_wire_bytes, response_wire_bytes = self._exchange(
            Frame(
                version=self._transport.protocol_version,
                message_type=MessageType.INFERENCE_REQUEST,
                request_id=request.request_id,
                metadata={
                    "prompt": request.prompt,
                    "model_fingerprint": self._model.fingerprint,
                    "image_suffix": request.image_path.suffix.lower(),
                },
                payload=image,
            )
        )
        client_end_to_end_ms = (perf_counter_ns() - started_ns) / 1_000_000
        self._validate_response(response, request.request_id)
        if response.message_type is not MessageType.INFERENCE_RESPONSE:
            raise ProtocolError("server returned a non-inference response")
        if response.metadata["model_fingerprint"] != self._model.fingerprint:
            raise ProtocolError("inference response model fingerprint differs")

        remote_metrics = cast(dict[str, MetricValue], response.metadata["metrics"])
        metrics: dict[str, MetricValue] = dict(remote_metrics)
        metrics.update(
            {
                "client_end_to_end_ms": client_end_to_end_ms,
                "request_wire_bytes": request_wire_bytes,
                "response_wire_bytes": response_wire_bytes,
                "image_bytes": len(image),
            }
        )
        return InferenceResult(
            request_id=response.request_id,
            output_text=cast(str, response.metadata["output_text"]),
            output_token_ids=tuple(
                cast(list[int], response.metadata["output_token_ids"])
            ),
            metrics=metrics,
        )

    def _exchange(self, frame: Frame) -> tuple[Frame, int, int]:
        connection = socket.create_connection(
            (self._transport.host, self._transport.port),
            timeout=self._transport.timeout_seconds,
        )
        try:
            connection.settimeout(self._transport.timeout_seconds)
            sent = send_frame(connection, frame)
            received = receive_frame(
                connection,
                expected_version=self._transport.protocol_version,
            )
            return received.frame, sent, received.wire_bytes
        finally:
            connection.close()

    @staticmethod
    def _validate_response(response: Frame, request_id: str) -> None:
        if response.request_id != request_id:
            raise ProtocolError("response request_id does not match request")
        if response.message_type is MessageType.ERROR:
            metadata: dict[str, Any] = dict(response.metadata)
            raise RemoteInferenceError(
                cast(str, metadata["code"]),
                cast(str, metadata["message"]),
                retryable=cast(bool, metadata["retryable"]),
            )
