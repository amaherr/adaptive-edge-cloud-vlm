"""All-cloud execution policy using the bounded TCP protocol."""

from __future__ import annotations

from time import perf_counter_ns

from adaptive_vlm.config import ExecutionMode, TransportConfig
from adaptive_vlm.contracts import InferenceRequest, InferenceResult, MetricValue
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.remote import HealthStatus as HealthStatus
from adaptive_vlm.execution.remote import RemoteCapabilities as RemoteCapabilities
from adaptive_vlm.execution.remote import RemoteInferenceError as RemoteInferenceError
from adaptive_vlm.execution.remote import (
    check_remote_capabilities,
    check_remote_health,
    parse_inference_response,
)
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.transport.client import exchange_frame
from adaptive_vlm.transport.protocol import Frame, MessageType


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
        return check_remote_health(self._transport, self._model)

    def capabilities(self) -> RemoteCapabilities:
        return check_remote_capabilities(self._transport, self._model)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        image = request.image_path.read_bytes()
        started_ns = perf_counter_ns()
        exchange = exchange_frame(
            self._transport,
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
        remote = parse_inference_response(
            exchange,
            request_id=request.request_id,
            model_fingerprint=self._model.fingerprint,
        )
        metrics: dict[str, MetricValue] = dict(remote.metrics)
        metrics.update(
            {
                "client_end_to_end_ms": client_end_to_end_ms,
                "request_wire_bytes": exchange.request_wire_bytes,
                "response_wire_bytes": exchange.response_wire_bytes,
                "image_bytes": len(image),
            }
        )
        return InferenceResult(
            request_id=remote.request_id,
            output_text=remote.output_text,
            output_token_ids=remote.output_token_ids,
            metrics=metrics,
        )
