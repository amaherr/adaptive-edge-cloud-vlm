"""Shared health and response handling for remote execution policies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, cast

from adaptive_vlm.config import TransportConfig
from adaptive_vlm.contracts import InferenceResult, MetricValue
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.transport.client import ExchangeResult, exchange_frame
from adaptive_vlm.transport.protocol import Frame, MessageType, ProtocolError


class RemoteInferenceError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class HealthStatus:
    ready: bool
    model_fingerprint: str


@dataclass(frozen=True)
class RemoteCapabilities:
    ready: bool
    model_fingerprint: str
    paths: frozenset[str]


def check_remote_capabilities(
    transport: TransportConfig, model: ModelMetadata
) -> RemoteCapabilities:
    request_id = f"capabilities-{uuid.uuid4().hex}"
    exchange = exchange_frame(
        transport,
        Frame(
            version=transport.protocol_version,
            message_type=MessageType.CAPABILITY_REQUEST,
            request_id=request_id,
            metadata={},
        ),
    )
    response = exchange.response
    validate_response(response, request_id)
    if response.message_type is not MessageType.CAPABILITY_RESPONSE:
        raise ProtocolError("server returned a non-capability response")
    fingerprint = cast(str, response.metadata["model_fingerprint"])
    if fingerprint != model.fingerprint:
        raise ProtocolError(
            "server capabilities reported a different model fingerprint"
        )
    return RemoteCapabilities(
        ready=cast(bool, response.metadata["ready"]),
        model_fingerprint=fingerprint,
        paths=frozenset(cast(list[str], response.metadata["paths"])),
    )


def check_remote_health(
    transport: TransportConfig, model: ModelMetadata
) -> HealthStatus:
    request_id = f"health-{uuid.uuid4().hex}"
    exchange = exchange_frame(
        transport,
        Frame(
            version=transport.protocol_version,
            message_type=MessageType.HEALTH_REQUEST,
            request_id=request_id,
            metadata={},
        ),
    )
    response = exchange.response
    validate_response(response, request_id)
    if response.message_type is not MessageType.HEALTH_RESPONSE:
        raise ProtocolError("server returned a non-health response")
    fingerprint = cast(str, response.metadata["model_fingerprint"])
    if fingerprint != model.fingerprint:
        raise ProtocolError("server health reported a different model fingerprint")
    return HealthStatus(
        ready=cast(bool, response.metadata["ready"]),
        model_fingerprint=fingerprint,
    )


def parse_inference_response(
    exchange: ExchangeResult,
    *,
    request_id: str,
    model_fingerprint: str,
) -> InferenceResult:
    response = exchange.response
    validate_response(response, request_id)
    if response.message_type is not MessageType.INFERENCE_RESPONSE:
        raise ProtocolError("server returned a non-inference response")
    if response.metadata["model_fingerprint"] != model_fingerprint:
        raise ProtocolError("inference response model fingerprint differs")
    return InferenceResult(
        request_id=response.request_id,
        output_text=cast(str, response.metadata["output_text"]),
        output_token_ids=tuple(cast(list[int], response.metadata["output_token_ids"])),
        metrics=cast(dict[str, MetricValue], response.metadata["metrics"]),
    )


def validate_response(response: Frame, request_id: str) -> None:
    if response.request_id != request_id:
        raise ProtocolError("response request_id does not match request")
    if response.message_type is MessageType.ERROR:
        metadata: dict[str, Any] = dict(response.metadata)
        raise RemoteInferenceError(
            cast(str, metadata["code"]),
            cast(str, metadata["message"]),
            retryable=cast(bool, metadata["retryable"]),
        )
