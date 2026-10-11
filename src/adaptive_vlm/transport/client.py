"""Reusable request/response exchange for framed TCP clients."""

from __future__ import annotations

import socket
from dataclasses import dataclass

from adaptive_vlm.config import TransportConfig
from adaptive_vlm.transport.protocol import Frame, receive_frame, send_frame


@dataclass(frozen=True)
class ExchangeResult:
    response: Frame
    request_wire_bytes: int
    response_wire_bytes: int


def exchange_frame(transport: TransportConfig, frame: Frame) -> ExchangeResult:
    connection = socket.create_connection(
        (transport.host, transport.port),
        timeout=transport.timeout_seconds,
    )
    try:
        connection.settimeout(transport.timeout_seconds)
        sent = send_frame(connection, frame)
        received = receive_frame(
            connection,
            expected_version=transport.protocol_version,
        )
        return ExchangeResult(
            response=received.frame,
            request_wire_bytes=sent,
            response_wire_bytes=received.wire_bytes,
        )
    finally:
        connection.close()
