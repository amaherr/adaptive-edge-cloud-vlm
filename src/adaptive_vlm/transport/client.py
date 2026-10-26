"""Reusable request/response exchange for framed TCP clients."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from time import perf_counter_ns

from adaptive_vlm.config import TransportConfig
from adaptive_vlm.transport.protocol import Frame, receive_frame, send_frame


@dataclass(frozen=True)
class ExchangeResult:
    response: Frame
    request_wire_bytes: int
    response_wire_bytes: int
    connect_ms: float = 0.0
    send_ms: float = 0.0
    receive_ms: float = 0.0


def exchange_frame(transport: TransportConfig, frame: Frame) -> ExchangeResult:
    connect_started_ns = perf_counter_ns()
    connection = socket.create_connection(
        (transport.host, transport.port),
        timeout=transport.timeout_seconds,
    )
    connect_ms = (perf_counter_ns() - connect_started_ns) / 1_000_000
    try:
        connection.settimeout(transport.timeout_seconds)
        send_started_ns = perf_counter_ns()
        sent = send_frame(connection, frame)
        send_ms = (perf_counter_ns() - send_started_ns) / 1_000_000
        receive_started_ns = perf_counter_ns()
        received = receive_frame(
            connection,
            expected_version=transport.protocol_version,
        )
        receive_ms = (perf_counter_ns() - receive_started_ns) / 1_000_000
        return ExchangeResult(
            response=received.frame,
            request_wire_bytes=sent,
            response_wire_bytes=received.wire_bytes,
            connect_ms=connect_ms,
            send_ms=send_ms,
            receive_ms=receive_ms,
        )
    finally:
        connection.close()
