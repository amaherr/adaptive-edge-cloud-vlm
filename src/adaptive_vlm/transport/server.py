"""Sequential cloud inference server with a testable request processor."""

from __future__ import annotations

import logging
import socket
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.transport.protocol import (
    DEFAULT_MAX_HEADER_BYTES,
    DEFAULT_MAX_PAYLOAD_BYTES,
    Frame,
    MessageType,
    PeerDisconnected,
    ProtocolError,
    ReceivedFrame,
    receive_frame,
    send_frame,
)

LOGGER = logging.getLogger(__name__)
InferenceFunction = Callable[[InferenceRequest], InferenceResult]
EventCallback = Callable[[str], None]


@dataclass(frozen=True)
class ServerStats:
    completed_requests: int
    rejected_requests: int
    failed_requests: int


class CloudRequestProcessor:
    """Validate requests and invoke one already-loaded model adapter."""

    def __init__(
        self,
        *,
        protocol_version: int,
        model_fingerprint: str,
        inference_function: InferenceFunction,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.protocol_version = protocol_version
        self.model_fingerprint = model_fingerprint
        self._inference_function = inference_function
        self._event_callback = event_callback
        self._completed_requests = 0
        self._rejected_requests = 0
        self._failed_requests = 0
        self._stats_lock = threading.Lock()

    @property
    def stats(self) -> ServerStats:
        with self._stats_lock:
            return ServerStats(
                completed_requests=self._completed_requests,
                rejected_requests=self._rejected_requests,
                failed_requests=self._failed_requests,
            )

    def handle(self, received: ReceivedFrame) -> Frame:
        frame = received.frame
        if frame.message_type is MessageType.HEALTH_REQUEST:
            return Frame(
                version=self.protocol_version,
                message_type=MessageType.HEALTH_RESPONSE,
                request_id=frame.request_id,
                metadata={
                    "ready": True,
                    "model_fingerprint": self.model_fingerprint,
                },
            )
        if frame.message_type is not MessageType.INFERENCE_REQUEST:
            self._increment("rejected")
            return self._error(
                frame.request_id,
                code="unsupported_message_type",
                message=f"server does not accept {frame.message_type.value}",
            )
        if frame.metadata["model_fingerprint"] != self.model_fingerprint:
            self._increment("rejected")
            return self._error(
                frame.request_id,
                code="model_fingerprint_mismatch",
                message="requested model fingerprint is not loaded on this server",
            )

        try:
            self._emit(
                f"Request {frame.request_id}: received {received.wire_bytes} bytes; "
                "running model inference"
            )
            return self._run_inference(received)
        except Exception as exc:  # The process must survive one bad model request.
            LOGGER.exception("inference request %s failed", frame.request_id)
            self._emit(f"Request {frame.request_id}: model inference failed")
            self._increment("failed")
            return self._error(
                frame.request_id,
                code="model_error",
                message=f"{type(exc).__name__}: {str(exc)[:1900]}",
            )

    def _run_inference(self, received: ReceivedFrame) -> Frame:
        frame = received.frame
        suffix = str(frame.metadata["image_suffix"])
        with tempfile.TemporaryDirectory(prefix="adaptive-vlm-") as directory:
            image_path = Path(directory) / f"input{suffix}"
            image_path.write_bytes(frame.payload)
            request = InferenceRequest(
                request_id=frame.request_id,
                image_path=image_path,
                prompt=str(frame.metadata["prompt"]),
            )
            started_ns = perf_counter_ns()
            result = self._inference_function(request)
            server_total_ms = (perf_counter_ns() - started_ns) / 1_000_000

        if result.request_id != frame.request_id:
            raise ValueError("model result request_id does not match request")
        metrics = {
            f"server_model_{key}": value for key, value in result.metrics.items()
        }
        metrics.update(
            {
                "server_total_ms": server_total_ms,
                "server_request_wire_bytes": received.wire_bytes,
            }
        )
        self._increment("completed")
        self._emit(
            f"Request {frame.request_id}: completed in {server_total_ms / 1000:.1f}s "
            f"with {len(result.output_token_ids)} output tokens"
        )
        return Frame(
            version=self.protocol_version,
            message_type=MessageType.INFERENCE_RESPONSE,
            request_id=frame.request_id,
            metadata={
                "model_fingerprint": self.model_fingerprint,
                "output_text": result.output_text,
                "output_token_ids": list(result.output_token_ids),
                "metrics": metrics,
            },
        )

    def _error(self, request_id: str, *, code: str, message: str) -> Frame:
        return Frame(
            version=self.protocol_version,
            message_type=MessageType.ERROR,
            request_id=request_id,
            metadata={"code": code, "message": message, "retryable": False},
        )

    def _increment(self, counter: str) -> None:
        with self._stats_lock:
            if counter == "completed":
                self._completed_requests += 1
            elif counter == "rejected":
                self._rejected_requests += 1
            else:
                self._failed_requests += 1

    def _emit(self, message: str) -> None:
        if self._event_callback is not None:
            self._event_callback(message)


class CloudServer:
    """Serve one framed request per TCP connection until graceful shutdown."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_seconds: float,
        processor: CloudRequestProcessor,
        max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.processor = processor
        self.max_header_bytes = max_header_bytes
        self.max_payload_bytes = max_payload_bytes
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def start(self) -> None:
        if self._listener is not None:
            raise RuntimeError("server is already started")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen()
            listener.settimeout(min(self.timeout_seconds, 0.25))
        except Exception:
            listener.close()
            raise
        self.port = int(listener.getsockname()[1])
        self._listener = listener
        self._stop.clear()
        self._ready.set()

    def serve_forever(self) -> None:
        if self._listener is None:
            self.start()
        listener = self._listener
        assert listener is not None
        try:
            while not self._stop.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                with connection:
                    connection.settimeout(self.timeout_seconds)
                    self._serve_connection(connection)
        finally:
            listener.close()
            self._listener = None
            self._ready.clear()

    def shutdown(self) -> None:
        self._stop.set()
        listener = self._listener
        if listener is not None:
            listener.close()

    def _serve_connection(self, connection: socket.socket) -> None:
        try:
            received = receive_frame(
                connection,
                expected_version=self.processor.protocol_version,
                max_header_bytes=self.max_header_bytes,
                max_payload_bytes=self.max_payload_bytes,
            )
            response = self.processor.handle(received)
            send_frame(
                connection,
                response,
                max_header_bytes=self.max_header_bytes,
                max_payload_bytes=self.max_payload_bytes,
            )
        except (PeerDisconnected, ProtocolError, TimeoutError, OSError) as exc:
            LOGGER.warning("connection rejected: %s", exc)
