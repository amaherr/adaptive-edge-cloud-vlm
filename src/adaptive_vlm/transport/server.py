"""Sequential cloud inference server with a testable request processor."""

from __future__ import annotations

import logging
import socket
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter_ns
from typing import Literal, cast

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
from adaptive_vlm.transport.tensors import TensorInferenceRequest, TensorPayload

LOGGER = logging.getLogger(__name__)
InferenceFunction = Callable[[InferenceRequest], InferenceResult]
TensorInferenceFunction = Callable[[TensorInferenceRequest], InferenceResult]
PromptPrepareFunction = Callable[[str], object]
PreparedTensorInferenceFunction = Callable[
    [TensorInferenceRequest, object], InferenceResult
]
EventCallback = Callable[[str], None]


@dataclass(frozen=True)
class ServerStats:
    completed_requests: int
    rejected_requests: int
    failed_requests: int


@dataclass(frozen=True)
class _PendingPrompt:
    prompt: str
    prepared: object
    expires_at: float


class CloudRequestProcessor:
    """Validate requests and invoke one already-loaded model adapter."""

    def __init__(
        self,
        *,
        protocol_version: int,
        model_fingerprint: str,
        inference_function: InferenceFunction | None = None,
        tensor_inference_function: TensorInferenceFunction | None = None,
        prompt_prepare_function: PromptPrepareFunction | None = None,
        prepared_tensor_inference_function: (
            PreparedTensorInferenceFunction | None
        ) = None,
        pending_timeout_seconds: float = 30.0,
        max_pending_requests: int = 128,
        clock: Callable[[], float] = monotonic,
        event_callback: EventCallback | None = None,
    ) -> None:
        if inference_function is None and tensor_inference_function is None:
            raise ValueError("at least one inference function is required")
        if (prompt_prepare_function is None) != (
            prepared_tensor_inference_function is None
        ):
            raise ValueError(
                "prompt preparation and prepared tensor inference must be paired"
            )
        if pending_timeout_seconds <= 0:
            raise ValueError("pending timeout must be positive")
        if max_pending_requests <= 0:
            raise ValueError("max pending requests must be positive")
        self.protocol_version = protocol_version
        self.model_fingerprint = model_fingerprint
        self._inference_function = inference_function
        self._tensor_inference_function = tensor_inference_function
        self._prompt_prepare_function = prompt_prepare_function
        self._prepared_tensor_inference_function = prepared_tensor_inference_function
        self._pending_timeout_seconds = pending_timeout_seconds
        self._max_pending_requests = max_pending_requests
        self._clock = clock
        self._pending: dict[str, _PendingPrompt] = {}
        self._pending_lock = threading.Lock()
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

    @property
    def pending_requests(self) -> int:
        self.cleanup_pending()
        with self._pending_lock:
            return len(self._pending)

    def cleanup_pending(self) -> int:
        """Discard expired prompt-first state and return the removal count."""

        now = self._clock()
        with self._pending_lock:
            expired = [
                request_id
                for request_id, pending in self._pending.items()
                if pending.expires_at <= now
            ]
            for request_id in expired:
                del self._pending[request_id]
        return len(expired)

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
        if frame.message_type not in {
            MessageType.INFERENCE_REQUEST,
            MessageType.PROMPT_REQUEST,
            MessageType.TENSOR_REQUEST,
        }:
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

        if (
            frame.message_type is MessageType.INFERENCE_REQUEST
            and self._inference_function is None
        ) or (
            frame.message_type is MessageType.TENSOR_REQUEST
            and self._tensor_inference_function is None
            and self._prepared_tensor_inference_function is None
        ) or (
            frame.message_type is MessageType.PROMPT_REQUEST
            and self._prompt_prepare_function is None
        ):
            self._increment("rejected")
            return self._error(
                frame.request_id,
                code="unsupported_execution_path",
                message=f"server is not configured for {frame.message_type.value}",
            )

        try:
            self._emit(
                f"Request {frame.request_id}: received {received.wire_bytes} bytes; "
                "running model inference"
            )
            if frame.message_type is MessageType.TENSOR_REQUEST:
                return self._run_tensor_inference(received)
            if frame.message_type is MessageType.PROMPT_REQUEST:
                return self._run_prompt_prepare(received)
            return self._run_image_inference(received)
        except Exception as exc:  # The process must survive one bad model request.
            LOGGER.exception("inference request %s failed", frame.request_id)
            self._emit(f"Request {frame.request_id}: model inference failed")
            self._increment("failed")
            return self._error(
                frame.request_id,
                code="model_error",
                message=f"{type(exc).__name__}: {str(exc)[:1900]}",
            )

    def _run_image_inference(self, received: ReceivedFrame) -> Frame:
        frame = received.frame
        inference_function = self._inference_function
        assert inference_function is not None
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
            result = inference_function(request)
            server_total_ms = (perf_counter_ns() - started_ns) / 1_000_000

        return self._response(received, result, server_total_ms)

    def _run_tensor_inference(self, received: ReceivedFrame) -> Frame:
        frame = received.frame
        metadata = frame.metadata
        tensor = TensorPayload(
            shape=tuple(cast(list[int], metadata["shape"])),
            dtype=cast(str, metadata["dtype"]),
            byte_order=cast(str, metadata["byte_order"]),
            layout=cast(str, metadata["layout"]),
            data=frame.payload,
        )
        request = TensorInferenceRequest(
            request_id=frame.request_id,
            prompt=cast(str, metadata["prompt"]),
            tensor=tensor,
            pipeline=cast(
                Literal["synchronous", "prompt_first"], metadata["pipeline"]
            ),
        )
        started_ns = perf_counter_ns()
        if request.pipeline == "synchronous":
            inference_function = self._tensor_inference_function
            if inference_function is None:
                self._increment("rejected")
                return self._error(
                    frame.request_id,
                    code="unsupported_execution_path",
                    message="server does not support synchronous tensor inference",
                )
            result = inference_function(request)
        else:
            prepared_function = self._prepared_tensor_inference_function
            if prepared_function is None:
                self._increment("rejected")
                return self._error(
                    frame.request_id,
                    code="unsupported_execution_path",
                    message="server does not support prompt-first tensor inference",
                )
            self.cleanup_pending()
            with self._pending_lock:
                pending = self._pending.pop(frame.request_id, None)
            if pending is None:
                self._increment("rejected")
                return self._error(
                    frame.request_id,
                    code="pending_prompt_not_found",
                    message="prompt-first state is missing or expired",
                )
            if pending.prompt != request.prompt:
                self._increment("rejected")
                return self._error(
                    frame.request_id,
                    code="prompt_mismatch",
                    message="tensor request prompt differs from prepared prompt",
                )
            result = prepared_function(request, pending.prepared)
        server_total_ms = (perf_counter_ns() - started_ns) / 1_000_000
        return self._response(received, result, server_total_ms)

    def _run_prompt_prepare(self, received: ReceivedFrame) -> Frame:
        frame = received.frame
        prepare = self._prompt_prepare_function
        assert prepare is not None
        self.cleanup_pending()
        with self._pending_lock:
            duplicate = frame.request_id in self._pending
            full = len(self._pending) >= self._max_pending_requests
        if duplicate:
            self._increment("rejected")
            return self._error(
                frame.request_id,
                code="duplicate_request_id",
                message="prompt state already exists for this request_id",
            )
        if full:
            self._increment("rejected")
            return self._error(
                frame.request_id,
                code="pending_capacity_exceeded",
                message="server pending prompt capacity is full",
            )
        prompt = cast(str, frame.metadata["prompt"])
        prepared = prepare(prompt)
        with self._pending_lock:
            self._pending[frame.request_id] = _PendingPrompt(
                prompt=prompt,
                prepared=prepared,
                expires_at=self._clock() + self._pending_timeout_seconds,
            )
        self._emit(f"Request {frame.request_id}: prompt state prepared")
        return Frame(
            version=self.protocol_version,
            message_type=MessageType.PROMPT_RESPONSE,
            request_id=frame.request_id,
            metadata={
                "accepted": True,
                "model_fingerprint": self.model_fingerprint,
            },
        )

    def _response(
        self,
        received: ReceivedFrame,
        result: InferenceResult,
        server_total_ms: float,
    ) -> Frame:
        frame = received.frame

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
                    self.processor.cleanup_pending()
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
