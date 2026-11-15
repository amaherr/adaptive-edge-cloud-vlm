"""Opt-in real TCP/process integration; sandboxed CI may skip socket binding."""

from __future__ import annotations

import hashlib
import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from adaptive_vlm.config import TransportConfig
from adaptive_vlm.contracts import InferenceRequest
from adaptive_vlm.execution.cloud import CloudExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer

FINGERPRINT = hashlib.sha256(b"subprocess-model-contract").hexdigest()


def _serve_fake(connection: Connection) -> None:
    adapter = FakeModelAdapter()
    processor = CloudRequestProcessor(
        protocol_version=1, model_fingerprint=FINGERPRINT,
        inference_function=adapter.generate,
    )
    server = CloudServer(
        host="127.0.0.1", port=0, timeout_seconds=5.0, processor=processor,
    )
    try:
        server.start()
    except OSError as exc:
        connection.send(("error", str(exc)))
        connection.close()
        return
    connection.send(("ready", server.port))
    connection.close()
    try:
        server.serve_forever()
    finally:
        server.shutdown()


def test_fake_cloud_server_in_separate_process(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_serve_fake, args=(child,))
    process.start()
    child.close()
    try:
        if not parent.poll(10):
            pytest.fail("cloud subprocess did not report readiness within 10 seconds")
        state, value = parent.recv()
        if state == "error" and "Operation not permitted" in value:
            pytest.skip("sandbox prohibits loopback socket binding")
        assert state == "ready", value
        policy = CloudExecutionPolicy(
            transport=TransportConfig(
                host="127.0.0.1", port=value, timeout_seconds=5.0,
                protocol_version=1,
            ),
            model=ModelMetadata("test/deterministic-vlm", "v1", FINGERPRINT),
        )
        assert policy.health().ready
        image = tmp_path / "image.ppm"
        image.write_bytes(b"test-image")
        request = InferenceRequest("subprocess-1", image, "What is shown?")
        result = policy.infer(request)
        assert result.output_token_ids == FakeModelAdapter().generate(
            request
        ).output_token_ids
    finally:
        parent.close()
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
