from __future__ import annotations

from pathlib import Path

from adaptive_vlm.config import ExecutionMode
from adaptive_vlm.contracts import InferenceRequest
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.models.fake import FakeModelAdapter


def test_fake_adapter_is_deterministic(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"fixture")
    request = InferenceRequest("request-1", image, "What is shown?")
    policy = LocalExecutionPolicy(FakeModelAdapter())

    first = policy.infer(request)
    second = policy.infer(request)

    assert policy.mode is ExecutionMode.EDGE
    assert first.output_text == second.output_text
    assert first.output_token_ids == second.output_token_ids
    assert first.metrics["input_bytes"] == 7


def test_fake_adapter_uses_image_and_prompt(tmp_path: Path) -> None:
    first_image = tmp_path / "first.bin"
    second_image = tmp_path / "second.bin"
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second")
    adapter = FakeModelAdapter()

    first = adapter.generate(InferenceRequest("one", first_image, "prompt"))
    second = adapter.generate(InferenceRequest("two", second_image, "prompt"))

    assert first.output_token_ids != second.output_token_ids

