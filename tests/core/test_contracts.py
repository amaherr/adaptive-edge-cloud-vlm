from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_vlm.config import ExecutionMode
from adaptive_vlm.contracts import (
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    RequestRecord,
    RunManifest,
)


def test_request_requires_a_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        InferenceRequest("request-1", Path("image.jpg"), "  ")


def test_result_rejects_negative_token_ids() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        InferenceResult("request-1", "answer", (1, -1))


def test_request_record_from_result_is_json_serializable() -> None:
    result = InferenceResult(
        request_id="request-1",
        output_text="answer",
        output_token_ids=(1, 2),
        metrics={"total_ms": 2.5},
    )

    record = RequestRecord.from_result(
        run_id="run-1", mode=ExecutionMode.EDGE, result=result
    )

    serialized = json.dumps(record.to_dict())
    assert '"status": "ok"' in serialized
    assert record.status is InferenceStatus.OK


def test_failed_record_requires_error_type() -> None:
    with pytest.raises(ValueError, match="error_type"):
        RequestRecord(
            schema_version=1,
            run_id="run-1",
            request_id="request-1",
            mode=ExecutionMode.CLOUD,
            status=InferenceStatus.TIMEOUT,
            output_text=None,
            output_token_ids=(),
            metrics={},
        )


def test_manifest_requires_explicit_utc_timestamp() -> None:
    with pytest.raises(ValueError, match="ending in Z"):
        RunManifest(
            schema_version=1,
            run_id="run-1",
            created_at_utc="2026-10-10T12:00:00",
            git_commit=None,
            git_dirty=None,
            config={},
            environment={},
        )

