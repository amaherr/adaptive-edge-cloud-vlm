from __future__ import annotations

import json
import logging

import pytest

from adaptive_vlm.logging import JsonFormatter, configure_logging


def test_json_formatter_includes_structured_fields() -> None:
    record = logging.LogRecord(
        name="adaptive_vlm.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="request complete",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-1"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["message"] == "request complete"
    assert payload["request_id"] == "request-1"
    assert payload["timestamp_utc"].endswith("Z")


def test_configure_logging_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    logging.getLogger("adaptive_vlm.test").info("ready")

    payload = json.loads(capsys.readouterr().err)
    assert payload["message"] == "ready"
