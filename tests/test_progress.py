from __future__ import annotations

from io import StringIO

import pytest

from adaptive_vlm.progress import ProgressReporter


def test_phase_reports_start_and_completion() -> None:
    output = StringIO()
    reporter = ProgressReporter(stream=output)

    with reporter.phase("Loading model"):
        pass

    messages = output.getvalue()
    assert "Loading model..." in messages
    assert "Loading model completed in" in messages


def test_details_require_verbose_mode() -> None:
    normal_output = StringIO()
    verbose_output = StringIO()

    ProgressReporter(stream=normal_output).detail("fingerprint")
    ProgressReporter(verbose=True, stream=verbose_output).detail("fingerprint")

    assert normal_output.getvalue() == ""
    assert "fingerprint" in verbose_output.getvalue()


def test_phase_reports_failure() -> None:
    output = StringIO()
    reporter = ProgressReporter(stream=output)

    with pytest.raises(RuntimeError, match="failure"), reporter.phase("Inference"):
        raise RuntimeError("failure")

    assert "Inference failed after" in output.getvalue()
