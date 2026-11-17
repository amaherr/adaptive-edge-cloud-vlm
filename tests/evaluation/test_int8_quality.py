"""int8 admission rule (amendment A2a rule 4)."""

from __future__ import annotations

import pytest

from adaptive_vlm.evaluation.int8_quality import (
    identical_share,
    long_report,
    short_decision,
)


def _accuracy(unchanged: float, lowercased: float) -> dict[str, dict[str, float]]:
    return {
        "unchanged": {"lossless": 56.7, "int8": unchanged},
        "lowercased": {"lossless": 79.2, "int8": lowercased},
    }


def test_admitted_only_when_both_scorings_and_agreement_pass() -> None:
    assert short_decision(_accuracy(56.0, 78.5), 0.96)["admitted"]
    assert short_decision(_accuracy(55.7, 78.2), 0.95)["admitted"]  # boundaries
    unchanged = short_decision(_accuracy(55.6, 79.2), 0.99)
    assert not unchanged["admitted"]
    assert unchanged["checks"]["accuracy_unchanged_within_1.0pp"] is False
    assert not short_decision(_accuracy(56.7, 78.0), 0.99)["admitted"]
    assert not short_decision(_accuracy(56.7, 79.2), 0.949)["admitted"]
    assert short_decision(_accuracy(57.5, 80.0), 0.97)["admitted"]  # int8 better


def test_identical_share_strips_and_casefolds() -> None:
    lossless = {"1": "Yes", "2": "two ", "3": "red"}
    assert identical_share(lossless, {"1": "yes", "2": "Two", "3": "blue"}) == 2 / 3
    with pytest.raises(ValueError, match="same"):
        identical_share(lossless, {"1": "yes"})


def test_long_report_measures_agreement_without_a_gate() -> None:
    report = long_report(
        {"a": [1, 2, 3, 4], "b": [5, 6]}, {"a": [1, 2, 9, 4], "b": [5, 6]}
    )
    assert report["identical_token_sequences"] == 1
    assert report["mean_common_prefix_tokens"] == 2
    assert report["mean_position_agreement"] == pytest.approx((3 / 4 + 1) / 2)
    assert report["lengths"]["lossless"]["mean"] == 3 and not report["gated"]
