"""int8 admission rule (G2 v2 protocol, amendment A2a rule 4).

On the 200 calibration questions with prompt P1, int8 is admitted only if
both hold:

- official VQA accuracy within ``MAX_ACCURACY_DROP_PP`` of lossless, scored
  both unchanged and lowercased (both must pass);
- at least ``MIN_IDENTICAL`` of answers identical to lossless after stripping
  whitespace and case-folding.

On the 200 E1 images, token agreement and the length distribution are
reported with no gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

MAX_ACCURACY_DROP_PP = 1.0
MIN_IDENTICAL = 0.95
SCORINGS = ("unchanged", "lowercased")


def _normalized(answer: str) -> str:
    return answer.strip().casefold()


def identical_share(lossless: Mapping[str, str], int8: Mapping[str, str]) -> float:
    if set(lossless) != set(int8) or not lossless:
        raise ValueError("both runs must answer the same, non-empty question set")
    same = sum(_normalized(lossless[key]) == _normalized(int8[key]) for key in lossless)
    return same / len(lossless)


def short_decision(
    accuracy: Mapping[str, Mapping[str, float]], identical: float
) -> dict[str, Any]:
    """``accuracy[scoring][run]`` in percent, for runs lossless and int8."""

    checks: dict[str, bool] = {}
    drops: dict[str, float] = {}
    for scoring in SCORINGS:
        drop = accuracy[scoring]["lossless"] - accuracy[scoring]["int8"]
        drops[scoring] = drop
        checks[f"accuracy_{scoring}_within_{MAX_ACCURACY_DROP_PP}pp"] = (
            drop <= MAX_ACCURACY_DROP_PP
        )
    checks[f"identical_at_least_{MIN_IDENTICAL:.0%}"] = identical >= MIN_IDENTICAL
    return {
        "admitted": all(checks.values()),
        "checks": checks,
        "accuracy_drop_pp": drops,
        "identical_share": identical,
        "rule": {
            "max_accuracy_drop_pp": MAX_ACCURACY_DROP_PP,
            "min_identical_share": MIN_IDENTICAL,
            "scorings": list(SCORINGS),
        },
    }


def long_report(
    lossless: Mapping[str, Sequence[int]], int8: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    """Token agreement and lengths for the long-answer task (no gate)."""

    if set(lossless) != set(int8) or not lossless:
        raise ValueError("both runs must cover the same, non-empty image set")
    positions = []
    prefixes = []
    for key in sorted(lossless):
        a, b = list(lossless[key]), list(int8[key])
        longest = max(len(a), len(b), 1)
        positions.append(sum(x == y for x, y in zip(a, b, strict=False)) / longest)
        prefix = 0
        for x, y in zip(a, b, strict=False):
            if x != y:
                break
            prefix += 1
        prefixes.append(prefix)

    def lengths(run: Mapping[str, Sequence[int]]) -> dict[str, float]:
        values = [len(tokens) for tokens in run.values()]
        return {"mean": mean(values), "median": median(values), "min": min(values)}

    return {
        "images": len(lossless),
        "identical_token_sequences": sum(
            list(lossless[key]) == list(int8[key]) for key in lossless
        ),
        "mean_position_agreement": mean(positions),
        "mean_common_prefix_tokens": mean(prefixes),
        "lengths": {"lossless": lengths(lossless), "int8": lengths(int8)},
        "gated": False,
    }
