"""Quantiles and rank correlations for training diagnostics."""

from __future__ import annotations

import math
from statistics import mean


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        for index in ordered[start:stop]:
            result[index] = (start + stop - 1) / 2
        start = stop
    return result


def correlation(x: list[float], y: list[float]) -> float | None:
    """Spearman association with average ranks for ties; no significance claim."""
    if len(x) < 3:
        return None
    xr, yr = ranks(x), ranks(y)
    xm, ym = mean(xr), mean(yr)
    denominator = math.sqrt(
        sum((v - xm) ** 2 for v in xr) * sum((v - ym) ** 2 for v in yr)
    )
    return (
        sum((a - xm) * (b - ym) for a, b in zip(xr, yr, strict=True)) / denominator
        if denominator
        else None
    )
