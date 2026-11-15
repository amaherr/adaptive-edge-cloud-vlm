"""Small nonnegative least-squares fits used by V2 calibration."""

from __future__ import annotations

import itertools
import math
from typing import Any


def _solve(matrix: list[list[float]], values: list[float]) -> list[float] | None:
    size = len(values)
    a = [[*row, value] for row, value in zip(matrix, values, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda i: abs(a[i][column]))
        if abs(a[pivot][column]) < 1e-12:
            return None
        a[column], a[pivot] = a[pivot], a[column]
        divisor = a[column][column]
        a[column] = [v / divisor for v in a[column]]
        for row in range(size):
            if row == column:
                continue
            multiplier = a[row][column]
            a[row] = [
                x - multiplier * y for x, y in zip(a[row], a[column], strict=True)
            ]
    return [row[-1] for row in a]


def nonnegative_fit(
    x: list[list[float]], y: list[float], *, alpha: float
) -> list[float]:
    """Enumerate active sets for <=3 coefficients; no target/residual clipping."""
    if not x or len(x) != len(y) or len(x[0]) > 3:
        raise ValueError("small nonnegative fit requires nonempty aligned inputs")
    width = len(x[0])
    best = [0.0] * width
    best_loss = sum(v * v for v in y)
    for active_count in range(1, width + 1):
        for subset in itertools.combinations(range(width), active_count):
            a = [
                [
                    sum(row[i] * row[j] for row in x)
                    + (alpha if i == j and i != 0 else 0)
                    for j in subset
                ]
                for i in subset
            ]
            b = [
                sum(row[i] * value for row, value in zip(x, y, strict=True))
                for i in subset
            ]
            solved = _solve(a, b)
            if solved is None or any(not math.isfinite(v) or v < 0 for v in solved):
                continue
            coefficients = [0.0] * width
            for index, value in zip(subset, solved, strict=True):
                coefficients[index] = value
            loss = sum(
                (sum(c * r for c, r in zip(coefficients, row, strict=True)) - value)
                ** 2
                for row, value in zip(x, y, strict=True)
            ) + alpha * sum(v * v for v in coefficients[1:])
            if loss < best_loss:
                best, best_loss = coefficients, loss
    return best


def _constant(value: float) -> dict[str, Any]:
    return {"intercept_ms": value, "slopes": {}, "scales": {}}
