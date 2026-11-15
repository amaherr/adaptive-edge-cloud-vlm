from __future__ import annotations

import random

import pytest

from adaptive_vlm.determinism import seed_everything


def test_seed_everything_repeats_python_sequence() -> None:
    providers = seed_everything(17)
    first = [random.random() for _ in range(3)]
    seed_everything(17)
    second = [random.random() for _ in range(3)]

    assert "python" in providers
    assert first == second


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)

