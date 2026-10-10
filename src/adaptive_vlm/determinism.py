"""Deterministic seed handling without mandatory numerical dependencies."""

from __future__ import annotations

import importlib
import os
import random
from typing import Any


def seed_everything(seed: int) -> tuple[str, ...]:
    """Seed Python and any installed NumPy/PyTorch runtimes.

    Returns the names of seeded providers so each run can record what was
    actually controlled.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    providers = ["python"]

    numpy = _optional_import("numpy")
    if numpy is not None:
        numpy.random.seed(seed)
        providers.append("numpy")

    torch = _optional_import("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        providers.append("torch")

    return tuple(providers)


def _optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None

