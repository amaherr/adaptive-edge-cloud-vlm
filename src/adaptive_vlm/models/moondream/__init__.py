"""Pinned Moondream2 adapter, artifact verification and weight residency."""

from adaptive_vlm.models.moondream.adapter import (
    ComponentRole,
    MoondreamAdapter,
    MoondreamRun,
)
from adaptive_vlm.models.moondream.residency import (
    WEIGHT_RESIDENCIES,
    WeightResidency,
)

__all__ = [
    "WEIGHT_RESIDENCIES",
    "ComponentRole",
    "MoondreamAdapter",
    "MoondreamRun",
    "WeightResidency",
]
