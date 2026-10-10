"""Model adapters."""

from adaptive_vlm.models.base import ModelAdapter, ModelMetadata
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.models.features import ProjectedVisionFeatures

__all__ = [
    "FakeModelAdapter",
    "ModelAdapter",
    "ModelMetadata",
    "ProjectedVisionFeatures",
]

