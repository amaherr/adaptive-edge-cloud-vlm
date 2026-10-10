from __future__ import annotations

from dataclasses import dataclass

import pytest

from adaptive_vlm.models.features import ProjectedVisionFeatures


@dataclass
class FakeTensor:
    shape: tuple[int, ...] = (729, 2048)
    dtype: str = "torch.bfloat16"
    device: str = "cuda:0"
    item_size: int = 2

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def element_size(self) -> int:
        return self.item_size


def test_projected_features_capture_boundary_metadata() -> None:
    features = ProjectedVisionFeatures.from_tensor(FakeTensor())

    assert features.shape == (729, 2048)
    assert features.dtype == "bfloat16"
    assert features.nbytes == 2_985_984
    assert features.metadata() == {
        "visual_tokens": 729,
        "hidden_size": 2048,
        "dtype": "bfloat16",
        "device": "cuda:0",
        "nbytes": 2_985_984,
    }


@pytest.mark.parametrize("shape", [(728, 2048), (729, 1024), (1, 729, 2048)])
def test_projected_features_reject_wrong_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        ProjectedVisionFeatures.from_tensor(FakeTensor(shape=shape))


def test_projected_features_reject_empty_payload() -> None:
    with pytest.raises(ValueError, match="at least one byte"):
        ProjectedVisionFeatures.from_tensor(FakeTensor(item_size=0))
