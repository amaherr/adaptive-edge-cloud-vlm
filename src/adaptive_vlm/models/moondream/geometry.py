"""Pinned Moondream image tiling geometry, computed without the model."""

from __future__ import annotations

import math


def crop_count(height: int, width: int) -> int:
    """Pinned Moondream geometry: 378px crops, 14px patch, margin 4, max 12.

    Mirrors select_tiling without importing/executing model code or loading images.
    Returns the global crop plus the local tiles. Images with either side at
    most 378px always give two crops.
    """
    height, width = height - 112, width - 112
    size, maximum = 266, 12
    if height <= size or width <= size:
        return 2
    min_h, min_w = math.ceil(height / size), math.ceil(width / size)
    if min_h * min_w > maximum:
        ratio = math.sqrt(maximum / (min_h * min_w))
        h, w = max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio))
    else:
        h = max(math.floor(math.sqrt(maximum * height / width)), min_h)
        w = max(math.floor(math.sqrt(maximum * width / height)), min_w)
        if h * w > maximum:
            if w > h:
                w = math.floor(maximum / h)
            else:
                h = math.floor(maximum / w)
        h, w = max(1, h), max(1, w)
    return h * w + 1
