"""Pre-execution features for r2: crop count and the slow-start network term."""

from __future__ import annotations

import math

# Pinned Moondream image_crops.py: 378 px crops of 14 px patches, an overlap
# margin of 4 patches on each side, at most 12 crops.
CROP_SIZE = 378
PATCH = 14
MARGIN_PATCHES = 4
MAX_CROPS = 12
_MARGIN_PX = 2 * PATCH * MARGIN_PATCHES  # 112
_WINDOW_PX = (CROP_SIZE // PATCH - 2 * MARGIN_PATCHES) * PATCH  # 266


def select_tiling(
    height: int, width: int, crop_size: int, max_crops: int
) -> tuple[int, int]:
    """Exact reimplementation of the pinned model's ``select_tiling``."""

    if height <= crop_size or width <= crop_size:
        return (1, 1)
    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)
    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))
    h_tiles = max(math.floor(math.sqrt(max_crops * height / width)), min_h)
    w_tiles = max(math.floor(math.sqrt(max_crops * width / height)), min_w)
    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)
    return (max(1, h_tiles), max(1, w_tiles))


def crop_count(width: int, height: int) -> int:
    """Crops the vision encoder runs: the local tiles plus the global crop."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    tiles = select_tiling(
        height - _MARGIN_PX, width - _MARGIN_PX, _WINDOW_PX, MAX_CROPS
    )
    return tiles[0] * tiles[1] + 1


def slow_start_transfer_ms(
    payload_bytes: float, uplink_bps: float, rtt_ms: float, initial_window: float
) -> float:
    """Time to deliver ``payload_bytes`` on a fresh TCP connection.

    The window starts at ``initial_window`` bytes and doubles every round.
    While it is below the bandwidth-delay product a round costs
    max(RTT, round bytes / rate); the last round costs only its bytes / rate.
    Once the window covers the bandwidth-delay product, the rest flows at the
    link rate.
    """

    if payload_bytes < 0 or uplink_bps <= 0 or rtt_ms < 0 or initial_window <= 0:
        raise ValueError("invalid transfer inputs")
    rate = uplink_bps / 8000.0  # bytes per ms
    bdp = rate * rtt_ms
    sent, window, elapsed = 0.0, float(initial_window), 0.0
    while sent < payload_bytes:
        if window >= bdp:
            return elapsed + (payload_bytes - sent) / rate
        chunk = min(window, payload_bytes - sent)
        if sent + chunk >= payload_bytes:
            return elapsed + chunk / rate
        elapsed += max(rtt_ms, chunk / rate)
        sent += chunk
        window *= 2
    return elapsed


def network_phi_ms(
    payload_bytes: float, uplink_bps: float, rtt_ms: float, initial_window: float
) -> float:
    """Handshake RTT + slow-start transfer + one RTT for the last byte and reply."""

    return 2 * rtt_ms + slow_start_transfer_ms(
        payload_bytes, uplink_bps, rtt_ms, initial_window
    )
