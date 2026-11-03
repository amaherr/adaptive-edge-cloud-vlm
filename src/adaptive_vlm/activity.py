"""Optional host-visible activity marker for the cloud idle watchdog."""

import os
from pathlib import Path


def mark_activity() -> None:
    """Touch only when configured; configuration errors must remain visible."""
    destination = os.environ.get("ADAPTIVE_VLM_HEARTBEAT")
    if destination:
        Path(destination).touch()
