#!/usr/bin/env python3
"""Print a deterministic JSON inventory for one safetensors artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_vlm.model_artifact import inspect_safetensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_safetensors(args.artifact), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
