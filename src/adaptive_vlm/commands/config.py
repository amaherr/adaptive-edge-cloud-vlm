"""Configuration and environment inspection commands."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.config import load_experiment_config
from adaptive_vlm.environment import collect_environment


def run_validate_config_command(path: Path, output: Path | None) -> int:
    config = load_experiment_config(path)
    payload = config.to_dict()
    if output is not None:
        write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_collect_environment_command(workdir: Path, output: Path | None) -> int:
    payload = collect_environment(workdir)
    if output is not None:
        write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
