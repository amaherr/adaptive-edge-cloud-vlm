from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from adaptive_vlm import environment as environment_module
from adaptive_vlm.environment import collect_environment


def test_collect_environment_has_required_sections() -> None:
    environment = collect_environment(Path(__file__).parents[1])

    assert environment["host"]["logical_cpu_count"]
    assert environment["python"]["version"]
    assert isinstance(environment["gpu"]["available"], bool)
    assert environment["git"]["available"] is True
    assert len(environment["git"]["commit"]) == 40


def test_nvidia_inventory_parses_multiple_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="GPU One, 4096, 610.62\nGPU Two, 8192, 610.62\n",
        stderr="",
    )
    monkeypatch.setattr(environment_module, "_run", lambda command: completed)

    inventory = environment_module._nvidia_gpu()

    assert inventory["available"] is True
    assert inventory["devices"][0]["memory_total_mib"] == 4096
    assert inventory["devices"][1]["name"] == "GPU Two"


def test_nvidia_inventory_handles_missing_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment_module, "_run", lambda command: None)

    inventory = environment_module._nvidia_gpu()

    assert inventory == {
        "available": False,
        "reason": "nvidia-smi is not installed",
    }


def test_nvidia_inventory_preserves_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=1,
        stdout="GPU access blocked by the operating system\n",
        stderr="",
    )
    monkeypatch.setattr(environment_module, "_run", lambda command: completed)

    inventory = environment_module._nvidia_gpu()

    assert inventory["available"] is False
    assert inventory["reason"] == "GPU access blocked by the operating system"


