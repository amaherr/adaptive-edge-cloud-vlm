"""Checks for the physical block runner's frozen ordering and network gate."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

_RUNNER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_physical_profile_block.py")
)
policy_order = _RUNNER["policy_order"]
verify_measurement = _RUNNER["verify_measurement"]


def test_policy_order_rotates_all_three_placements() -> None:
    orders = [policy_order(f"N{index}", 1707) for index in range(1, 7)]
    assert len(set(orders)) == 6
    assert all(set(order) == {"edge", "cloud", "distributed"} for order in orders)
    assert policy_order("N7", 1707) == orders[0]


def test_frozen_grid_and_measurement_gate() -> None:
    grid = json.loads(
        Path("configs/profiles/physical_network_blocks_v1.json").read_text()
    )
    assert set(grid["blocks"]) == {f"N{index}" for index in range(1, 10)}
    measurement = {
        "edge_to_cloud_receiver_bps": 9_410_000,
        "cloud_to_edge_receiver_bps": 9_830_000,
        "echo_median_rtt_ms": 23.338,
    }
    verify_measurement(measurement, 10, 20, 1.635)
    measurement["edge_to_cloud_receiver_bps"] = 3_000_000
    with pytest.raises(RuntimeError, match="outside target band"):
        verify_measurement(measurement, 10, 20, 1.635)


def test_saved_block_plan_resumes_after_json_roundtrip(tmp_path: Path) -> None:
    plan = {
        "block_id": "N1",
        "policy_order": policy_order("N1", 1707),
        "git_sha": "frozen",
    }
    path = tmp_path / "plan.json"
    _RUNNER["save_plan"](path, plan)
    _RUNNER["save_plan"](path, plan)
    with pytest.raises(RuntimeError, match="plan differs"):
        _RUNNER["save_plan"](path, {**plan, "git_sha": "changed"})


def test_interrupted_initialization_preserves_evidence(tmp_path: Path) -> None:
    directory = tmp_path / "edge"
    directory.mkdir()
    (directory / "plan.json").write_text("{}")
    assert _RUNNER["prepare_run"](directory) is False
    assert not directory.exists()
    assert len(list((tmp_path / "initialization_recovery").glob("edge-*"))) == 1
    directory.mkdir()
    (directory / "requests.jsonl").write_text('{"saved":true}\n')
    with pytest.raises(RuntimeError, match="saved attempts"):
        _RUNNER["prepare_run"](directory)
    assert (directory / "requests.jsonl").is_file()
