"""The AWS campaign grid matches amendment A2b."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from adaptive_vlm.benchmark.edge_load import IN_PROCESS, load_levels
from adaptive_vlm.campaign.grid import load_grid

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "configs/profiles/aws_network_grid_v2plus.json"


def test_aws_grid_cells_counts_and_levels() -> None:
    grid = load_grid(GRID)
    blocks = grid.subblocks()
    assert len(blocks) == 196
    combos = Counter((b.task, b.load_level, b.cloud_load) for b in blocks)
    for edge in ("L0", "L1"):
        for cloud in ("C0", "C2"):
            assert combos["short", edge, cloud] == 22 * 2
    assert combos["long", "L0", "C0"] + combos["long", "L1", "C0"] == 10 * 2
    assert sum(n for (task, *_), n in combos.items() if task == "long") == 20
    assert grid.expected_attempts() == {"measured": 43_200, "warmups": 980}
    assert grid.repetitions == (0, 1)
    assert len(grid.configurations) == 5

    # Every C2 cell has its C0 twin at the same condition and edge level.
    base = set(grid.cells_by_id())
    assert all(cell[:3] in base for cell in grid.cloud_load_cells)

    assert grid.cloud_loads()["C2"].busy_ms == 20
    edge = load_levels(grid.resolve("edge_load_levels_path"))
    assert set(edge) == {"L1"} == grid.all_load_levels()
    assert edge["L1"].generator == IN_PROCESS and edge["L1"].kernel_mib == 8
    assert len(grid.execution_order()) == 196


def test_aws_pilot_grid() -> None:
    grid = load_grid(ROOT / "configs/profiles/aws_pilot_grid_v2plus.json")
    blocks = grid.subblocks()
    assert len(blocks) == 16
    ids = {b.subblock_id for b in blocks}
    assert {"P02-L0-r0", "P02-L0-C2-r0", "P02-L1-r0", "P02-L1-C2-r0"} <= ids
    assert {"P01-L0-long-r0", "P05-L1-long-r0"} <= ids
    assert all(b.condition.split == "training" for b in blocks)
    assert grid.cloud_loads()["C2"].busy_ms == 20
    assert grid.expected_attempts() == {"measured": 750, "warmups": 80}
