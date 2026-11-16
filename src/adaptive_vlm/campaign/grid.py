"""Schema-2 network/load grid: cells, sub-blocks and their seeded order.

Implements the G2 v2 protocol sections 5-7: every network condition runs at
L0, the loaded conditions also run at each load level, and each cell runs once
per repetition as its own sub-block. Sub-blocks are shuffled with a fixed seed,
re-drawn until no two sub-blocks of the same cell are adjacent.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_vlm.benchmark.edge_load import NO_LOAD
from adaptive_vlm.benchmark.interleaved import PLACEMENTS

SPLITS = ("training", "validation", "evaluation")
_MAX_ORDER_DRAWS = 10_000


@dataclass(frozen=True)
class Condition:
    condition_id: str
    split: str
    rate_mbps: float
    added_rtt_ms: float

    @property
    def egress_delay_ms(self) -> float:
        """Added RTT is split evenly between the two hosts' egress queues."""

        return self.added_rtt_ms / 2


@dataclass(frozen=True)
class SubBlock:
    condition: Condition
    load_level: str
    repetition: int

    @property
    def cell_id(self) -> str:
        return f"{self.condition.condition_id}-{self.load_level}"

    @property
    def subblock_id(self) -> str:
        return f"{self.cell_id}-r{self.repetition}"


@dataclass(frozen=True)
class Grid:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    conditions: Mapping[str, Condition]
    load_levels: tuple[str, ...]
    loaded_conditions: tuple[str, ...]
    repetitions: tuple[int, ...]

    @property
    def order_seed(self) -> int:
        return int(self.payload["order_seed"])

    @property
    def warmups(self) -> int:
        return int(self.payload["warmups_per_subblock"])

    def resolve(self, name: str) -> Path:
        """A path stored in the grid, relative to the grid file."""

        return (self.path.parent / str(self.payload[name])).resolve()

    def cells(self) -> list[tuple[Condition, str]]:
        cells = [(self.conditions[c], NO_LOAD) for c in self.conditions]
        for level in self.load_levels:
            cells += [(self.conditions[c], level) for c in self.loaded_conditions]
        return cells

    def subblocks(self) -> list[SubBlock]:
        return [
            SubBlock(condition, level, repetition)
            for condition, level in self.cells()
            for repetition in self.repetitions
        ]

    def execution_order(self) -> list[SubBlock]:
        """Seeded shuffle with no two sub-blocks of one cell back to back."""

        blocks = self.subblocks()
        rng = random.Random(self.order_seed)
        for _ in range(_MAX_ORDER_DRAWS):
            order = blocks.copy()
            rng.shuffle(order)
            if all(
                first.cell_id != second.cell_id
                for first, second in itertools.pairwise(order)
            ):
                return order
        raise ValueError("no sub-block order without adjacent cells was found")

    def subblock_seed(self, block: SubBlock) -> int:
        """Per-sub-block interleaving seed derived from the order seed."""

        digest = hashlib.sha256(f"{self.order_seed}:{block.subblock_id}".encode())
        return int(digest.hexdigest()[:8], 16)

    def expected_attempts(self, split: Mapping[str, Sequence[str]]) -> dict[str, int]:
        measured = warmups = 0
        for block in self.subblocks():
            ids = len(split[f"{block.condition.split}_request_ids"])
            measured += ids * len(PLACEMENTS)
            warmups += self.warmups * len(PLACEMENTS)
        return {"measured": measured, "warmups": warmups}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_grid(path: Path, *, rate_fallback: bool = False) -> Grid:
    """Load and validate a schema-2 grid.

    ``rate_fallback`` applies the protocol's predeclared substitutions for
    conditions the edge uplink cannot sustain (decided at the on-site pilot).
    """

    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("grid must have schema_version 2")
    raw_conditions = dict(payload["conditions"])
    if rate_fallback:
        for condition_id, replacement in payload.get("rate_fallback", {}).items():
            raw_conditions[condition_id] = {
                **raw_conditions[condition_id],
                **replacement,
            }
    conditions: dict[str, Condition] = {}
    for condition_id, values in raw_conditions.items():
        condition = Condition(
            condition_id=condition_id,
            split=str(values["split"]),
            rate_mbps=float(values["rate_mbps"]),
            added_rtt_ms=float(values["added_rtt_ms"]),
        )
        if condition.split not in SPLITS:
            raise ValueError(f"{condition_id} has unknown split {condition.split}")
        if condition.rate_mbps <= 0 or condition.added_rtt_ms < 0:
            raise ValueError(f"{condition_id} needs a positive rate")
        conditions[condition_id] = condition
    loaded = tuple(payload.get("loaded_conditions", ()))
    unknown = set(loaded) - set(conditions)
    if unknown:
        raise ValueError(f"loaded conditions not in the grid: {sorted(unknown)}")
    levels = tuple(payload.get("load_levels", ()))
    if NO_LOAD in levels:
        raise ValueError("L0 is implicit; list only loaded levels")
    repetitions = tuple(int(r) for r in payload["repetitions"])
    if not repetitions or len(set(repetitions)) != len(repetitions):
        raise ValueError("repetitions must be unique and non-empty")
    if int(payload["netem_limit_packets"]) < 1000:
        raise ValueError("netem_limit_packets must be at least 1000")
    if int(payload["warmups_per_subblock"]) < 0:
        raise ValueError("warmups_per_subblock must be non-negative")
    grid = Grid(
        path=resolved,
        sha256=_sha256(resolved),
        payload=payload,
        conditions=conditions,
        load_levels=levels,
        loaded_conditions=loaded,
        repetitions=repetitions,
    )
    grid.execution_order()  # Fail early if the adjacency constraint is impossible.
    return grid


def check_split(grid: Grid, split: Mapping[str, Any]) -> None:
    """The split must list request IDs for every split and the same conditions."""

    for name in SPLITS:
        if not split.get(f"{name}_request_ids"):
            raise ValueError(f"split has no {name} request IDs")
        listed = set(split.get(f"{name}_network_condition_ids", ()))
        in_grid = {c for c, v in grid.conditions.items() if v.split == name}
        if listed and listed != in_grid:
            raise ValueError(f"split and grid disagree on {name} conditions")
