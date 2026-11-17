"""Schema-2 network/load grid: cells, sub-blocks and their seeded order.

Implements the G2 v2 protocol sections 3.5 and 5-7: every network condition
runs at L0, the loaded conditions also run at each load level, and each cell
runs once per repetition as its own sub-block. These cells use the primary
(short-answer) task. Extra tasks (the long-answer bucket) name their own
experiment config, split and cells as ``[condition, load level]`` pairs.
Sub-blocks are shuffled with a fixed seed, re-drawn until no two sub-blocks of
the same cell are adjacent.
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

from adaptive_vlm.benchmark.cloud_load import NO_CLOUD_LOAD
from adaptive_vlm.benchmark.edge_load import NO_LOAD, LoadLevel, load_levels
from adaptive_vlm.benchmark.interleaved import PLACEMENTS
from adaptive_vlm.scheduler.configuration import (
    SPEC_BY_ID,
    SPEC_BY_PLACEMENT,
    ConfigurationId,
)

SPLITS = ("training", "validation", "evaluation")
PRIMARY_TASK = "short"
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
class Task:
    name: str
    task_bucket: str
    experiment_config: Path
    split: Path
    cells: tuple[tuple[str, str], ...]  # empty for the primary task


@dataclass(frozen=True)
class SubBlock:
    condition: Condition
    load_level: str
    repetition: int
    task: str = PRIMARY_TASK
    cloud_load: str = NO_CLOUD_LOAD

    @property
    def cell_id(self) -> str:
        base = f"{self.condition.condition_id}-{self.load_level}"
        if self.task != PRIMARY_TASK:
            base = f"{base}-{self.task}"
        # C0 IDs are unchanged from G2 v2; cloud-loaded cells gain a suffix.
        return base if self.cloud_load == NO_CLOUD_LOAD else f"{base}-{self.cloud_load}"

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
    tasks: Mapping[str, Task]
    placements: tuple[str, ...] = PLACEMENTS
    cloud_load_level: str | None = None
    cloud_load_cells: tuple[tuple[str, str, str], ...] = ()

    @property
    def configurations(self) -> tuple[str, ...]:
        return tuple(SPEC_BY_PLACEMENT[p].id.value for p in self.placements)

    @property
    def order_seed(self) -> int:
        return int(self.payload["order_seed"])

    @property
    def warmups(self) -> int:
        return int(self.payload["warmups_per_subblock"])

    def resolve(self, name: str) -> Path:
        """A path stored in the grid, relative to the grid file."""

        return (self.path.parent / str(self.payload[name])).resolve()

    def cells(self) -> list[tuple[Condition, str, str]]:
        cells = [(self.conditions[c], NO_LOAD, PRIMARY_TASK) for c in self.conditions]
        for level in self.load_levels:
            cells += [
                (self.conditions[c], level, PRIMARY_TASK)
                for c in self.loaded_conditions
            ]
        for task in self.tasks.values():
            cells += [(self.conditions[c], level, task.name) for c, level in task.cells]
        return cells

    def cloud_load(self) -> LoadLevel | None:
        """The calibrated cloud-load level, or None if the grid has no C1 cells.

        Raises if the grid names a level the levels file does not define yet
        (C1 is filled in only after calibration, A2a rule 6).
        """

        if self.cloud_load_level is None:
            return None
        path = (self.path.parent / self.payload["cloud_load"]["levels_path"]).resolve()
        levels = load_levels(path, no_load=NO_CLOUD_LOAD)
        if self.cloud_load_level not in levels:
            raise ValueError(
                f"cloud-load level {self.cloud_load_level} is not calibrated in {path}"
            )
        return levels[self.cloud_load_level]

    def cells_by_id(self) -> list[tuple[str, str, str]]:
        """C0 cells as ``(condition ID, edge load level, task)``."""

        return [(c.condition_id, level, task) for c, level, task in self.cells()]

    def all_load_levels(self) -> set[str]:
        return {level for _, level, _ in self.cells()} - {NO_LOAD}

    def subblocks(self) -> list[SubBlock]:
        blocks = [
            SubBlock(condition, level, repetition, task)
            for condition, level, task in self.cells()
            for repetition in self.repetitions
        ]
        if self.cloud_load_level is not None:
            blocks += [
                SubBlock(
                    self.conditions[condition],
                    level,
                    repetition,
                    task,
                    self.cloud_load_level,
                )
                for condition, level, task in self.cloud_load_cells
                for repetition in self.repetitions
            ]
        return blocks

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

    def load_split(self, task: str = PRIMARY_TASK) -> dict[str, Any]:
        split: dict[str, Any] = json.loads(
            self.tasks[task].split.read_text(encoding="utf-8")
        )
        return split

    def expected_attempts(
        self, splits: Mapping[str, Mapping[str, Sequence[str]]] | None = None
    ) -> dict[str, int]:
        """Measured attempts and warmups; ``splits`` maps task name to split."""

        loaded = splits or {name: self.load_split(name) for name in self.tasks}
        measured = warmups = 0
        for block in self.subblocks():
            ids = len(loaded[block.task][f"{block.condition.split}_request_ids"])
            measured += ids * len(self.placements)
            warmups += self.warmups * len(self.placements)
        return {"measured": measured, "warmups": warmups}


def _relative(grid_path: Path, value: str) -> Path:
    return (grid_path.parent / value).resolve()


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
    tasks = {
        PRIMARY_TASK: Task(
            name=PRIMARY_TASK,
            task_bucket=str(
                payload.get("task_bucket", "vqa-short-answer-32-token-budget")
            ),
            experiment_config=_relative(resolved, payload["experiment_config_path"]),
            split=_relative(resolved, payload["split_path"]),
            cells=(),
        )
    }
    for name, values in payload.get("extra_tasks", {}).items():
        if name in tasks:
            raise ValueError(f"duplicate task name: {name}")
        cells = tuple((str(c), str(level)) for c, level in values["cells"])
        if len(set(cells)) != len(cells) or not cells:
            raise ValueError(f"task {name} needs unique, non-empty cells")
        missing = {c for c, _ in cells} - set(conditions)
        if missing:
            raise ValueError(f"task {name} names unknown conditions: {sorted(missing)}")
        tasks[name] = Task(
            name=name,
            task_bucket=str(values["task_bucket"]),
            experiment_config=_relative(resolved, values["experiment_config_path"]),
            split=_relative(resolved, values["split_path"]),
            cells=cells,
        )
    configurations = payload.get(
        "configurations", [SPEC_BY_PLACEMENT[p].id.value for p in PLACEMENTS]
    )
    try:
        specs = [SPEC_BY_ID[ConfigurationId(value)] for value in configurations]
    except ValueError as exc:
        raise ValueError(f"unknown configuration in grid: {exc}") from exc
    if len(set(configurations)) != len(configurations):
        raise ValueError("grid configurations must be unique")
    if not {spec.placement for spec in specs} >= set(PLACEMENTS):
        raise ValueError(
            "grid configurations must include edge/full, cloud/raw_image and "
            "distributed/projected/synchronous"
        )
    cloud = payload.get("cloud_load")
    cloud_level: str | None = None
    cloud_cells: tuple[tuple[str, str, str], ...] = ()
    if cloud is not None:
        cloud_level = str(cloud["level"])
        if cloud_level == NO_CLOUD_LOAD:
            raise ValueError("C0 is implicit; name the cloud-load level")
        cloud_cells = tuple((str(c), str(e), str(t)) for c, e, t in cloud["cells"])
        if not cloud_cells or len(set(cloud_cells)) != len(cloud_cells):
            raise ValueError("cloud-load cells must be unique and non-empty")
        base_cells = {
            (condition, level, task)
            for condition, level, task in Grid(
                path=resolved,
                sha256="",
                payload=payload,
                conditions=conditions,
                load_levels=levels,
                loaded_conditions=loaded,
                repetitions=repetitions,
                tasks=tasks,
            ).cells_by_id()
        }
        uncovered = set(cloud_cells) - base_cells
        if uncovered:
            raise ValueError(
                f"cloud-load cells need the same C0 cell: {sorted(uncovered)}"
            )
    grid = Grid(
        path=resolved,
        sha256=_sha256(resolved),
        payload=payload,
        conditions=conditions,
        load_levels=levels,
        loaded_conditions=loaded,
        repetitions=repetitions,
        tasks=tasks,
        # Run directories in the canonical order, whatever order the grid lists.
        placements=tuple(
            p for p in SPEC_BY_PLACEMENT if p in {spec.placement for spec in specs}
        ),
        cloud_load_level=cloud_level,
        cloud_load_cells=cloud_cells,
    )
    grid.execution_order()  # Fail early if the adjacency constraint is impossible.
    return grid


def check_split(grid: Grid, split: Mapping[str, Any], task: str = PRIMARY_TASK) -> None:
    """The split must list request IDs for every split and the same conditions.

    For an extra task, the request IDs must be the primary task's IDs of the
    same split with ``-<task>`` appended (the same questions and images).
    """

    if task == PRIMARY_TASK:
        used = set(grid.conditions)
    else:
        used = {c for c, _ in grid.tasks[task].cells}
    for name in SPLITS:
        if not split.get(f"{name}_request_ids"):
            raise ValueError(f"{task} split has no {name} request IDs")
        listed = set(split.get(f"{name}_network_condition_ids", ()))
        in_grid = {
            c for c, v in grid.conditions.items() if v.split == name and c in used
        }
        if listed and listed != in_grid:
            raise ValueError(f"{task} split and grid disagree on {name} conditions")
    if task != PRIMARY_TASK:
        primary = grid.load_split()
        for name in SPLITS:
            base = {
                rid.removesuffix(f"-{task}") for rid in split[f"{name}_request_ids"]
            }
            if not all(
                rid.endswith(f"-{task}") for rid in split[f"{name}_request_ids"]
            ):
                raise ValueError(f"{task} request IDs must end with -{task}")
            if not base <= set(primary[f"{name}_request_ids"]):
                raise ValueError(
                    f"{task} {name} IDs are not a subset of the primary split"
                )
