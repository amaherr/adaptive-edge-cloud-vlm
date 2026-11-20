"""Request-cell units for r2 (specification section 2).

A unit is one request ID in one cell, for one configuration. Its inputs are
the pre-execution snapshot values (network probe values are the median over
the unit's repetitions, each taken before its sub-block); its targets are the
medians of the successful repetitions' stage times.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from adaptive_vlm.prediction.r2.features import crop_count

CONFIGURATIONS = (
    "edge/full",
    "cloud/raw_image",
    "distributed/projected/synchronous",
    "distributed/projected/prompt_first",
    "distributed/projected/int8",
)
EDGE = "edge/full"
CLOUD = "cloud/raw_image"
SPLITS = CONFIGURATIONS[2:]
REMOTE = CONFIGURATIONS[1:]
TARGETS = ("e2e", "edge", "server", "transport", "residual", "request_bytes")


@dataclass(frozen=True)
class UnitInputs:
    request_id: str
    cell_id: str
    condition_id: str
    task: str
    edge_load: str
    cloud_load: str
    width: int
    height: int
    image_bytes: int
    uplink_bps: float
    rtt_ms: float

    @property
    def crops(self) -> int:
        return crop_count(self.width, self.height)

    @property
    def key(self) -> tuple[str, str]:
        return (self.request_id, self.cell_id)


@dataclass
class Unit:
    inputs: UnitInputs
    configuration: str
    targets: dict[str, float] = field(default_factory=dict)
    successes: int = 0
    attempts: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "inputs": asdict(self.inputs),
            "configuration": self.configuration,
            "targets": self.targets,
            "successes": self.successes,
            "attempts": self.attempts,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> Unit:
        return cls(
            inputs=UnitInputs(**value["inputs"]),
            configuration=value["configuration"],
            targets=dict(value["targets"]),
            successes=int(value["successes"]),
            attempts=int(value["attempts"]),
        )


def _snapshot_inputs(snapshot: dict[str, Any]) -> dict[str, Any]:
    task = snapshot["task"]
    network = snapshot["network"]
    return {
        "request_id": snapshot["request_id"],
        "cell_id": snapshot["cell_id"],
        "condition_id": snapshot["condition_id"],
        "task": snapshot["task_name"],
        "edge_load": snapshot["edge_load_level"],
        "cloud_load": snapshot["cloud_load_level"],
        "width": int(task["image_width"]),
        "height": int(task["image_height"]),
        "image_bytes": int(task["image_bytes"]),
        "uplink_bps": float(network["uplink_bps"]),
        "rtt_ms": float(network["rtt_ms"]),
    }


def _stage_targets(row: dict[str, Any]) -> dict[str, float]:
    fields = row["fields"]
    e2e = float(row["observed_end_to_end_ms"])
    edge = float(fields.get("edge_total_ms") or 0.0)
    exchange = float(fields.get("network_exchange_ms") or 0.0)
    server = float(fields.get("server_processing_ms") or 0.0)
    return {
        "e2e": e2e,
        "edge": edge,
        "server": server,
        "transport": exchange - server if exchange else 0.0,
        "residual": e2e - edge - exchange,
        "request_bytes": float(fields.get("request_wire_bytes") or 0.0),
    }


def _merge_inputs(snapshots: list[dict[str, Any]]) -> UnitInputs:
    first = snapshots[0]
    for other in snapshots[1:]:
        for key in ("request_id", "cell_id", "task", "edge_load", "cloud_load"):
            if other[key] != first[key]:
                raise ValueError(f"unit repetitions disagree on {key}")
    merged = dict(first)
    merged["uplink_bps"] = median(s["uplink_bps"] for s in snapshots)
    merged["rtt_ms"] = median(s["rtt_ms"] for s in snapshots)
    return UnitInputs(**merged)


def build_units(
    pairs: Iterable[tuple[dict[str, Any], dict[str, Any] | None]],
) -> list[Unit]:
    """Group (input snapshot, target row or None) pairs into request-cell units."""

    snapshots: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    stages: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    attempts: dict[tuple[str, str, str], int] = defaultdict(int)
    for snapshot, target in pairs:
        key = (
            snapshot["request_id"],
            snapshot["cell_id"],
            snapshot["configuration_id"],
        )
        snapshots[key].append(_snapshot_inputs(snapshot))
        attempts[key] += 1
        if target is not None and target.get("status") == "ok":
            stages[key].append(_stage_targets(target))
    units = []
    for key in sorted(snapshots):
        values = stages.get(key, [])
        unit = Unit(
            inputs=_merge_inputs(snapshots[key]),
            configuration=key[2],
            successes=len(values),
            attempts=attempts[key],
        )
        if values:
            unit.targets = {t: median(v[t] for v in values) for t in TARGETS}
        units.append(unit)
    return units


def read_training(path: Path) -> list[Unit]:
    def pairs() -> Iterator[tuple[dict[str, Any], dict[str, Any] | None]]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield row["inputs"], row["target"]

    return build_units(pairs())


def read_inputs(path: Path) -> list[Unit]:
    def pairs() -> Iterator[tuple[dict[str, Any], dict[str, Any] | None]]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line), None

    return build_units(pairs())


def scored_units(snapshots: Path, targets: list[bytes]) -> list[Unit]:
    """Validation units with outcomes: input snapshots and the sealed target
    lines, which the partitioner wrote in the same order."""

    with snapshots.open(encoding="utf-8") as handle:
        inputs = [json.loads(line) for line in handle]
    if len(inputs) != len(targets):
        raise ValueError("validation inputs and targets differ in length")
    rows = [json.loads(line) for line in targets]
    for snapshot, row in zip(inputs, rows, strict=True):
        if (
            snapshot["request_id"] != row["request_id"]
            or snapshot["trial_id"] != row["trial_id"]
            or snapshot["configuration_id"] != row["configuration_id"]
            or not row["campaign_id"].endswith(":" + snapshot["subblock_id"])
        ):
            raise ValueError("validation input and target rows are misaligned")
    return build_units(zip(inputs, rows, strict=True))
