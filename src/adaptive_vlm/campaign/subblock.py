"""One G2 v2 sub-block: verify shaping, probe, run, re-probe, validate.

The validity gates are the protocol's section 8:

1. pre and post live measurements pass and neither host's netem queue
   dropped a packet during the sub-block;
2. (campaign preflight) the heterogeneous-correctness reference is verified;
3. every planned attempt has a record, at most ``max_failure_rate`` failed;
4. at L0 the edge never spills: Windows dedicated usage on the CUDA adapter
   stays ``spill_margin_mib`` below its maximum at every request snapshot;
5. at L1/L2 the load generator was alive at the first and last measured
   request snapshots.

An invalid sub-block is reported, not silently accepted; the supervisor moves
it aside and reruns it as a whole.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Protocol

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.edge_load import NO_LOAD
from adaptive_vlm.benchmark.interleaved import PLACEMENTS, RESTARTABLE_EXIT_CODES
from adaptive_vlm.campaign.grid import Grid, SubBlock
from adaptive_vlm.campaign.network import QdiscState, measurement_failures


class Network(Protocol):
    controlled: bool

    def verify(self, *, rate_mbps: float, delay_ms: float) -> dict[str, QdiscState]: ...

    def measure(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Gates:
    max_failure_rate: float = 0.01
    edge_dedicated_limit_mib: float = 3960.0
    spill_margin_mib: float = 128.0
    require_spill_evidence: bool = True


class SubBlockAborted(RuntimeError):
    """The benchmark child failed in a way a restart cannot fix."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_bytes().splitlines(keepends=True)
        if line.endswith(b"\n")
    ]


def benchmark_arguments(
    grid: Grid, block: SubBlock, run_dir: Path, *, resume: bool
) -> list[str]:
    """Sub-block-specific ``interleaved-benchmark`` arguments."""

    arguments = [
        "--run-dir",
        str(run_dir),
        "--split-name",
        block.condition.split,
        "--order-seed",
        str(grid.subblock_seed(block)),
        "--repetition-indices",
        str(block.repetition),
        "--warmups",
        str(grid.warmups),
        "--edge-load-level",
        block.load_level,
    ]
    return [*arguments, "--resume"] if resume else arguments


def _drops(states: Mapping[str, QdiscState]) -> dict[str, int | None]:
    return {name: state.dropped for name, state in states.items()}


def run_subblock(
    *,
    grid: Grid,
    block: SubBlock,
    output: Path,
    network: Network,
    baseline_rtt_ms: float | None,
    command_prefix: Sequence[str],
    run_child: Callable[[list[str], Path], int],
    gates: Gates,
    max_restarts: int,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run (or resume) one sub-block and return its validation report."""

    output.mkdir(parents=True, exist_ok=True)
    condition = block.condition
    shaped = {
        "rate_mbps": condition.rate_mbps,
        "delay_ms": condition.egress_delay_ms,
    }
    plan = {
        "schema_version": 1,
        "subblock_id": block.subblock_id,
        "cell_id": block.cell_id,
        "condition": asdict(condition),
        "load_level": block.load_level,
        "task": block.task,
        "repetition": block.repetition,
        "order_seed": grid.subblock_seed(block),
        "grid_sha256": grid.sha256,
        "controlled_network": network.controlled,
        "egress_delay_ms": condition.egress_delay_ms,
    }
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError(f"sub-block plan changed: {block.subblock_id}")
    write_json_atomic(plan_path, plan)

    before = network.verify(**shaped)
    write_json_atomic(
        output / "qdisc_before.json",
        {"time": _now(), "states": {k: asdict(v) for k, v in before.items()}},
    )
    reasons: list[str] = []
    pre_path = output / "network_pre.json"
    if not pre_path.exists():
        write_json_atomic(pre_path, network.measure())
    pre = json.loads(pre_path.read_text())

    run_dir = output / "run"
    restarts_path = output / "restarts.jsonl"
    restarts = sum(1 for _ in _rows(restarts_path))
    while True:
        command = [
            *command_prefix,
            *benchmark_arguments(grid, block, run_dir, resume=run_dir.exists()),
        ]
        log(f"{block.subblock_id}: {'resuming' if run_dir.exists() else 'starting'}")
        code = run_child(command, run_dir)
        if code == 0:
            break
        if code in RESTARTABLE_EXIT_CODES and restarts < max_restarts:
            restarts += 1
            with restarts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": _now(), "exit_code": code}) + "\n")
            log(f"{block.subblock_id}: exit {code}; restart {restarts}/{max_restarts}")
            continue
        raise SubBlockAborted(f"{block.subblock_id}: benchmark exited {code}")

    after = network.verify(**shaped)
    write_json_atomic(
        output / "qdisc_after.json",
        {"time": _now(), "states": {k: asdict(v) for k, v in after.items()}},
    )
    post_path = output / "network_post.json"
    if not post_path.exists():
        write_json_atomic(post_path, network.measure())
    post = json.loads(post_path.read_text())

    if network.controlled:
        if baseline_rtt_ms is None:
            raise RuntimeError("a controlled network needs the unshaped baseline RTT")
        for label, measurement in (("pre", pre), ("post", post)):
            reasons += [
                f"{label}: {failure}"
                for failure in measurement_failures(
                    measurement,
                    rate_mbps=condition.rate_mbps,
                    added_rtt_ms=condition.added_rtt_ms,
                    baseline_rtt_ms=baseline_rtt_ms,
                )
            ]
        drops_before, drops_after = _drops(before), _drops(after)
        for host, dropped in drops_after.items():
            start = drops_before.get(host)
            if dropped is None or start is None or dropped - start != 0:
                reasons.append(f"{host} netem dropped packets ({start} -> {dropped})")

    placements = {}
    for placement in PLACEMENTS:
        directory = run_dir / placement
        planned = json.loads((directory / "plan.json").read_text())["schedule"]
        rows = _rows(directory / "requests.jsonl")
        measured = [row for row in rows if not row["warmup"]]
        expected = sum(1 for trial in planned if not trial["warmup"])
        ok = [row for row in measured if row["status"] == "ok"]
        if len(rows) != len(planned):
            reasons.append(f"{placement}: {len(rows)}/{len(planned)} attempts recorded")
        failure_rate = (len(measured) - len(ok)) / expected if expected else 0.0
        if failure_rate > gates.max_failure_rate:
            reasons.append(f"{placement}: failure rate {failure_rate:.2%}")
        placements[placement] = {
            "attempts": len(rows),
            "measured": len(measured),
            "ok": len(ok),
            "failure_rate": failure_rate,
            "median_ms": (
                median(row["observed_end_to_end_ms"] for row in ok) if ok else None
            ),
            "requests_sha256": _sha256(directory / "requests.jsonl"),
            "plan_sha256": _sha256(directory / "plan.json"),
        }

    snapshots = [
        row.get("resource_state") or {}
        for row in sorted(
            (
                row
                for placement in PLACEMENTS
                for row in _rows(run_dir / placement / "requests.jsonl")
                if not row["warmup"]
            ),
            key=lambda row: row["interleave_position"],
        )
    ]
    if block.load_level == NO_LOAD:
        limit = gates.edge_dedicated_limit_mib - gates.spill_margin_mib
        dedicated = [s.get("edge_windows_gpu_dedicated_mib") for s in snapshots]
        if any(value is None for value in dedicated):
            if gates.require_spill_evidence:
                reasons.append("edge spill not verifiable: missing dedicated usage")
        elif (
            dedicated
            and (peak := max(float(v) for v in dedicated if v is not None)) >= limit
        ):
            reasons.append(
                f"edge spill: dedicated usage reached {peak:.0f} MiB "
                f"(limit {limit:.0f})"
            )
    elif snapshots:
        for label, state in (("first", snapshots[0]), ("last", snapshots[-1])):
            if state.get("edge_load_alive") is not True:
                reasons.append(f"load generator not alive at the {label} request")

    report = {
        "schema_version": 1,
        "subblock_id": block.subblock_id,
        "valid": not reasons,
        "reasons": reasons,
        "restarts": restarts,
        "completed_at_utc": _now(),
        "placements": placements,
        "artifact_sha256": {
            name: _sha256(output / name)
            for name in (
                "plan.json",
                "network_pre.json",
                "network_post.json",
                "qdisc_before.json",
                "qdisc_after.json",
                "run/interleave.json",
            )
        },
    }
    write_json_atomic(output / "validation_report.json", report)
    return report


def check_completed(output: Path) -> dict[str, Any] | None:
    """A finished sub-block's report, if its evidence is still unchanged."""

    path = output / "validation_report.json"
    if not path.exists():
        return None
    report: dict[str, Any] = json.loads(path.read_text())
    for name, expected in report["artifact_sha256"].items():
        if _sha256(output / name) != expected:
            raise RuntimeError(f"changed completed sub-block artifact: {output / name}")
    for placement, values in report["placements"].items():
        journal = output / "run" / placement / "requests.jsonl"
        if _sha256(journal) != values["requests_sha256"]:
            raise RuntimeError(f"changed completed sub-block journal: {journal}")
    return report
