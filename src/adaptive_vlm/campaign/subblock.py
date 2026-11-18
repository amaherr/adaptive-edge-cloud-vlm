"""One G2 v2 sub-block: verify shaping, probe, run, re-probe, validate.

The validity gates are the protocol's section 8:

1. pre and post live measurements pass and neither host's netem queue
   dropped a packet during the sub-block;
2. (campaign preflight) the heterogeneous-correctness reference is verified;
3. every planned attempt has a record, at most ``max_failure_rate`` failed;
4. at L0 the edge never spills: Windows dedicated usage on the CUDA adapter
   stays ``spill_margin_mib`` below its maximum at every request snapshot;
5. at L1/L2 the load generator was alive at the first and last measured
   request snapshots;
6. (G2 v2+, amendment A2a rule 7) in a C1 sub-block every successful cloud
   and distributed response reports the cloud-load worker alive with a
   heartbeat younger than 5 s; in a C0 sub-block of a grid with cloud load,
   no response reports it alive.

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
from adaptive_vlm.benchmark.cloud_load import NO_CLOUD_LOAD
from adaptive_vlm.benchmark.edge_load import NO_LOAD, LoadLevel
from adaptive_vlm.benchmark.interleaved import PLACEMENTS, RESTARTABLE_EXIT_CODES
from adaptive_vlm.campaign.cloud_load import CloudLoad
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
    # Which snapshot field shows edge GPU memory pressure at L0. On Windows/WSL
    # a spill shows as WDDM dedicated usage reaching the adapter maximum; on
    # Linux there is no spill, only NVML used memory approaching the usable
    # limit, followed by CUDA OOM (counted as a failure either way).
    spill_metric: str = "edge_windows_gpu_dedicated_mib"
    cloud_load_max_age_s: float = 5.0


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
    if grid.placements != PLACEMENTS:
        arguments += ["--placements", *grid.placements]
    if grid.cloud_load_levels:
        arguments += ["--cloud-load-level", block.cloud_load]
    return [*arguments, "--resume"] if resume else arguments


def cloud_load_reasons(
    rows: Sequence[Mapping[str, Any]], block: SubBlock, *, max_age_s: float
) -> list[str]:
    """Gate A2a rule 7 over the successful remote responses of a sub-block."""

    remote = [
        row.get("metrics") or {}
        for row in rows
        if row.get("status") == "ok" and row.get("mode") != "edge"
    ]
    if block.cloud_load == NO_CLOUD_LOAD:
        alive = sum(m.get("server_state_load_alive") is True for m in remote)
        return [f"cloud load alive in {alive} C0 responses"] if alive else []
    if not remote:
        return ["no successful remote response to show the cloud load"]
    unreported = sum("server_state_load_alive" not in m for m in remote)
    if unreported:
        return [f"cloud load state missing from {unreported} responses"]
    reasons = []
    dead = sum(
        m.get("server_state_load_alive") is not True
        or m.get("server_state_load_level") != block.cloud_load
        for m in remote
    )
    if dead:
        reasons.append(f"cloud load {block.cloud_load} not alive in {dead} responses")
    ages = [m.get("server_state_load_heartbeat_age_s") for m in remote]
    stale = sum(age is None or float(age) >= max_age_s for age in ages)
    if stale:
        reasons.append(f"cloud load heartbeat {max_age_s:.0f}s+ old in {stale}")
    return reasons


def _append(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": _now(), **value}, sort_keys=True) + "\n")


def _drops(states: Mapping[str, QdiscState]) -> dict[str, int | None]:
    return {name: state.dropped for name, state in states.items()}


def _run_benchmark(
    *,
    grid: Grid,
    block: SubBlock,
    run_dir: Path,
    restarts_path: Path,
    restarts: int,
    command_prefix: Sequence[str],
    run_child: Callable[[list[str], Path], int],
    max_restarts: int,
    log: Callable[[str], None],
) -> None:
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
    cloud_load: CloudLoad | None = None,
    cloud_level: LoadLevel | None = None,
) -> dict[str, Any]:
    """Run (or resume) one sub-block and return its validation report.

    With ``cloud_load``, any leftover cloud worker is stopped first; a C1
    sub-block starts ``cloud_level`` before the benchmark and stops it after,
    whatever happens in between.
    """

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
        "cloud_load": block.cloud_load,
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
    cloud_log = output / "cloud_load.jsonl"
    loaded = block.cloud_load != NO_CLOUD_LOAD
    if loaded and (cloud_load is None or cloud_level is None):
        raise RuntimeError(f"{block.subblock_id} needs a cloud-load controller")
    if cloud_load is not None:
        reaped = cloud_load.reap()
        if reaped:
            log(f"{block.subblock_id}: stopped leftover cloud load {reaped}")
            _append(cloud_log, {"event": "reaped", "pids": reaped})
    if loaded:
        assert cloud_load is not None and cloud_level is not None
        started = cloud_load.start(cloud_level)
        _append(cloud_log, {"event": "started", **started})
        log(f"{block.subblock_id}: cloud load {cloud_level.name} running")
    try:
        _run_benchmark(
            grid=grid,
            block=block,
            run_dir=run_dir,
            restarts_path=restarts_path,
            restarts=restarts,
            command_prefix=command_prefix,
            run_child=run_child,
            max_restarts=max_restarts,
            log=log,
        )
    finally:
        if loaded:
            assert cloud_load is not None
            stopped = cloud_load.stop()
            _append(cloud_log, {"event": "stopped", **stopped})
    restarts = sum(1 for _ in _rows(restarts_path))

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
    for placement in grid.placements:
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
                for placement in grid.placements
                for row in _rows(run_dir / placement / "requests.jsonl")
                if not row["warmup"]
            ),
            key=lambda row: row["interleave_position"],
        )
    ]
    if block.load_level == NO_LOAD:
        limit = gates.edge_dedicated_limit_mib - gates.spill_margin_mib
        dedicated = [s.get(gates.spill_metric) for s in snapshots]
        if any(value is None for value in dedicated):
            if gates.require_spill_evidence:
                reasons.append(
                    f"edge spill not verifiable: missing {gates.spill_metric}"
                )
        elif (
            dedicated
            and (peak := max(float(v) for v in dedicated if v is not None)) >= limit
        ):
            reasons.append(
                f"edge spill: {gates.spill_metric} reached {peak:.0f} MiB "
                f"(limit {limit:.0f})"
            )
        oom = sum(
            row.get("status") == "oom"
            for placement in grid.placements
            for row in _rows(run_dir / placement / "requests.jsonl")
        )
        if oom:
            reasons.append(f"edge out of GPU memory in {oom} attempts at L0")
    elif snapshots:
        for label, state in (("first", snapshots[0]), ("last", snapshots[-1])):
            if state.get("edge_load_alive") is not True:
                reasons.append(f"load generator not alive at the {label} request")
    if grid.cloud_load_levels:
        reasons += cloud_load_reasons(
            [
                row
                for placement in grid.placements
                for row in _rows(run_dir / placement / "requests.jsonl")
                if not row["warmup"]
            ],
            block,
            max_age_s=gates.cloud_load_max_age_s,
        )

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
