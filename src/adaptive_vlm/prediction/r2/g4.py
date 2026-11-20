"""G4 on the G2 v2+ campaign: the frozen r2 selector against the measured oracle.

docs/predictor-v2/g4-specification.md fixes every rule here. The selector is
``scheduler.selector.select`` over ``ALL_CONFIGURATIONS`` (or the four
lossless ones), fed with frozen ``r2_component`` predictions; no measured
outcome enters a decision. The measured oracle is the configuration with the
lowest observed latency in the same request-cell, used only afterwards.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from adaptive_vlm.prediction.r2.models import Calibration, predict
from adaptive_vlm.prediction.r2.units import (
    CONFIGURATIONS,
    Unit,
    UnitInputs,
    build_units,
)
from adaptive_vlm.prediction.v1.calibration import percentile
from adaptive_vlm.prediction.v1.predictor import Prediction
from adaptive_vlm.scheduler.configuration import (
    ALL_CONFIGURATIONS,
    CapabilityStatus,
    ConfigurationSpec,
)
from adaptive_vlm.scheduler.selector import (
    CandidateSnapshot,
    SelectionDecision,
    predicted_oracle,
    select,
)

SEED = 1707
RESAMPLES = 2000
MEDIAN_RELATIVE_REGRET_LIMIT = 10.0
MIN_COMPLETE_BLOCKS = 10
LOSSLESS = tuple(s for s in ALL_CONFIGURATIONS if s.payload != "int8")
STATIC = ("edge/full", "cloud/raw_image", "distributed/projected/int8")


# --------------------------------------------------------------------------
# Evaluation units straight from profiles (the partitioner never opens E)


def profile_units(
    *, campaign: Path, profiles: Path, feature_dir: Path, split: str
) -> list[Unit]:
    """Request-cell units of one split, built like the r2 partitions."""

    features = {
        task: json.loads((feature_dir / f"task_features-{task}.json").read_text())[
            "tasks"
        ]
        for task in ("short", "long")
    }

    def pairs() -> Iterator[tuple[dict[str, Any], dict[str, Any] | None]]:
        for directory in sorted((campaign / "subblocks").iterdir()):
            plan = json.loads((directory / "plan.json").read_text())
            if plan["condition"]["split"] != split:
                continue
            path = profiles / plan["subblock_id"] / "profile" / "profile.jsonl"
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["warmup"]:
                    continue
                if (
                    row["edge_load_level"] != plan["load_level"]
                    or row["cloud_load_level"] != plan["cloud_load"]
                    or row["network_state"]["bucket_id"]
                    != plan["condition"]["condition_id"]
                ):
                    raise ValueError(f"profile row differs from {plan['subblock_id']}")
                snapshot = {
                    "request_id": row["request_id"],
                    "cell_id": plan["cell_id"],
                    "condition_id": plan["condition"]["condition_id"],
                    "configuration_id": row["configuration_id"],
                    "task_name": plan["task"],
                    "edge_load_level": row["edge_load_level"],
                    "cloud_load_level": row["cloud_load_level"],
                    "task": features[plan["task"]][row["request_id"]],
                    "network": {
                        "uplink_bps": row["network_state"]["uplink_bps"],
                        "rtt_ms": row["network_state"]["rtt_ms"],
                    },
                }
                yield snapshot, row

    return build_units(pairs())


# --------------------------------------------------------------------------
# Selection with the existing selector


def candidate_prediction(
    calibration: Calibration, inputs: UnitInputs, configuration: str
) -> Prediction:
    """An r2 prediction in the selector's Prediction shape."""

    result = predict(calibration, inputs, configuration)
    total = result["total"]
    return Prediction(
        configuration_id=configuration,
        valid=total is not None,
        reason=None if total is not None else "; ".join(result["reasons"]),
        coverage=result["label"],
        calibration_id=calibration["provenance"]["training_units_sha256"],
        estimated_output_tokens=None,
        output_estimate_source="per-task training medians (r2)",
        components_ms={
            k: float(result[k])
            for k in ("edge", "server", "network", "residual")
            if k in result
        },
        total_ms=total,
        lower_ms=None,
        upper_ms=None,
        request_payload_bytes=None,
        request_wire_bytes=None,
        response_wire_bytes=None,
        edge_memory_requirement_mib=None,
    )


def decide(
    calibration: Calibration,
    inputs: UnitInputs,
    configurations: tuple[ConfigurationSpec, ...] = ALL_CONFIGURATIONS,
) -> SelectionDecision:
    snapshots = tuple(
        CandidateSnapshot(
            configuration_id=spec.id,
            capability=CapabilityStatus(spec.id, True),
            prediction=candidate_prediction(calibration, inputs, spec.id.value),
        )
        for spec in configurations
    )
    return select(snapshots, configurations=configurations)


# --------------------------------------------------------------------------
# Best static configuration (validation cells only)


def best_static(units: Iterable[Unit]) -> dict[str, Any]:
    """Lowest median request-cell latency; configuration order breaks ties."""

    values: dict[str, list[float]] = defaultdict(list)
    for unit in units:
        if unit.successes:
            values[unit.configuration].append(unit.targets["e2e"])
    medians = {c: median(values[c]) for c in CONFIGURATIONS}
    choice = min(CONFIGURATIONS, key=lambda c: (medians[c], CONFIGURATIONS.index(c)))
    return {
        "rule": "lowest median request-cell latency over the validation cells; "
        "configuration order breaks ties",
        "median_ms": medians,
        "units": {c: len(values[c]) for c in CONFIGURATIONS},
        "configuration": choice,
    }


# --------------------------------------------------------------------------
# Evaluation against the measured oracle


def _bootstrap(
    groups: dict[str, list[float]], statistic: Callable[[list[float]], float]
) -> dict[str, float]:
    keys = sorted(groups)
    generator = random.Random(SEED)
    estimates = sorted(
        statistic([v for k in generator.choices(keys, k=len(keys)) for v in groups[k]])
        for _ in range(RESAMPLES)
    )
    return {
        "estimate": statistic([v for k in keys for v in groups[k]]),
        "ci_low": percentile(estimates, 0.025),
        "ci_high": percentile(estimates, 0.975),
    }


def _p50(values: list[float]) -> float:
    return percentile(values, 0.5)


def request_cells(units: Iterable[Unit]) -> dict[tuple[str, str], dict[str, Unit]]:
    cells: dict[tuple[str, str], dict[str, Unit]] = defaultdict(dict)
    for unit in units:
        cells[(unit.inputs.request_id, unit.inputs.cell_id)][unit.configuration] = unit
    return dict(cells)


def evaluate_decisions(
    calibration: Calibration,
    units: Sequence[Unit],
    *,
    best_static_configuration: str,
    holdout_conditions: Sequence[str],
    fitted_conditions: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells = request_cells(units)
    policies = {
        "always_edge": "edge/full",
        "always_cloud": "cloud/raw_image",
        "always_int8": "distributed/projected/int8",
        "best_static": best_static_configuration,
    }
    records = []
    for (request_id, cell_id), members in sorted(cells.items()):
        inputs = next(iter(members.values())).inputs
        observed = {
            c: (u.targets["e2e"] if u.successes == u.attempts and u.successes else None)
            for c, u in members.items()
        }
        complete = set(members) == set(CONFIGURATIONS) and all(
            v is not None for v in observed.values()
        )
        record: dict[str, Any] = {
            "request_id": request_id,
            "cell_id": cell_id,
            "condition_id": inputs.condition_id,
            "task": inputs.task,
            "load_state": f"{inputs.edge_load}/{inputs.cloud_load}",
            "complete": complete,
            "observed_ms": observed,
        }
        for name, configurations in (
            ("selector", ALL_CONFIGURATIONS),
            ("lossless", LOSSLESS),
        ):
            decision = decide(calibration, inputs, configurations)
            record[name] = {
                "selected": decision.selected_configuration_id,
                "status": decision.status,
                "predicted_oracle": predicted_oracle(decision),
                "predicted_ms": {
                    c.configuration_id: (
                        c.prediction.total_ms if c.prediction else None
                    )
                    for c in decision.candidates
                },
                "coverage": {
                    c.configuration_id: (
                        c.prediction.coverage if c.prediction else None
                    )
                    for c in decision.candidates
                },
            }
        if complete:
            for name, pool in (
                ("oracle", CONFIGURATIONS),
                ("oracle_lossless", tuple(s.id.value for s in LOSSLESS)),
            ):
                best = min(pool, key=lambda c: (observed[c], pool.index(c)))
                record[name] = {"configuration": best, "latency_ms": observed[best]}
            chosen = {
                "selector": record["selector"]["selected"],
                "lossless": record["lossless"]["selected"],
                **policies,
            }
            record["regret"] = {}
            for name, configuration in chosen.items():
                oracle = record["oracle_lossless" if name == "lossless" else "oracle"]
                latency = observed.get(configuration) if configuration else None
                record["regret"][name] = (
                    None
                    if latency is None
                    else {
                        "configuration": configuration,
                        "latency_ms": latency,
                        "regret_ms": latency - oracle["latency_ms"],
                        "relative_regret_percent": 100.0
                        * (latency - oracle["latency_ms"])
                        / oracle["latency_ms"],
                        "agree": configuration == oracle["configuration"],
                    }
                )
        records.append(record)
    return records, summarise(
        records,
        holdout_conditions=holdout_conditions,
        fitted_conditions=fitted_conditions,
    )


def _policy_summary(complete: list[dict[str, Any]], name: str) -> dict[str, Any]:
    rows = [r for r in complete if r["regret"][name] is not None]
    relative = [r["regret"][name]["relative_regret_percent"] for r in rows]
    absolute = [r["regret"][name]["regret_ms"] for r in rows]
    by_request: dict[str, list[float]] = defaultdict(list)
    rel_by_request: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_request[r["request_id"]].append(r["regret"][name]["regret_ms"])
        rel_by_request[r["request_id"]].append(
            r["regret"][name]["relative_regret_percent"]
        )
    return {
        "request_cells": len(rows),
        "top1_agreement": sum(r["regret"][name]["agree"] for r in rows),
        "top1_agreement_percent": 100.0
        * sum(r["regret"][name]["agree"] for r in rows)
        / len(rows),
        "median_relative_regret_percent": _p50(relative),
        "p90_relative_regret_percent": percentile(relative, 0.9),
        "mean_relative_regret_percent": mean(relative),
        "mean_regret_ms": mean(absolute),
        "median_regret_ms": _p50(absolute),
        "max_regret_ms": max(absolute),
        "mean_latency_ms": mean(r["regret"][name]["latency_ms"] for r in rows),
        "mean_regret_ms_ci": _bootstrap(by_request, mean),
        "mean_relative_regret_percent_ci": _bootstrap(rel_by_request, mean),
        "choices": dict(
            sorted(_counts(r["regret"][name]["configuration"] for r in rows).items())
        ),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    return counts


def summarise(
    records: list[dict[str, Any]],
    *,
    holdout_conditions: Sequence[str],
    fitted_conditions: Sequence[str],
) -> dict[str, Any]:
    complete = [r for r in records if r["complete"]]
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        cells[r["cell_id"]].append(r)
    complete_blocks = sorted(
        c for c, rs in cells.items() if all(r["complete"] for r in rs)
    )
    policies = (
        "selector",
        "lossless",
        "always_edge",
        "always_cloud",
        "always_int8",
        "best_static",
    )
    summary: dict[str, Any] = {
        "request_cells": len(records),
        "complete_request_cells": len(complete),
        "blocks": len(cells),
        "complete_blocks": len(complete_blocks),
        "conditions": sorted({r["condition_id"] for r in records}),
        "oracle_choices": dict(
            sorted(_counts(r["oracle"]["configuration"] for r in complete).items())
        ),
        "policies": {p: _policy_summary(complete, p) for p in policies},
    }
    # Paired differences: selector latency minus each static policy's latency.
    summary["selector_minus_policy_ms"] = {}
    for policy in policies[2:]:
        groups: dict[str, list[float]] = defaultdict(list)
        for r in complete:
            groups[r["request_id"]].append(
                r["regret"]["selector"]["latency_ms"]
                - r["regret"][policy]["latency_ms"]
            )
        summary["selector_minus_policy_ms"][policy] = _bootstrap(groups, mean)
    summary["by_load_state"] = {
        state: _policy_summary(
            [r for r in complete if r["load_state"] == state], "selector"
        )
        for state in sorted({r["load_state"] for r in complete})
    }
    summary["by_condition"] = {
        condition: _policy_summary(
            [r for r in complete if r["condition_id"] == condition], "selector"
        )
        for condition in summary["conditions"]
    }
    summary["by_cell"] = {
        cell: {
            "oracle_choices": dict(
                sorted(
                    _counts(
                        r["oracle"]["configuration"] for r in rs if r["complete"]
                    ).items()
                )
            ),
            "selector_choices": dict(
                sorted(_counts(r["selector"]["selected"] for r in rs).items())
            ),
            "mean_regret_ms": mean(
                r["regret"]["selector"]["regret_ms"] for r in rs if r["complete"]
            ),
        }
        for cell, rs in sorted(cells.items())
    }
    no_feasible = sum(r["selector"]["selected"] is None for r in records)
    unsafe = sum(
        1
        for r in records
        if r["selector"]["selected"] is not None
        and r["observed_ms"].get(r["selector"]["selected"]) is None
    )
    consistent = all(
        r[n]["selected"] == r[n]["predicted_oracle"]
        for r in records
        for n in ("selector", "lossless")
    )
    holdout = (
        bool(summary["conditions"])
        and set(summary["conditions"]) <= set(holdout_conditions)
        and not set(summary["conditions"]) & set(fitted_conditions)
    )
    median_regret = summary["policies"]["selector"]["median_relative_regret_percent"]
    summary["safety"] = {
        "no_feasible_decisions": no_feasible,
        "unsafe_selections": unsafe,
        "predicted_oracle_consistent": consistent,
        "extrapolated_selected_predictions": sum(
            r["selector"]["coverage"][r["selector"]["selected"]] != "interpolation"
            for r in records
            if r["selector"]["selected"]
        ),
    }
    summary["gate"] = {
        "complete_blocks_ok": len(complete_blocks) >= MIN_COMPLETE_BLOCKS,
        "network_holdout": holdout,
        "median_relative_regret_ok": median_regret <= MEDIAN_RELATIVE_REGRET_LIMIT,
        "safety_ok": no_feasible == 0 and unsafe == 0,
        "predicted_oracle_consistent": consistent,
    }
    summary["gate"]["pass"] = all(summary["gate"].values())
    return summary
