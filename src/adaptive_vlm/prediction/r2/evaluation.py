"""Metrics, ranking, grouped cross-validation and the G3 gate for r2
(specification sections 6-8)."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from statistics import mean, median
from typing import Any

from adaptive_vlm.prediction.r2.models import (
    FITTERS,
    MODELS,
    UNSUPPORTED,
    Calibration,
    predict,
)
from adaptive_vlm.prediction.r2.units import CONFIGURATIONS, Unit
from adaptive_vlm.prediction.v1.calibration import percentile

SEED = 1707
RESAMPLES = 2000
FOLDS = 5
LOSSLESS = tuple(c for c in CONFIGURATIONS if not c.endswith("int8"))
MEDIAN_APE_LIMIT = 15.0
P90_APE_LIMIT = 30.0

Record = dict[str, Any]


def predict_units(
    calibration: Calibration, units: Iterable[Unit], model: str | None = None
) -> list[Record]:
    """One record per unit: inputs key, configuration, prediction, observed."""

    name = model or calibration["model"]
    records = []
    for unit in units:
        prediction = predict(calibration, unit.inputs, unit.configuration)
        records.append(
            {
                "model": name,
                "request_id": unit.inputs.request_id,
                "cell_id": unit.inputs.cell_id,
                "condition_id": unit.inputs.condition_id,
                "task": unit.inputs.task,
                "load_state": f"{unit.inputs.edge_load}/{unit.inputs.cloud_load}",
                "configuration": unit.configuration,
                "prediction": prediction,
                "observed": unit.targets.get("e2e") if unit.successes else None,
                "successes": unit.successes,
                "attempts": unit.attempts,
            }
        )
    return records


def error_metrics(
    records: Sequence[Record], interval: float | None = None
) -> dict[str, Any]:
    """Errors over valid predictions with an observed value."""

    pairs = [
        (r["prediction"]["total"], r["observed"])
        for r in records
        if r["prediction"]["total"] is not None and r["observed"] is not None
    ]
    units = len(records)
    valid = sum(r["prediction"]["total"] is not None for r in records)
    labels: dict[str, int] = defaultdict(int)
    for r in records:
        labels[r["prediction"]["label"]] += 1
    result: dict[str, Any] = {
        "units": units,
        "valid": valid,
        "coverage_percent": 100.0 * valid / units if units else 0.0,
        "labels": dict(sorted(labels.items())),
        "scored": len(pairs),
    }
    if not pairs:
        return result
    ape = [100.0 * abs(p - o) / o for p, o in pairs]
    result.update(
        median_ape=percentile(ape, 0.5),
        p90_ape=percentile(ape, 0.9),
        mae_ms=mean(abs(p - o) for p, o in pairs),
        bias_ms=mean(p - o for p, o in pairs),
    )
    if interval is not None:
        result["interval_q"] = interval
        result["interval_coverage_percent"] = (
            100.0
            * sum(abs(p - o) <= interval * p / 100.0 for p, o in pairs)
            / len(pairs)
        )
    return result


def _choose(values: dict[str, float | None]) -> str | None:
    best: tuple[float, str] | None = None
    for configuration in CONFIGURATIONS:  # fixed order breaks exact ties
        value = values.get(configuration)
        if value is not None and (best is None or value < best[0]):
            best = (value, configuration)
    return None if best is None else best[1]


def ranking_units(
    records: Sequence[Record], configurations: Sequence[str] = CONFIGURATIONS
) -> list[Record]:
    """Per request-cell: the predicted and the observed fastest configuration."""

    grouped: dict[tuple[str, str], dict[str, Record]] = defaultdict(dict)
    for r in records:
        if r["configuration"] in configurations:
            grouped[(r["request_id"], r["cell_id"])][r["configuration"]] = r
    out = []
    for (request_id, cell_id), members in sorted(grouped.items()):
        predicted = _choose({c: m["prediction"]["total"] for c, m in members.items()})
        observed = _choose({c: m["observed"] for c, m in members.items()})
        regret = None
        if predicted is not None and observed is not None:
            chosen = members[predicted]["observed"]
            if chosen is not None:
                regret = chosen - members[observed]["observed"]
        out.append(
            {
                "request_id": request_id,
                "cell_id": cell_id,
                "predicted": predicted,
                "observed": observed,
                "agree": predicted is not None and predicted == observed,
                "regret_ms": regret,
            }
        )
    return out


def ranking_summary(units: Sequence[Record]) -> dict[str, Any]:
    regrets = [u["regret_ms"] for u in units if u["regret_ms"] is not None]
    return {
        "units": len(units),
        "agreement": sum(u["agree"] for u in units),
        "agreement_percent": 100.0 * sum(u["agree"] for u in units) / len(units)
        if units
        else 0.0,
        "mean_regret_ms": mean(regrets) if regrets else None,
        "median_regret_ms": median(regrets) if regrets else None,
        "max_regret_ms": max(regrets) if regrets else None,
        "predicted_choices": dict(
            sorted(_count(u["predicted"] for u in units).items())
        ),
        "observed_fastest": dict(sorted(_count(u["observed"] for u in units).items())),
    }


def _count(values: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        counts[str(v)] += 1
    return counts


def cluster_bootstrap(
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


def by_configuration(
    records: Sequence[Record], intervals: dict[str, float] | None = None
) -> dict[str, Any]:
    return {
        c: error_metrics(
            [r for r in records if r["configuration"] == c],
            None if intervals is None else intervals.get(c),
        )
        for c in CONFIGURATIONS
    }


def breakdown(records: Sequence[Record], field: str) -> dict[str, Any]:
    keys = sorted({r[field] for r in records})
    return {k: by_configuration([r for r in records if r[field] == k]) for k in keys}


# --------------------------------------------------------------------------
# Training review (section 6)


def grouped_folds(request_ids: Iterable[str], folds: int = FOLDS) -> dict[str, int]:
    ordered = sorted(set(request_ids))
    random.Random(SEED).shuffle(ordered)
    return {request_id: i % folds for i, request_id in enumerate(ordered)}


def out_of_fold(
    units: Sequence[Unit], fold_of: Callable[[Unit], Any]
) -> dict[str, list[Record]]:
    folds = sorted({fold_of(u) for u in units})
    records: dict[str, list[Record]] = {m: [] for m in MODELS}
    for fold in folds:
        train = [u for u in units if fold_of(u) != fold]
        held = [u for u in units if fold_of(u) == fold]
        for model in MODELS:
            records[model] += predict_units(FITTERS[model](train), held, model)
    return records


def training_review(units: Sequence[Unit], component: Calibration) -> dict[str, Any]:
    folds = grouped_folds(u.inputs.request_id for u in units)
    cv = out_of_fold(units, lambda u: folds[u.inputs.request_id])
    loco = out_of_fold(units, lambda u: u.inputs.condition_id)
    intervals = {
        model: {
            c: percentile(
                [
                    100.0
                    * abs(r["prediction"]["total"] - r["observed"])
                    / r["observed"]
                    for r in cv[model]
                    if r["configuration"] == c
                    and r["prediction"]["total"] is not None
                    and r["observed"] is not None
                ],
                0.95,
            )
            for c in CONFIGURATIONS
        }
        for model in MODELS
    }
    review: dict[str, Any] = {
        "folds": FOLDS,
        "seed": SEED,
        "grouped_cv": {
            m: {
                "by_configuration": by_configuration(cv[m], intervals[m]),
                "ranking": ranking_summary(ranking_units(cv[m])),
            }
            for m in MODELS
        },
        "leave_one_condition_out": {
            m: {
                "by_configuration": by_configuration(loco[m]),
                "ranking": ranking_summary(ranking_units(loco[m])),
            }
            for m in MODELS
        },
        "intervals_q95_ape": intervals,
        "component_signs": {
            "network_scales": component["network"],
            "clamped": component["clamped"],
        },
    }
    worst = max(
        review["grouped_cv"]["r2_component"]["by_configuration"][c]["p90_ape"]
        for c in CONFIGURATIONS
    )
    review["candidate"] = "r2_additive" if worst > P90_APE_LIMIT else "r2_component"
    review["candidate_rule"] = (
        "r2_component unless its grouped-CV P90 APE exceeds 30% for a "
        "configuration (specification section 6)"
    )
    return review


# --------------------------------------------------------------------------
# Validation scoring and the G3 gate (sections 7-8)


def score_model(
    records: Sequence[Record], constant: Sequence[Record], intervals: dict[str, float]
) -> dict[str, Any]:
    configurations = by_configuration(records, intervals)
    ranking = ranking_units(records)
    baseline = ranking_units(constant)
    agreement = ranking_summary(ranking)
    baseline_agreement = ranking_summary(baseline)
    unsafe = sum(
        1
        for r in records
        if r["prediction"]["label"] != UNSUPPORTED and r["successes"] < r["attempts"]
    )
    by_request: dict[str, list[float]] = defaultdict(list)
    base = {(u["request_id"], u["cell_id"]): u for u in baseline}
    for u in ranking:
        other = base[(u["request_id"], u["cell_id"])]
        by_request[u["request_id"]].append(float(u["agree"]) - float(other["agree"]))
    regrets: dict[str, list[float]] = defaultdict(list)
    for u in ranking:
        if u["regret_ms"] is not None:
            regrets[u["request_id"]].append(u["regret_ms"])
    numeric = {
        c: (
            m.get("median_ape") is not None
            and m["median_ape"] <= MEDIAN_APE_LIMIT
            and m["p90_ape"] <= P90_APE_LIMIT
        )
        for c, m in configurations.items()
    }
    coverage = all(m["valid"] == m["units"] for m in configurations.values())
    ranks = agreement["agreement"] >= baseline_agreement["agreement"]
    return {
        "by_configuration": configurations,
        "ranking": agreement,
        "ranking_lossless": ranking_summary(ranking_units(records, LOSSLESS)),
        "agreement_minus_constant": cluster_bootstrap(by_request, lambda v: sum(v)),
        "mean_regret_ms": cluster_bootstrap(regrets, mean) if regrets else None,
        "unsafe_feasibility_choices": unsafe,
        "gate": {
            "numeric": numeric,
            "numeric_pass": all(numeric.values()),
            "coverage_pass": coverage,
            "ranking_pass": ranks,
            "safety_pass": unsafe == 0,
            "pass": all(numeric.values()) and coverage and ranks and unsafe == 0,
        },
    }


def decide(candidate: str, scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Section 8: the candidate if it passes, else the additive fallback."""

    order = [candidate] + (["r2_additive"] if candidate != "r2_additive" else [])
    selected = next((m for m in order if scores[m]["gate"]["pass"]), None)
    return {
        "candidate": candidate,
        "considered": order,
        "selected": selected,
        "g3": "passed" if selected else "failed",
        "rule": (
            "select the candidate if it passes; otherwise r2_additive if it "
            "passes; otherwise G3 fails (constant_median is never selected)"
        ),
    }
