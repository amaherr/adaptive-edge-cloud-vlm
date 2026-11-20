"""The three r2 models (specification section 4): fit and predict.

Calibrations are plain JSON-serialisable dicts. ``predict`` returns a dict
with the total, its components (component model) and a coverage label.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from statistics import mean, median
from typing import Any

from adaptive_vlm.prediction.r2.features import network_phi_ms
from adaptive_vlm.prediction.r2.units import (
    CLOUD,
    CONFIGURATIONS,
    EDGE,
    REMOTE,
    SPLITS,
    Unit,
    UnitInputs,
)
from adaptive_vlm.prediction.v2.calibration.regression import _solve

SCHEMA = "predictor-v2-r2"
INITIAL_WINDOWS_KIB = (16, 32, 48, 64, 96, 128, 192, 256)
MODELS = ("r2_component", "r2_additive", "constant_median")
INTERPOLATION = "interpolation"
EXTRAPOLATION = "physics_extrapolation"
UNSUPPORTED = "unsupported"

Calibration = dict[str, Any]
Prediction = dict[str, Any]


def _key(*parts: str) -> str:
    return "|".join(parts)


def _fitted(units: Iterable[Unit]) -> list[Unit]:
    return [u for u in units if u.successes > 0 and u.targets]


def fit_line(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, bool]:
    """y = a + b x by least squares; b < 0 becomes b = 0, a = median(y).

    Returns (a, b, clamped). With a single distinct x the fit is the median.
    """

    if len(set(xs)) < 2:
        return median(ys), 0.0, False
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    if b < 0:
        return median(ys), 0.0, True
    return my - b * mx, b, False


def _fit_network_line(
    xs: Sequence[float], ys: Sequence[float]
) -> tuple[float, float, bool]:
    """Network scale: b < 0 becomes b = 0 and a = mean(y) (section 4.1)."""

    a, b, clamped = fit_line(xs, ys)
    if clamped:
        return mean(ys), 0.0, True
    return a, b, False


def least_squares(rows: Sequence[Sequence[float]], ys: Sequence[float]) -> list[float]:
    width = len(rows[0])
    matrix = [
        [sum(r[i] * r[j] for r in rows) for j in range(width)] for i in range(width)
    ]
    vector = [
        sum(r[i] * y for r, y in zip(rows, ys, strict=True)) for i in range(width)
    ]
    solved = _solve(matrix, vector)
    if solved is None:
        raise ValueError("singular least-squares system")
    return solved


def _support(units: list[Unit]) -> dict[str, Any]:
    remote = [u.inputs for u in units if u.configuration in REMOTE]
    crops = [u.inputs.crops for u in units]
    return {
        "uplink_bps": [
            min(i.uplink_bps for i in remote),
            max(i.uplink_bps for i in remote),
        ],
        "rtt_ms": [min(i.rtt_ms for i in remote), max(i.rtt_ms for i in remote)],
        "crops": [min(crops), max(crops)],
    }


def _label(
    support: dict[str, Any], inputs: UnitInputs, configuration: str
) -> tuple[str, list[str]]:
    reasons = []
    low, high = support["crops"]
    if not low <= inputs.crops <= high:
        reasons.append(f"crops {inputs.crops} outside [{low}, {high}]")
    if configuration in REMOTE:
        for name in ("uplink_bps", "rtt_ms"):
            value = getattr(inputs, name)
            low, high = support[name]
            if not low <= value <= high:
                reasons.append(f"{name} {value:.6g} outside [{low:.6g}, {high:.6g}]")
    return (EXTRAPOLATION if reasons else INTERPOLATION), reasons


def _finite_inputs(inputs: UnitInputs) -> bool:
    return (
        all(math.isfinite(v) for v in (inputs.uplink_bps, inputs.rtt_ms))
        and inputs.uplink_bps > 0
        and inputs.rtt_ms >= 0
        and inputs.image_bytes > 0
        and inputs.width > 0
        and inputs.height > 0
    )


def _invalid(reason: str) -> Prediction:
    return {"total": None, "label": UNSUPPORTED, "reasons": [reason]}


def request_bytes(
    calibration: Calibration, inputs: UnitInputs, configuration: str
) -> float:
    sizes = calibration["request_bytes"]
    if configuration == CLOUD:
        return inputs.image_bytes + float(sizes[CLOUD]["framing"])
    return float(sizes[configuration]["bytes"])


def _fit_request_bytes(units: list[Unit]) -> dict[str, dict[str, float]]:
    cloud = [u for u in units if u.configuration == CLOUD]
    sizes: dict[str, dict[str, float]] = {
        CLOUD: {
            "framing": median(
                u.targets["request_bytes"] - u.inputs.image_bytes for u in cloud
            )
        }
    }
    for configuration in SPLITS:
        members = [u for u in units if u.configuration == configuration]
        sizes[configuration] = {
            "bytes": median(u.targets["request_bytes"] for u in members)
        }
    return sizes


# --------------------------------------------------------------------------
# r2_component


def fit_component(training: Iterable[Unit]) -> Calibration:
    units = _fitted(training)
    clamped: list[str] = []
    edge_groups: dict[str, list[Unit]] = defaultdict(list)
    server_groups: dict[str, list[Unit]] = defaultdict(list)
    residual_groups: dict[str, list[Unit]] = defaultdict(list)
    for u in units:
        i, c = u.inputs, u.configuration
        if c == EDGE:
            edge_groups[_key(c, i.task, i.edge_load)].append(u)
        elif c in SPLITS:
            edge_groups[_key(c, "*", i.edge_load)].append(u)
        if c == CLOUD or c in SPLITS:
            server_groups[_key(c, i.task, i.cloud_load)].append(u)
        residual_groups[_key(c, i.edge_load)].append(u)

    def line(group: str, members: list[Unit], target: str) -> list[float]:
        a, b, was = fit_line(
            [m.inputs.crops for m in members], [m.targets[target] for m in members]
        )
        if was:
            clamped.append(f"{target}:{group}")
        return [a, b]

    edge = {g: line(g, m, "edge") for g, m in sorted(edge_groups.items())}
    server = {
        g: (
            line(g, m, "server")
            if g.startswith(CLOUD)
            else [median(x.targets["server"] for x in m), 0.0]
        )
        for g, m in sorted(server_groups.items())
    }
    residual = {
        g: median(x.targets["residual"] for x in m)
        for g, m in sorted(residual_groups.items())
    }
    calibration: Calibration = {
        "schema": SCHEMA,
        "model": "r2_component",
        "edge": edge,
        "server": server,
        "residual": residual,
        "request_bytes": _fit_request_bytes(units),
        "support": _support(units),
    }
    remote = [u for u in units if u.configuration in REMOTE]

    def network_fit(window: float) -> tuple[float, dict[str, list[float]], list[str]]:
        loss, lines, was_clamped = 0.0, {}, []
        for configuration in REMOTE:
            members = [u for u in remote if u.configuration == configuration]
            xs = [
                network_phi_ms(
                    request_bytes(calibration, u.inputs, configuration),
                    u.inputs.uplink_bps,
                    u.inputs.rtt_ms,
                    window,
                )
                for u in members
            ]
            ys = [u.targets["transport"] for u in members]
            a, b, was = _fit_network_line(xs, ys)
            if was:
                was_clamped.append(f"network:{configuration}")
            lines[configuration] = [a, b]
            loss += sum(((a + b * x - y) / y) ** 2 for x, y in zip(xs, ys, strict=True))
        return loss, lines, was_clamped

    scored = []
    for kib in INITIAL_WINDOWS_KIB:
        loss, lines, was = network_fit(kib * 1024.0)
        scored.append((loss, kib, lines, was))
    loss, kib, lines, was = min(scored, key=lambda s: (s[0], s[1]))
    calibration["initial_window_bytes"] = kib * 1024
    calibration["initial_window_search"] = {str(s[1]): s[0] for s in scored}
    calibration["network"] = lines
    calibration["clamped"] = clamped + was
    return calibration


def predict_component(
    calibration: Calibration, inputs: UnitInputs, configuration: str
) -> Prediction:
    if configuration not in CONFIGURATIONS:
        return _invalid(f"unknown configuration {configuration}")
    if not _finite_inputs(inputs):
        return _invalid("missing or non-finite input")
    k = inputs.crops
    edge = server = network = 0.0
    if configuration in (EDGE, *SPLITS):
        group = _key(
            configuration,
            inputs.task if configuration == EDGE else "*",
            inputs.edge_load,
        )
        if group not in calibration["edge"]:
            return _invalid(f"no edge-stage fit for {group}")
        a, b = calibration["edge"][group]
        edge = a + b * k
    if configuration in REMOTE:
        group = _key(configuration, inputs.task, inputs.cloud_load)
        if group not in calibration["server"]:
            return _invalid(f"no server fit for {group}")
        a, b = calibration["server"][group]
        server = a + b * k
        a, b = calibration["network"][configuration]
        network = a + b * network_phi_ms(
            request_bytes(calibration, inputs, configuration),
            inputs.uplink_bps,
            inputs.rtt_ms,
            float(calibration["initial_window_bytes"]),
        )
    group = _key(configuration, inputs.edge_load)
    if group not in calibration["residual"]:
        return _invalid(f"no residual fit for {group}")
    residual = float(calibration["residual"][group])
    total = edge + server + network + residual
    if not math.isfinite(total) or total <= 0:
        return _invalid("non-positive prediction")
    label, reasons = _label(calibration["support"], inputs, configuration)
    return {
        "total": total,
        "edge": edge,
        "server": server,
        "network": network,
        "residual": residual,
        "label": label,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------
# r2_additive

_ADDITIVE_COLUMNS = ("L1", "C2", "crops", "rtt_ms", "payload_ms")


def _additive_row(
    calibration: Calibration, inputs: UnitInputs, configuration: str
) -> dict[str, float]:
    row = {
        "L1": 1.0 if inputs.edge_load == "L1" else 0.0,
        "C2": 1.0 if inputs.cloud_load == "C2" else 0.0,
        "crops": float(inputs.crops),
    }
    if configuration in REMOTE:
        row["rtt_ms"] = inputs.rtt_ms
        row["payload_ms"] = (
            8000.0
            * request_bytes(calibration, inputs, configuration)
            / inputs.uplink_bps
        )
    return row


def fit_additive(training: Iterable[Unit]) -> Calibration:
    units = _fitted(training)
    calibration: Calibration = {
        "schema": SCHEMA,
        "model": "r2_additive",
        "request_bytes": _fit_request_bytes(units),
        "support": _support(units),
        "groups": {},
    }
    groups: dict[str, list[Unit]] = defaultdict(list)
    for u in units:
        groups[_key(u.configuration, u.inputs.task)].append(u)
    for group, members in sorted(groups.items()):
        configuration = members[0].configuration
        rows = [_additive_row(calibration, m.inputs, configuration) for m in members]
        columns = [
            c
            for c in _ADDITIVE_COLUMNS
            if c in rows[0] and len({r[c] for r in rows}) > 1
        ]
        design = [[1.0, *(r[c] for c in columns)] for r in rows]
        coefficients = least_squares(design, [m.targets["e2e"] for m in members])
        calibration["groups"][group] = {
            "columns": columns,
            "coefficients": coefficients,
            "edge_levels": sorted({m.inputs.edge_load for m in members}),
            "cloud_levels": sorted({m.inputs.cloud_load for m in members}),
        }
    return calibration


def predict_additive(
    calibration: Calibration, inputs: UnitInputs, configuration: str
) -> Prediction:
    if configuration not in CONFIGURATIONS:
        return _invalid(f"unknown configuration {configuration}")
    if not _finite_inputs(inputs):
        return _invalid("missing or non-finite input")
    group = calibration["groups"].get(_key(configuration, inputs.task))
    if group is None:
        return _invalid(f"no fit for {configuration} {inputs.task}")
    if (
        inputs.edge_load not in group["edge_levels"]
        or inputs.cloud_load not in group["cloud_levels"]
    ):
        return _invalid("load level absent from training for this group")
    row = _additive_row(calibration, inputs, configuration)
    values = [1.0, *(row[c] for c in group["columns"])]
    total = sum(c * v for c, v in zip(group["coefficients"], values, strict=True))
    if not math.isfinite(total) or total <= 0:
        return _invalid("non-positive prediction")
    label, reasons = _label(calibration["support"], inputs, configuration)
    return {"total": total, "label": label, "reasons": reasons}


# --------------------------------------------------------------------------
# constant_median


def fit_constant(training: Iterable[Unit]) -> Calibration:
    groups: dict[str, list[float]] = defaultdict(list)
    for u in _fitted(training):
        groups[_key(u.configuration, u.inputs.task)].append(u.targets["e2e"])
    return {
        "schema": SCHEMA,
        "model": "constant_median",
        "medians": {g: median(v) for g, v in sorted(groups.items())},
    }


def predict_constant(
    calibration: Calibration, inputs: UnitInputs, configuration: str
) -> Prediction:
    value = calibration["medians"].get(_key(configuration, inputs.task))
    if value is None:
        return _invalid(f"no median for {configuration} {inputs.task}")
    return {"total": float(value), "label": INTERPOLATION, "reasons": []}


FITTERS: dict[str, Callable[[Iterable[Unit]], Calibration]] = {
    "r2_component": fit_component,
    "r2_additive": fit_additive,
    "constant_median": fit_constant,
}
PREDICTORS: dict[str, Callable[[Calibration, UnitInputs, str], Prediction]] = {
    "r2_component": predict_component,
    "r2_additive": predict_additive,
    "constant_median": predict_constant,
}


def predict(
    calibration: Calibration, inputs: UnitInputs, configuration: str
) -> Prediction:
    if calibration.get("schema") != SCHEMA:
        raise ValueError("not an r2 calibration")
    return PREDICTORS[calibration["model"]](calibration, inputs, configuration)
