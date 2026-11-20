"""Predictor V2 r2: features, models and the G3 decision on synthetic units."""

from __future__ import annotations

import pytest

from adaptive_vlm.prediction.r2.evaluation import (
    decide,
    grouped_folds,
    predict_units,
    ranking_summary,
    ranking_units,
    score_model,
)
from adaptive_vlm.prediction.r2.features import (
    crop_count,
    network_phi_ms,
    slow_start_transfer_ms,
)
from adaptive_vlm.prediction.r2.models import (
    UNSUPPORTED,
    fit_additive,
    fit_component,
    fit_constant,
    predict,
)
from adaptive_vlm.prediction.r2.units import (
    CLOUD,
    CONFIGURATIONS,
    EDGE,
    Unit,
    UnitInputs,
    build_units,
)

INT8 = "distributed/projected/int8"
WINDOW = 64 * 1024


@pytest.mark.parametrize(
    ("width", "height", "crops"),
    [
        (500, 375, 2),
        (375, 500, 2),
        (336, 500, 2),
        (640, 480, 9),
        (640, 427, 9),
        (427, 640, 9),
        (640, 513, 10),
        (500, 405, 10),
        (640, 640, 10),
    ],
)
def test_crop_count_matches_the_pinned_tiling(
    width: int, height: int, crops: int
) -> None:
    assert crop_count(width, height) == crops


def test_slow_start_limits() -> None:
    # Rate-bound: a window already above the bandwidth-delay product.
    assert slow_start_transfer_ms(1_000_000, 8e6, 1.0, WINDOW) == pytest.approx(1000.0)
    # Latency-bound: 64 KiB doubling to cover 1,000,000 bytes takes 4 full
    # rounds (64+128+256+512 KiB) and a short last round.
    fast = slow_start_transfer_ms(1_000_000, 1e12, 100.0, WINDOW)
    assert 400.0 <= fast < 401.0
    assert network_phi_ms(0, 1e6, 50.0, WINDOW) == pytest.approx(100.0)
    slower = slow_start_transfer_ms(2e6, 5e7, 100.0, WINDOW)
    assert slower > slow_start_transfer_ms(2e6, 5e7, 20.0, WINDOW)
    assert slower > slow_start_transfer_ms(1e6, 5e7, 100.0, WINDOW)
    with pytest.raises(ValueError):
        slow_start_transfer_ms(1.0, 0.0, 1.0, WINDOW)


def _inputs(
    request: int, condition: str, edge: str, cloud: str, rate: float, rtt: float
) -> UnitInputs:
    width, height = ((640, 480), (500, 375), (640, 640))[request % 3]
    cell = f"{condition}-{edge}" + ("-C2" if cloud == "C2" else "")
    return UnitInputs(
        request_id=f"q{request}",
        cell_id=cell,
        condition_id=condition,
        task="short",
        edge_load=edge,
        cloud_load=cloud,
        width=width,
        height=height,
        image_bytes=100_000 + 1_000 * request,
        uplink_bps=rate * 1e6,
        rtt_ms=rtt,
    )


def _truth(i: UnitInputs, configuration: str) -> dict[str, float]:
    """A synthetic world that follows the component model exactly."""

    k = i.crops
    factor = 1.4 if i.edge_load == "L1" else 1.0
    penalty = 300.0 if i.cloud_load == "C2" else 0.0
    edge = server = transport = 0.0
    size = 0.0
    if configuration == EDGE:
        edge = factor * (450.0 + 55.0 * k)
    elif configuration == CLOUD:
        server = 100.0 + 2 * penalty + 19.0 * k
        size = i.image_bytes + 400
    else:
        edge = factor * (10.0 + 57.0 * k)
        server = 90.0 + penalty
        size = 1_494_000 if configuration == INT8 else 2_986_000
    if configuration != EDGE:
        transport = 5.0 + network_phi_ms(size, i.uplink_bps, i.rtt_ms, WINDOW)
    residual = 1.0
    return {
        "e2e": edge + server + transport + residual,
        "edge": edge,
        "server": server,
        "transport": transport,
        "residual": residual,
        "request_bytes": size,
    }


CONDITIONS = {
    "T01": (5.0, 20.0),
    "T02": (25.0, 20.0),
    "T03": (100.0, 5.0),
    "T04": (100.0, 100.0),
    "T05": (150.0, 150.0),
}


def _world(requests: range = range(12)) -> list[Unit]:
    units = []
    for condition, (rate, rtt) in CONDITIONS.items():
        for edge in ("L0", "L1"):
            for cloud in ("C0", "C2"):
                for request in requests:
                    inputs = _inputs(request, condition, edge, cloud, rate, rtt)
                    for configuration in CONFIGURATIONS:
                        units.append(
                            Unit(
                                inputs=inputs,
                                configuration=configuration,
                                targets=_truth(inputs, configuration),
                                successes=2,
                                attempts=2,
                            )
                        )
    return units


def test_component_model_recovers_a_component_world() -> None:
    units = _world()
    calibration = fit_component(units)
    assert calibration["initial_window_bytes"] == WINDOW
    assert calibration["clamped"] == []
    for unit in units[:200]:
        prediction = predict(calibration, unit.inputs, unit.configuration)
        assert prediction["total"] == pytest.approx(unit.targets["e2e"], rel=1e-6)
        parts = sum(prediction[k] for k in ("edge", "server", "network", "residual"))
        assert parts == pytest.approx(prediction["total"])
        assert prediction["label"] == "interpolation"


def test_coverage_labels() -> None:
    calibration = fit_component(_world())
    inside = _inputs(1, "X", "L0", "C0", 50.0, 50.0)
    assert predict(calibration, inside, INT8)["label"] == "interpolation"
    far = _inputs(1, "X", "L0", "C0", 400.0, 50.0)
    labelled = predict(calibration, far, INT8)
    assert labelled["label"] == "physics_extrapolation"
    assert labelled["total"] is not None
    # edge/full has no network input, so the link cannot push it outside support.
    assert predict(calibration, far, EDGE)["label"] == "interpolation"
    long_task = UnitInputs(**{**inside.__dict__, "task": "long"})
    assert predict(calibration, long_task, CLOUD)["label"] == UNSUPPORTED
    assert predict(calibration, inside, "edge/other")["label"] == UNSUPPORTED


def test_additive_and_constant_models_fit_and_predict() -> None:
    units = _world()
    additive = fit_additive(units)
    group = additive["groups"][f"{CLOUD}|short"]
    assert group["columns"] == ["L1", "C2", "crops", "rtt_ms", "payload_ms"]
    assert additive["groups"][f"{EDGE}|short"]["columns"] == ["L1", "C2", "crops"]
    constant = fit_constant(units)
    unit = units[0]
    assert predict(additive, unit.inputs, unit.configuration)["total"] > 0
    assert predict(constant, unit.inputs, unit.configuration)["total"] == pytest.approx(
        constant["medians"][f"{unit.configuration}|short"]
    )


def test_build_units_take_medians_over_repetitions() -> None:
    def snapshot(rep: int) -> dict[str, object]:
        return {
            "request_id": "q1",
            "cell_id": "T01-L0",
            "condition_id": "T01",
            "configuration_id": CLOUD,
            "task_name": "short",
            "edge_load_level": "L0",
            "cloud_load_level": "C0",
            "task": {"image_width": 640, "image_height": 480, "image_bytes": 1000},
            "network": {"uplink_bps": 1e6 * (5 + rep), "rtt_ms": 20.0},
        }

    def target(e2e: float, status: str = "ok") -> dict[str, object]:
        return {
            "status": status,
            "observed_end_to_end_ms": e2e,
            "fields": {
                "edge_total_ms": None,
                "network_exchange_ms": e2e - 1,
                "server_processing_ms": 100.0,
                "request_wire_bytes": 1400,
            },
        }

    units = build_units([(snapshot(0), target(300.0)), (snapshot(1), target(500.0))])
    assert len(units) == 1
    unit = units[0]
    assert unit.targets["e2e"] == 400.0
    assert unit.targets["transport"] == pytest.approx(299.0)
    assert unit.inputs.uplink_bps == pytest.approx(5.5e6)
    failed = build_units([(snapshot(0), target(300.0, "error"))])[0]
    assert failed.successes == 0 and failed.attempts == 1 and failed.targets == {}


def test_folds_are_seeded_and_grouped() -> None:
    folds = grouped_folds([f"q{i}" for i in range(10)] * 3)
    assert folds == grouped_folds(f"q{i}" for i in range(10))
    assert sorted(folds.values()) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_scoring_gate_and_decision() -> None:
    units = _world(range(6))
    component = fit_component(units)
    constant = fit_constant(units)
    records = predict_units(component, units)
    baseline = predict_units(constant, units)
    intervals = {c: 1.0 for c in CONFIGURATIONS}
    scores = score_model(records, baseline, intervals)
    assert scores["gate"]["pass"] is True
    assert scores["ranking"]["agreement_percent"] == 100.0
    assert scores["ranking"]["mean_regret_ms"] == 0.0
    baseline_scores = score_model(baseline, baseline, intervals)
    assert baseline_scores["gate"]["numeric_pass"] is False
    ranking = ranking_summary(ranking_units(baseline))
    assert ranking["agreement"] < scores["ranking"]["agreement"]
    choice = decide(
        "r2_component",
        {"r2_component": scores, "r2_additive": baseline_scores},
    )
    assert choice == choice | {"selected": "r2_component", "g3": "passed"}
    failed = decide(
        "r2_component",
        {"r2_component": baseline_scores, "r2_additive": baseline_scores},
    )
    assert failed["selected"] is None and failed["g3"] == "failed"


def test_unsafe_choice_counts_a_failed_attempt() -> None:
    units = _world(range(3))
    units[0].successes = 1
    component = fit_component(units)
    records = predict_units(component, units)
    scores = score_model(records, records, {c: 1.0 for c in CONFIGURATIONS})
    assert scores["unsafe_feasibility_choices"] == 1
    assert scores["gate"]["safety_pass"] is False
