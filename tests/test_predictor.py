from __future__ import annotations

from dataclasses import replace

import pytest

from adaptive_vlm.benchmark.calibrate import _network_fit, percentile
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS
from adaptive_vlm.scheduler.predictor import (
    Calibration,
    CandidateFit,
    NetworkFit,
    PredictionState,
    PredictionTask,
    Predictor,
    transfer_ms,
)

FINGERPRINT = "a" * 64
PLACEMENT = "b" * 64
TASK = PredictionTask("fixture", 180, 4, 4, 11, 32)
CLOUD = CONFIGURATIONS[1]
EDGE = CONFIGURATIONS[0]


def _fit(
    network: NetworkFit, *, configuration_id: str = CLOUD.id.value
) -> CandidateFit:
    return CandidateFit(
        configuration_id=configuration_id,
        placement_profile_sha256=PLACEMENT,
        sample_count=10,
        task_bucket="fixture",
        image_bytes_min=160,
        image_bytes_max=210,
        image_width_min=4,
        image_width_max=4,
        image_height_min=4,
        image_height_max=4,
        prompt_tokens_min=9,
        prompt_tokens_max=13,
        requested_max_new_tokens=32,
        expected_output_tokens=8.0,
        edge_compute_ms=0.0,
        client_preprocess_ms=0.2,
        serialization_ms=0.0,
        server_fixed_ms=20.0,
        decode_ms_per_token=2.0,
        client_residual_ms=0.3,
        request_payload_bytes=180.0,
        request_framing_bytes=300.0,
        response_wire_bytes=1_000.0,
        edge_memory_requirement_mib=None,
        constant_median_ms=37.0,
        interval_half_width_ms=4.0,
        network=network,
    )


def _predictor(fit: CandidateFit) -> Predictor:
    return Predictor(
        Calibration(
            schema_version=1,
            calibration_id="calibration-1",
            source_profile_sha256="c" * 64,
            split_specification_sha256="d" * 64,
            model_contract_sha256=FINGERPRINT,
            dataset_manifest_sha256="e" * 64,
            host_pair_id="host-a",
            training_request_ids=("a", "b", "c"),
            holdout_request_ids=("d",),
            holdout_network_bucket_ids=(),
            candidates={fit.configuration_id: fit},
        )
    )


def _state(**changes: object) -> PredictionState:
    values = dict(
        host_pair_id="host-a",
        model_contract_sha256=FINGERPRINT,
        placement_profile_sha256=PLACEMENT,
        network_bucket_id="controlled-a",
        network_source="replay",
        uplink_bps=2_000_000.0,
        downlink_bps=2_000_000.0,
        rtt_ms=10.0,
    )
    values.update(changes)
    return PredictionState(**values)


def _controlled() -> NetworkFit:
    return NetworkFit(
        kind="controlled",
        source=None,
        bucket_id=None,
        transport_ms=0.0,
        intercept_ms=2.0,
        rtt_coefficient=0.5,
        uplink_min_bps=1_000_000.0,
        uplink_max_bps=10_000_000.0,
        downlink_min_bps=1_000_000.0,
        downlink_max_bps=10_000_000.0,
        rtt_min_ms=5.0,
        rtt_max_ms=30.0,
    )


def test_transfer_units_and_bandwidth_monotonicity() -> None:
    assert transfer_ms(1_000_000, 8_000_000) == 1000.0
    assert transfer_ms(1_000_000, 16_000_000) == 500.0
    predictor = _predictor(_fit(_controlled()))
    slow = predictor.predict(CLOUD, _state(), TASK)
    fast = predictor.predict(
        CLOUD, _state(uplink_bps=8_000_000.0, downlink_bps=8_000_000.0), TASK
    )
    assert slow.valid and fast.valid
    assert slow.total_ms is not None and fast.total_ms is not None
    assert fast.total_ms < slow.total_ms
    assert slow.total_ms == pytest.approx(sum(slow.components_ms.values()))
    assert slow.request_wire_bytes == TASK.image_bytes + 300
    assert slow.response_wire_bytes == 1_000
    assert predictor.predict(CLOUD, _state(), TASK) == slow


def test_fingerprint_task_network_and_memory_exclusions() -> None:
    cloud = _predictor(_fit(_controlled()))
    assert (
        cloud.predict(CLOUD, _state(host_pair_id="host-b"), TASK).reason
        == "host_pair_mismatch"
    )
    assert (
        cloud.predict(CLOUD, _state(model_contract_sha256="f" * 64), TASK).reason
        == "model_contract_mismatch"
    )
    assert (
        cloud.predict(CLOUD, _state(placement_profile_sha256="f" * 64), TASK).reason
        == "placement_profile_mismatch"
    )
    assert (
        cloud.predict(CLOUD, _state(), replace(TASK, prompt_tokens=None)).reason
        == "prompt_tokens_unknown"
    )
    assert (
        cloud.predict(CLOUD, _state(), replace(TASK, image_width=5)).reason
        == "task_out_of_coverage"
    )
    assert (
        cloud.predict(CLOUD, _state(), replace(TASK, task_bucket="different")).reason
        == "task_bucket_mismatch"
    )
    assert (
        cloud.predict(CLOUD, _state(uplink_bps=None), TASK).reason
        == "network_state_missing"
    )
    assert (
        cloud.predict(CLOUD, _state(uplink_bps=20_000_000), TASK).reason
        == "network_out_of_coverage"
    )
    source_bound = _predictor(_fit(replace(_controlled(), source="measured")))
    assert (
        source_bound.predict(CLOUD, _state(), TASK).reason == "network_source_mismatch"
    )

    edge_network = replace(_controlled(), kind="none")
    unknown = _predictor(_fit(edge_network, configuration_id=EDGE.id.value))
    assert (
        unknown.predict(EDGE, _state(edge_memory_budget_mib=900), TASK).reason
        == "edge_memory_requirement_unknown"
    )
    known_fit = replace(
        _fit(edge_network, configuration_id=EDGE.id.value),
        edge_memory_requirement_mib=1_000.0,
    )
    known = _predictor(known_fit)
    assert (
        known.predict(EDGE, _state(edge_memory_budget_mib=900), TASK).reason
        == "edge_memory_budget_exceeded"
    )
    assert known.predict(EDGE, _state(edge_memory_budget_mib=1_100), TASK).valid


def test_uncontrolled_profile_rejects_claimed_bandwidth() -> None:
    network = NetworkFit(
        "loopback",
        "localhost_uncontrolled",
        "loopback",
        2.0,
        0.0,
        0.0,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    predictor = _predictor(_fit(network))
    local = _state(
        network_bucket_id="loopback",
        network_source="localhost_uncontrolled",
        uplink_bps=None,
        downlink_bps=None,
        rtt_ms=None,
    )
    assert predictor.predict(CLOUD, local, TASK).valid
    assert (
        predictor.predict(CLOUD, replace(local, uplink_bps=1_000_000), TASK).reason
        == "network_out_of_coverage"
    )


def test_percentile_and_invalid_units() -> None:
    assert percentile([1.0, 3.0, 5.0], 0.9) == pytest.approx(4.6)
    with pytest.raises(ValueError):
        transfer_ms(10, 0)
    with pytest.raises(ValueError):
        PredictionTask("fixture", 10, 4, 4, 2, 32, expected_output_tokens=33)


def test_controlled_transport_fit_uses_both_transfer_directions() -> None:
    rows = []
    for rtt in (5.0, 10.0, 20.0):
        up = 2_000_000.0
        down = 4_000_000.0
        request_bytes = 1_000.0
        response_bytes = 2_000.0
        residual = (
            3.0
            + 0.5 * rtt
            + transfer_ms(request_bytes, up)
            + transfer_ms(response_bytes, down)
        )
        rows.append(
            {
                "network_state": {
                    "controlled": True,
                    "source": "controlled-fixture",
                    "uplink_bps": up,
                    "downlink_bps": down,
                    "rtt_ms": rtt,
                },
                "fields": {
                    "request_wire_bytes": request_bytes,
                    "response_wire_bytes": response_bytes,
                    "transport_residual_ms": residual,
                },
            }
        )
    fit = _network_fit(rows, "cloud")
    assert fit.kind == "controlled"
    assert fit.source == "controlled-fixture"
    assert fit.intercept_ms == pytest.approx(3.0)
    assert fit.rtt_coefficient == pytest.approx(0.5)
