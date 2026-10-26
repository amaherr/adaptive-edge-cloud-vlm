"""Fit a small, provenance-bound latency model without held-out outcomes."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile with explicitly bounded fraction."""

    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile requires values and a fraction in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _field(row: dict[str, Any], name: str) -> float:
    return _number(row["fields"].get(name), name)


def load_profile_and_split(
    profile_path: Path, split_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    profile_bytes = profile_path.read_bytes()
    profile_hash = hashlib.sha256(profile_bytes).hexdigest()
    split_bytes = split_path.read_bytes()
    split_hash = hashlib.sha256(split_bytes).hexdigest()
    split = json.loads(split_bytes)
    if not isinstance(split, dict) or split.get("schema_version") != 1:
        raise ValueError("invalid predictor split specification")
    if split.get("source_profile_sha256") != profile_hash:
        raise ValueError("profile SHA-256 differs from frozen split specification")
    train_ids = split.get("training_request_ids")
    holdout_ids = split.get("holdout_request_ids")
    if not isinstance(train_ids, list) or not isinstance(holdout_ids, list):
        raise ValueError("training and holdout IDs must be lists")
    if (
        not train_ids
        or not holdout_ids
        or len(set(train_ids)) != len(train_ids)
        or len(set(holdout_ids)) != len(holdout_ids)
        or set(train_ids) & set(holdout_ids)
    ):
        raise ValueError("training and holdout IDs must be nonempty and disjoint")
    network_holdout = split.get("holdout_network_bucket_ids")
    if not isinstance(network_holdout, list) or len(set(network_holdout)) != len(
        network_holdout
    ):
        raise ValueError("network holdout IDs must be a unique list")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(profile_bytes.decode().splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("schema_version") != 1:
            raise ValueError(f"invalid profile row {number}")
        if (
            row.get("status") == "ok"
            and _number(row.get("observed_end_to_end_ms"), "observed_end_to_end_ms")
            <= 0
        ):
            raise ValueError(f"non-positive successful latency in profile row {number}")
        rows.append(row)
    if not rows:
        raise ValueError("profile has no rows")
    measured_ids = {row["request_id"] for row in rows if not row["warmup"]}
    if measured_ids != set(train_ids) | set(holdout_ids):
        raise ValueError("split request IDs do not partition measured profile IDs")
    model_ids = {row["model_contract_sha256"] for row in rows}
    dataset_ids = {row["dataset_manifest_sha256"] for row in rows}
    host_ids = {row["host_pair_id"] for row in rows}
    if any(len(values) != 1 for values in (model_ids, dataset_ids, host_ids)):
        raise ValueError("profile mixes model, dataset, or host pair")
    return rows, split, profile_hash, split_hash


def _network_fit(rows: list[dict[str, Any]], mode: str) -> NetworkFit:
    if mode == "edge":
        return NetworkFit(
            "none", None, None, 0.0, 0.0, 0.0, None, None, None, None, None, None
        )
    states = [row["network_state"] for row in rows]
    controlled = {state["controlled"] for state in states}
    if len(controlled) != 1:
        raise ValueError("calibration cannot mix controlled and uncontrolled networks")
    if not controlled.pop():
        buckets = {(state["bucket_id"], state["source"]) for state in states}
        if len(buckets) != 1:
            raise ValueError("uncontrolled calibration requires one network bucket")
        bucket, source = next(iter(buckets))
        return NetworkFit(
            "loopback",
            source,
            bucket,
            median(_field(row, "transport_residual_ms") for row in rows),
            0.0,
            0.0,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    sources = {state["source"] for state in states}
    if len(sources) != 1:
        raise ValueError("controlled calibration mixes network state sources")
    corrected: list[tuple[float, float]] = []
    for row, state in zip(rows, states, strict=True):
        up = _number(state["uplink_bps"], "uplink_bps")
        down = _number(state["downlink_bps"], "downlink_bps")
        rtt = _number(state["rtt_ms"], "rtt_ms")
        corrected.append(
            (
                rtt,
                _field(row, "transport_residual_ms")
                - transfer_ms(_field(row, "request_wire_bytes"), up)
                - transfer_ms(_field(row, "response_wire_bytes"), down),
            )
        )
    by_rtt: dict[float, list[float]] = defaultdict(list)
    for rtt, value in corrected:
        by_rtt[rtt].append(value)
    points = sorted((rtt, median(values)) for rtt, values in by_rtt.items())
    slopes = [
        (y2 - y1) / (x2 - x1)
        for index, (x1, y1) in enumerate(points)
        for x2, y2 in points[index + 1 :]
    ]
    rtt_coefficient = max(0.0, median(slopes)) if slopes else 0.0
    intercept = median(value - rtt_coefficient * rtt for rtt, value in corrected)
    return NetworkFit(
        "controlled",
        next(iter(sources)),
        None,
        0.0,
        intercept,
        rtt_coefficient,
        min(_number(state["uplink_bps"], "uplink_bps") for state in states),
        max(_number(state["uplink_bps"], "uplink_bps") for state in states),
        min(_number(state["downlink_bps"], "downlink_bps") for state in states),
        max(_number(state["downlink_bps"], "downlink_bps") for state in states),
        min(_number(state["rtt_ms"], "rtt_ms") for state in states),
        max(_number(state["rtt_ms"], "rtt_ms") for state in states),
    )


def _candidate_fit(
    rows: list[dict[str, Any]], *, minimum_rows: int, headroom: float
) -> CandidateFit:
    if len(rows) < minimum_rows or len({row["request_id"] for row in rows}) < 3:
        raise ValueError("candidate has insufficient independent training samples")
    config_id = rows[0]["configuration_id"]
    mode = config_id.split("/")[0]
    placement_hashes = {canonical_sha256(row["placement_profile"]) for row in rows}
    task_buckets = {row["task_bucket"] for row in rows}
    budgets = {row["requested_max_new_tokens"] for row in rows}
    if len(placement_hashes) != 1 or len(task_buckets) != 1 or len(budgets) != 1:
        raise ValueError("candidate mixes placement, task bucket, or generation budget")
    decode_rates = [
        _field(row, "decode_ms") / _field(row, "actual_output_tokens")
        for row in rows
        if _field(row, "actual_output_tokens") > 0
    ]
    decode_rate = median(decode_rates) if decode_rates else 0.0
    edge_compute = (
        median(
            _field(row, "edge_total_ms")
            - (_field(row, "decode_ms") if mode == "edge" else 0.0)
            for row in rows
        )
        if mode != "cloud"
        else 0.0
    )
    server_fixed = (
        median(
            _field(row, "server_processing_ms") - _field(row, "decode_ms")
            for row in rows
        )
        if mode != "edge"
        else 0.0
    )
    if edge_compute < 0 or server_fixed < 0:
        raise ValueError("negative non-decode compute component")
    client_preprocess = (
        median(
            _number(row["raw_metrics"].get("image_read_ms"), "image_read_ms")
            for row in rows
        )
        if mode == "cloud"
        else 0.0
    )
    serialization = (
        median(_field(row, "serialization_ms") for row in rows)
        if mode == "distributed"
        else 0.0
    )
    memory_values = [row["fields"]["edge_peak_reserved_mib"] for row in rows]
    edge_memory = (
        max(_number(value, "edge_peak_reserved_mib") for value in memory_values)
        * (1 + headroom)
        if mode != "cloud" and all(value is not None for value in memory_values)
        else None
    )
    payload = (
        median(_field(row, "request_payload_bytes") for row in rows)
        if mode != "edge"
        else 0.0
    )
    framing = (
        median(
            _field(row, "request_wire_bytes") - _field(row, "request_payload_bytes")
            for row in rows
        )
        if mode != "edge"
        else 0.0
    )
    return CandidateFit(
        configuration_id=config_id,
        placement_profile_sha256=next(iter(placement_hashes)),
        sample_count=len(rows),
        task_bucket=next(iter(task_buckets)),
        image_bytes_min=int(min(_field(row, "image_bytes") for row in rows)),
        image_bytes_max=int(max(_field(row, "image_bytes") for row in rows)),
        image_width_min=int(min(_field(row, "image_width") for row in rows)),
        image_width_max=int(max(_field(row, "image_width") for row in rows)),
        image_height_min=int(min(_field(row, "image_height") for row in rows)),
        image_height_max=int(max(_field(row, "image_height") for row in rows)),
        prompt_tokens_min=int(min(_field(row, "prompt_tokens") for row in rows)),
        prompt_tokens_max=int(max(_field(row, "prompt_tokens") for row in rows)),
        requested_max_new_tokens=int(next(iter(budgets))),
        expected_output_tokens=median(
            _field(row, "actual_output_tokens") for row in rows
        ),
        edge_compute_ms=edge_compute,
        client_preprocess_ms=client_preprocess,
        serialization_ms=serialization,
        server_fixed_ms=server_fixed,
        decode_ms_per_token=decode_rate,
        client_residual_ms=median(_field(row, "stage_residual_ms") for row in rows),
        request_payload_bytes=payload,
        request_framing_bytes=framing,
        response_wire_bytes=(
            median(_field(row, "response_wire_bytes") for row in rows)
            if mode != "edge"
            else 0.0
        ),
        edge_memory_requirement_mib=edge_memory,
        constant_median_ms=median(
            _number(row["observed_end_to_end_ms"], "observed") for row in rows
        ),
        interval_half_width_ms=0.0,
        network=_network_fit(rows, mode),
    )


def task_and_state_from_row(
    row: dict[str, Any],
) -> tuple[PredictionTask, PredictionState]:
    state = row["network_state"]
    task = PredictionTask(
        task_bucket=row["task_bucket"],
        image_bytes=int(_field(row, "image_bytes")),
        image_width=int(_field(row, "image_width")),
        image_height=int(_field(row, "image_height")),
        prompt_tokens=int(_field(row, "prompt_tokens")),
        requested_max_new_tokens=int(row["requested_max_new_tokens"]),
        expected_output_tokens=None,  # Never read held-out generated length.
    )
    prediction_state = PredictionState(
        host_pair_id=row["host_pair_id"],
        model_contract_sha256=row["model_contract_sha256"],
        placement_profile_sha256=canonical_sha256(row["placement_profile"]),
        network_bucket_id=state["bucket_id"],
        network_source=state["source"],
        uplink_bps=state.get("uplink_bps"),
        downlink_bps=state.get("downlink_bps"),
        rtt_ms=state.get("rtt_ms"),
    )
    return task, prediction_state


def fit_calibration(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    profile_sha256: str,
    split_sha256: str,
) -> Calibration:
    train_ids = set(split["training_request_ids"])
    heldout_network = set(split["holdout_network_bucket_ids"])
    training = [
        row
        for row in rows
        if not row["warmup"]
        and row["request_id"] in train_ids
        and row["network_state"]["bucket_id"] not in heldout_network
        and row["status"] == "ok"
    ]
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in training:
        by_candidate[row["configuration_id"]].append(row)
    required = {spec.id.value for spec in CONFIGURATIONS}
    if set(by_candidate) != required:
        raise ValueError("training profile lacks one or more baseline configurations")
    headroom = _number(split["edge_memory_headroom_fraction"], "headroom")
    if not 0 <= headroom <= 1:
        raise ValueError("memory headroom fraction must be in [0, 1]")
    candidates = {
        name: _candidate_fit(
            candidate_rows,
            minimum_rows=int(split["minimum_training_rows_per_configuration"]),
            headroom=headroom,
        )
        for name, candidate_rows in by_candidate.items()
    }
    reference = rows[0]
    calibration = Calibration(
        schema_version=1,
        calibration_id="pending",
        source_profile_sha256=profile_sha256,
        split_specification_sha256=split_sha256,
        model_contract_sha256=reference["model_contract_sha256"],
        dataset_manifest_sha256=reference["dataset_manifest_sha256"],
        host_pair_id=reference["host_pair_id"],
        training_request_ids=tuple(split["training_request_ids"]),
        holdout_request_ids=tuple(split["holdout_request_ids"]),
        holdout_network_bucket_ids=tuple(split["holdout_network_bucket_ids"]),
        candidates=candidates,
    )
    predictor = Predictor(calibration)
    fraction = _number(split["interval_absolute_error_quantile"], "interval quantile")
    if not 0 < fraction < 1:
        raise ValueError("interval quantile must be between zero and one")
    for spec in CONFIGURATIONS:
        name = spec.id.value
        fit = candidates[name]
        errors = []
        for row in by_candidate[name]:
            task, state = task_and_state_from_row(row)
            prediction = predictor.predict(spec, state, task)
            if not prediction.valid or prediction.total_ms is None:
                raise ValueError(f"training row is out of coverage: {name}")
            errors.append(
                abs(
                    prediction.total_ms
                    - _number(row["observed_end_to_end_ms"], "observed_end_to_end_ms")
                )
            )
        candidates[name] = replace(
            fit, interval_half_width_ms=percentile(errors, fraction)
        )
    calibration = replace(calibration, candidates=candidates)
    identifier = canonical_sha256(
        {
            key: value
            for key, value in calibration.to_dict().items()
            if key != "calibration_id"
        }
    )
    return replace(calibration, calibration_id=identifier)
