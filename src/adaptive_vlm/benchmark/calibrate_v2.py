"""Training-only low-dimensional V2 fitting and grouped internal diagnostics."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.calibrate import (
    _candidate_fit,
    canonical_sha256,
    percentile,
)
from adaptive_vlm.benchmark.predictor_v2_data import (
    BLOCKS,
    load_training,
    number,
    strict_json,
)
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS
from adaptive_vlm.scheduler.predictor import PredictionTask, transfer_ms
from adaptive_vlm.scheduler.predictor_v2 import (
    VARIANTS,
    CalibrationV2,
    PredictorV2,
    state_from_snapshot,
)


def _solve(matrix: list[list[float]], values: list[float]) -> list[float] | None:
    size = len(values)
    a = [[*row, value] for row, value in zip(matrix, values, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda i: abs(a[i][column]))
        if abs(a[pivot][column]) < 1e-12:
            return None
        a[column], a[pivot] = a[pivot], a[column]
        divisor = a[column][column]
        a[column] = [v / divisor for v in a[column]]
        for row in range(size):
            if row == column:
                continue
            multiplier = a[row][column]
            a[row] = [
                x - multiplier * y for x, y in zip(a[row], a[column], strict=True)
            ]
    return [row[-1] for row in a]


def nonnegative_fit(
    x: list[list[float]], y: list[float], *, alpha: float
) -> list[float]:
    """Enumerate active sets for <=3 coefficients; no target/residual clipping."""
    if not x or len(x) != len(y) or len(x[0]) > 3:
        raise ValueError("small nonnegative fit requires nonempty aligned inputs")
    width = len(x[0])
    best = [0.0] * width
    best_loss = sum(v * v for v in y)
    for active_count in range(1, width + 1):
        for subset in itertools.combinations(range(width), active_count):
            a = [
                [
                    sum(row[i] * row[j] for row in x)
                    + (alpha if i == j and i != 0 else 0)
                    for j in subset
                ]
                for i in subset
            ]
            b = [
                sum(row[i] * value for row, value in zip(x, y, strict=True))
                for i in subset
            ]
            solved = _solve(a, b)
            if solved is None or any(not math.isfinite(v) or v < 0 for v in solved):
                continue
            coefficients = [0.0] * width
            for index, value in zip(subset, solved, strict=True):
                coefficients[index] = value
            loss = sum(
                (sum(c * r for c, r in zip(coefficients, row, strict=True)) - value)
                ** 2
                for row, value in zip(x, y, strict=True)
            ) + alpha * sum(v * v for v in coefficients[1:])
            if loss < best_loss:
                best, best_loss = coefficients, loss
    return best


def _constant(value: float) -> dict[str, Any]:
    return {"intercept_ms": value, "slopes": {}, "scales": {}}


def _request_component(
    rows: list[dict[str, Any]],
    targets: list[float],
    features: tuple[str, ...],
    alpha: float,
) -> dict[str, Any]:
    if any(not math.isfinite(value) or value < 0 for value in targets):
        raise ValueError("negative/nonfinite measured nondecode component")
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row["inputs"]["request_id"]].append(i)
    ids = sorted(grouped)
    feature_values = []
    target_values = []
    for rid in ids:
        indexes = grouped[rid]
        task = rows[indexes[0]]["inputs"]["task"]
        feature_values.append(
            {
                "image_pixels": task["image_width"] * task["image_height"],
                "prompt_tokens": task["prompt_tokens"],
            }
        )
        target_values.append(float(median(targets[i] for i in indexes)))
    if not features:
        return _constant(float(median(target_values)))
    scales = {k: float(median(v[k] for v in feature_values)) for k in features}
    x = [[1.0] + [v[k] / scales[k] for k in features] for v in feature_values]
    coefficients = nonnegative_fit(x, target_values, alpha=alpha)
    return {
        "intercept_ms": coefficients[0],
        "slopes": dict(zip(features, coefficients[1:], strict=True)),
        "scales": scales,
    }


def _memory(
    rows: list[dict[str, Any]], name: str, headroom: float, required: bool
) -> float | None:
    if not required:
        return 0.0
    values = [r["target"]["fields"].get(name) for r in rows]
    if any(v is None for v in values):
        return None
    return max(number(v, name, positive=True) for v in values) * (1 + headroom)


def policy_for_fit(contract: dict[str, Any]) -> dict[str, Any]:
    # Full-archive digests/locations and validation/final store digests belong in
    # external source receipts. They cannot affect training coefficients or ID.
    return {
        k: v
        for k, v in contract.items()
        if k not in {"archive_sha256", "archive_prefix", "frozen_split_sha256"}
    }


def _candidate(
    rows: list[dict[str, Any]], variant: str, policy: dict[str, Any]
) -> dict[str, Any]:
    good = [r for r in rows if r["target"]["status"] == "ok"]
    if len({r["inputs"]["request_id"] for r in good}) < 3:
        raise ValueError("candidate lacks independent successful training requests")
    cid = good[0]["inputs"]["configuration_id"]
    mode = cid.split("/")[0]
    targets = [r["target"] for r in good]
    fields = [r["fields"] for r in targets]
    tasks = [r["inputs"]["task"] for r in good]
    decode_rates = [
        number(f["decode_ms"], "decode")
        / number(f["actual_output_tokens"], "output tokens", positive=True)
        for f in fields
        if f["actual_output_tokens"] > 0
    ]
    if not decode_rates:
        raise ValueError("decode rate unidentifiable from empty outputs")
    fit: dict[str, Any] = {
        "placement_profile_sha256": policy["placement_profile_sha256"][cid],
        "successful_training_rows": len(good),
        "scheduled_training_rows": len(rows),
        "independent_training_requests": len({r["inputs"]["request_id"] for r in good}),
        "training_status_counts": dict(Counter(r["target"]["status"] for r in rows)),
        "task_support": {
            k: [min(t[k] for t in tasks), max(t[k] for t in tasks)]
            for k in policy["task_envelope"]
        },
        "expected_output_tokens": float(
            median(f["actual_output_tokens"] for f in fields)
        ),
        "decode_ms_per_token": float(median(decode_rates)),
        "constant_median_ms": float(
            median(r["observed_end_to_end_ms"] for r in targets)
        ),
        "request_framing_bytes": float(
            median(f["request_wire_bytes"] - f["request_payload_bytes"] for f in fields)
        )
        if mode != "edge"
        else 0.0,
        "response_wire_bytes": float(median(f["response_wire_bytes"] for f in fields))
        if mode != "edge"
        else 0.0,
        "edge_memory_requirement_mib": _memory(
            good,
            "edge_peak_reserved_mib",
            policy["memory_headroom_fraction"],
            mode != "cloud",
        ),
        "cloud_memory_requirement_mib": _memory(
            good,
            "cloud_peak_reserved_mib",
            policy["memory_headroom_fraction"],
            mode != "edge",
        ),
        "interval_half_width_ms": 0.0,
    }
    components: dict[str, Any] = {}
    alpha = policy["compute_ridge_alpha"]
    request_aware = variant == "v2_a"
    if mode == "edge":
        components["edge_nondecode_ms"] = _request_component(
            good,
            [
                number(f["edge_total_ms"], "edge") - number(f["decode_ms"], "decode")
                for f in fields
            ],
            ("image_pixels", "prompt_tokens") if request_aware else (),
            alpha,
        )
    if mode == "distributed":
        components["edge_vision_and_projector_ms"] = _request_component(
            good,
            [number(f["edge_total_ms"], "vision") for f in fields],
            ("image_pixels",) if request_aware else (),
            alpha,
        )
        components["tensor_serialization_ms"] = _request_component(
            good,
            [number(f["serialization_ms"], "serialization") for f in fields],
            (),
            alpha,
        )
    if mode != "edge":
        components["cloud_nondecode_including_prefill_ms"] = _request_component(
            good,
            [
                number(f["server_processing_ms"], "server")
                - number(f["decode_ms"], "decode")
                for f in fields
            ],
            (
                ("image_pixels", "prompt_tokens")
                if mode == "cloud"
                else ("prompt_tokens",)
            )
            if request_aware
            else (),
            alpha,
        )
    if mode == "cloud":
        components["client_image_read_ms"] = _request_component(
            good,
            [number(r["raw_metrics"]["image_read_ms"], "image read") for r in targets],
            (),
            alpha,
        )
    components["client_residual_ms"] = _request_component(
        good,
        [number(f["stage_residual_ms"], "client residual") for f in fields],
        (),
        alpha,
    )
    states = [r["inputs"]["network"] for r in good]
    network: dict[str, Any] = {
        "intercept_ms": 0.0,
        "rtt_coefficient": 0.0,
        "support": {
            k: [min(s[k] for s in states), max(s[k] for s in states)]
            for k in ("uplink_bps", "downlink_bps", "rtt_ms")
        },
    }
    if mode != "edge":
        block_residuals: dict[str, list[float]] = defaultdict(list)
        block_rtt = {}
        for row in good:
            task = row["inputs"]["task"]
            state = row["inputs"]["network"]
            payload = task["image_bytes"] if mode == "cloud" else 2985984
            wire = payload + fit["request_framing_bytes"]
            observed_transport = number(
                row["target"]["fields"]["transport_residual_ms"], "transport"
            )
            correction = (
                observed_transport
                - transfer_ms(wire, state["uplink_bps"])
                - transfer_ms(fit["response_wire_bytes"], state["downlink_bps"])
            )
            block = row["inputs"]["block"]
            block_residuals[block].append(correction)
            block_rtt[block] = state["rtt_ms"]
        blocks = sorted(block_residuals)
        values = [float(median(block_residuals[b])) for b in blocks]
        beta0, beta_rtt = nonnegative_fit(
            [[1.0, block_rtt[b]] for b in blocks], values, alpha=alpha
        )
        network.update(
            intercept_ms=beta0,
            rtt_coefficient=beta_rtt,
            independent_network_blocks=len(blocks),
            signed_training_serialization_corrections_ms=dict(
                zip(blocks, values, strict=True)
            ),
        )
    if variant == "v1_recalibrated":
        # Reuse V1's actual fitting algorithm, with only permitted training rows
        # and corrected pre-execution probe snapshots. No old coefficients.
        legacy_rows = [
            {
                **r["target"],
                "network_state": {
                    "controlled": True,
                    "bucket_id": r["inputs"]["block"],
                    "source": policy["network_source"],
                    **{
                        k: r["inputs"]["network"][k]
                        for k in ("uplink_bps", "downlink_bps", "rtt_ms")
                    },
                },
            }
            for r in good
        ]
        legacy = _candidate_fit(
            legacy_rows, minimum_rows=3, headroom=policy["memory_headroom_fraction"]
        )
        components = {
            k: _constant(v)
            for k, v in {
                "edge_compute_ms": legacy.edge_compute_ms,
                "client_preprocess_ms": legacy.client_preprocess_ms,
                "serialization_ms": legacy.serialization_ms,
                "server_fixed_ms": legacy.server_fixed_ms,
                "client_residual_ms": legacy.client_residual_ms,
            }.items()
        }
        network.update(
            intercept_ms=legacy.network.intercept_ms,
            rtt_coefficient=legacy.network.rtt_coefficient,
        )
    fit["components"] = components
    fit["network"] = network
    return fit


def fit_models(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    contract: dict[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, CalibrationV2]:
    allowed = set(split["training_request_ids"])
    keys = set()
    for row in rows:
        before, target = row["inputs"], row["target"]
        if (
            before["request_id"] not in allowed
            or before["block"] not in BLOCKS["train"]
        ):
            raise ValueError("forbidden request/block in fit")
        for identity in (
            "host_pair_id",
            "model_contract_sha256",
            "dataset_manifest_sha256",
        ):
            if target.get(identity) != contract[identity]:
                raise ValueError("mixed deployment/model/dataset in fit")
        if (
            target.get("configuration_id") != before["configuration_id"]
            or canonical_sha256(target["placement_profile"])
            != contract["placement_profile_sha256"][before["configuration_id"]]
        ):
            raise ValueError("fit placement identity mismatch")
        if (
            target["request_id"] != before["request_id"]
            or target["network_state"]["bucket_id"] != before["block"]
            or target["warmup"] is not False
        ):
            raise ValueError("target/input trial alignment mismatch")
        key = (
            before["request_id"],
            before["block"],
            before["configuration_id"],
            before["repetition"],
        )
        if key in keys:
            raise ValueError("duplicate fit key")
        keys.add(key)
    if require_complete and keys != {
        (rid, block, c.id.value, rep)
        for rid in allowed
        for block in BLOCKS["train"]
        for c in CONFIGURATIONS
        for rep in range(3)
    }:
        raise ValueError("incomplete fit schedule")
    policy = policy_for_fit(contract)
    source_hash = canonical_sha256(rows)
    fitted = {}
    for variant in VARIANTS:
        candidates = {
            spec.id.value: _candidate(
                [r for r in rows if r["inputs"]["configuration_id"] == spec.id.value],
                variant,
                policy,
            )
            for spec in CONFIGURATIONS
        }
        fitted[variant] = CalibrationV2.create(
            variant=variant,
            training_rows_sha256=source_hash,
            split_sha256=contract["frozen_split_sha256"],
            policy=policy,
            candidates=candidates,
            request_ids=tuple(sorted({r["inputs"]["request_id"] for r in rows})),
            block_ids=tuple(sorted({r["inputs"]["block"] for r in rows})),
        )
    return fitted


def _predict_rows(
    calibrations: dict[str, CalibrationV2], rows: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    predictions = []
    for variant, calibration in calibrations.items():
        predictor = PredictorV2(calibration)
        for row in rows:
            before, target = row["inputs"], row["target"]
            spec = next(
                s for s in CONFIGURATIONS if s.id.value == before["configuration_id"]
            )
            task = PredictionTask(**before["task"])
            prediction = predictor.predict(
                spec, state_from_snapshot(before, calibration.policy), task
            )
            predictions.append(
                {
                    "variant": variant,
                    "diagnostic": label,
                    "request_id": before["request_id"],
                    "block": before["block"],
                    "configuration_id": spec.id.value,
                    "repetition": before["repetition"],
                    "status": target["status"],
                    "observed_ms": target["observed_end_to_end_ms"],
                    "prediction": prediction.to_dict(),
                }
            )
    return predictions


def train_bundle(bundle: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite a fit")
    rows = load_training(bundle)
    split = strict_json((bundle / "split.json").read_bytes())
    contract = strict_json((bundle / "contract.json").read_bytes())
    fitted = fit_models(rows, split, contract)
    ids = sorted(split["training_request_ids"])
    oof = []
    for fold in range(contract["grouped_request_folds"]):
        held = {
            rid
            for i, rid in enumerate(ids)
            if i % contract["grouped_request_folds"] == fold
        }
        keep = [r for r in rows if r["inputs"]["request_id"] not in held]
        observed = [r for r in rows if r["inputs"]["request_id"] in held]
        models = fit_models(keep, split, contract, require_complete=False)
        oof.extend(_predict_rows(models, observed, f"grouped_request_fold_{fold}"))
    lobo = []
    for block in sorted(BLOCKS["train"]):
        models = fit_models(
            [r for r in rows if r["inputs"]["block"] != block],
            split,
            contract,
            require_complete=False,
        )
        lobo.extend(
            _predict_rows(
                models,
                [r for r in rows if r["inputs"]["block"] == block],
                f"leave_training_block_{block}_out",
            )
        )
    for variant, calibration in list(fitted.items()):
        candidates = {}
        for cid, candidate in calibration.candidates.items():
            errors: dict[str, list[float]] = defaultdict(list)
            for row in oof:
                if (
                    row["variant"] == variant
                    and row["configuration_id"] == cid
                    and row["status"] == "ok"
                    and row["prediction"]["total_ms"] is not None
                ):
                    errors[row["request_id"]].append(
                        abs(row["prediction"]["total_ms"] - row["observed_ms"])
                    )
            if not errors:
                raise ValueError("no out-of-fold residuals for uncertainty")
            width = percentile(
                [max(v) for v in errors.values()], contract["interval_quantile"]
            )
            candidates[cid] = {
                **candidate,
                "interval_half_width_ms": width,
                "uncertainty_independent_requests": len(errors),
                "uncertainty_source": (
                    "training grouped-OOF request max absolute error, "
                    "not a probability guarantee"
                ),
            }
        fitted[variant] = CalibrationV2.create(
            variant=variant,
            training_rows_sha256=calibration.training_rows_sha256,
            split_sha256=calibration.split_sha256,
            policy=calibration.policy,
            candidates=candidates,
            request_ids=calibration.training_request_ids,
            block_ids=calibration.training_block_ids,
        )
    output.mkdir(parents=True)
    for variant, calibration in fitted.items():
        write_json_atomic(output / f"calibration_{variant}.json", calibration.to_dict())
    for name, records in [("training_oof.jsonl", oof), ("training_lobo.jsonl", lobo)]:
        (output / name).write_text(
            "".join(
                json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records
            )
        )
    diagnostics = {
        "schema_version": 2,
        "training_attempts": len(rows),
        "unique_training_requests": len(ids),
        "network_blocks": sorted(BLOCKS["train"]),
        "status_counts": dict(Counter(r["target"]["status"] for r in rows)),
        "oof_predictions": len(oof),
        "lobo_predictions": len(lobo),
        "interval_method": (
            "95th percentile request-max grouped OOF absolute residual; "
            "twice width on extrapolation"
        ),
        "models": {v: m.calibration_id for v, m in fitted.items()},
        "validation_outcomes_read": False,
        "final_outcomes_read": False,
        "optional_v2_b": "not a declared candidate in this revision",
    }
    write_json_atomic(output / "training_diagnostics.json", diagnostics)
    return diagnostics
