"""Turn validated fixed-policy benchmark attempts into versioned profile rows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any, cast

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.measurement import measure_call

_CONFIGURATIONS = {
    ("edge", "synchronous"): "edge/full",
    ("cloud", "synchronous"): "cloud/raw_image",
    ("distributed", "synchronous"): "distributed/projected/synchronous",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _metric(metrics: Mapping[str, Any], name: str, *, remote: bool) -> Any:
    return metrics.get(f"server_model_{name}" if remote else name)


def _read_run(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = _read_json(run_dir / "plan.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    saved_hash = plan.pop("plan_sha256", None)
    if saved_hash != _digest(plan):
        raise ValueError(f"plan hash mismatch: {run_dir}")
    plan["plan_sha256"] = saved_hash
    if manifest.get("config") != plan.get("config") or (
        manifest.get("dataset_manifest_sha256") != plan.get("dataset_manifest_sha256")
    ):
        raise ValueError(f"run manifest differs from plan: {run_dir}")
    expected = {trial["trial_id"]: trial for trial in plan["schedule"]}
    if len(expected) != len(plan["schedule"]):
        raise ValueError(f"duplicate scheduled trial: {run_dir}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        trial_id = row.get("trial_id")
        if trial_id not in expected or trial_id in seen:
            raise ValueError(f"unknown or duplicate trial {trial_id}: {run_dir}")
        trial = expected[trial_id]
        for name in ("request_id", "warmup", "repetition"):
            if row.get(name) != trial[name]:
                raise ValueError(f"trial {trial_id} differs on {name}: {run_dir}")
        for name in (
            "plan_sha256",
            "config_sha256",
            "model_contract_sha256",
            "dataset_manifest_sha256",
        ):
            if row.get(name) != plan[name]:
                raise ValueError(f"trial {trial_id} differs on {name}: {run_dir}")
        if (
            row.get("run_id") != manifest.get("run_id")
            or row.get("mode") != plan["mode"]
        ):
            raise ValueError(f"trial {trial_id} has wrong run or mode: {run_dir}")
        seen.add(trial_id)
        rows.append(row)
    if seen != set(expected):
        raise ValueError(
            f"run has {len(expected) - len(seen)} missing trials: {run_dir}"
        )
    return plan, manifest, rows


def build_profile(
    *, run_dirs: Sequence[Path], specification: Path, output_dir: Path
) -> dict[str, Any]:
    """Validate provenance and emit a profile with failures and coverage counts."""

    if output_dir.exists():
        raise FileExistsError(f"profile output already exists: {output_dir}")
    spec = _read_json(specification)
    if spec.get("schema_version") != 1 or not spec.get("campaign_id"):
        raise ValueError("profile specification requires version 1 and campaign_id")
    network = spec["network_state"]
    if not isinstance(network.get("controlled"), bool):
        raise ValueError("network_state.controlled must be boolean")
    if not network["controlled"] and any(
        network.get(name) is not None
        for name in ("uplink_bps", "downlink_bps", "rtt_ms")
    ):
        raise ValueError("uncontrolled network cannot claim measured bandwidth/RTT")
    if spec.get("requested_max_new_tokens") != 32:
        raise ValueError("Phase 2 requires pinned 32-token generation contract")
    if not run_dirs:
        raise ValueError("at least one benchmark run is required")

    runs = [_read_run(path) for path in run_dirs]
    reference = runs[0][0]
    modes: set[str] = set()
    for plan, _, _ in runs:
        if plan["mode"] in modes:
            raise ValueError(f"duplicate configuration mode: {plan['mode']}")
        modes.add(plan["mode"])
        for name in (
            "config_sha256",
            "model_contract_sha256",
            "dataset_manifest_sha256",
            "schedule",
        ):
            if plan[name] != reference[name]:
                raise ValueError(f"benchmark runs differ on {name}")
        if (plan["mode"], plan["pipeline"]) not in _CONFIGURATIONS:
            raise ValueError("unsupported configuration or pipeline")
    config = reference["config"]
    if config["model"]["max_new_tokens"] != spec["requested_max_new_tokens"]:
        raise ValueError("profile token budget differs from benchmark model contract")
    dataset = config["dataset"]
    frozen = load_frozen_manifest(
        Path(dataset["manifest_path"]), expected_revision=dataset["revision"]
    )
    if frozen.sha256 != reference["dataset_manifest_sha256"]:
        raise ValueError("frozen manifest hash differs from benchmark plan")
    items = {item.request_id: item for item in frozen.items}
    rows_out: list[dict[str, Any]] = []
    for plan, run_manifest, rows in runs:
        mode = plan["mode"]
        remote = mode != "edge"
        configuration_id = _CONFIGURATIONS[(mode, plan["pipeline"])]
        placement = run_manifest.get("environment", {}).get("execution_placement", {})
        for row in rows:
            item = items[row["request_id"]]
            if row.get("image_sha256") != item.image_sha256:
                raise ValueError(f"image hash mismatch: {row['trial_id']}")
            metrics = row.get("metrics", {})
            success = row["status"] == "ok"
            if success and not isinstance(metrics, dict):
                raise ValueError(f"missing metrics: {row['trial_id']}")
            required = (
                "prompt_tokens",
                "prompt_prefill_ms",
                "decode_ms",
                "output_tokens",
            )
            if success and config["model"]["adapter"] == "moondream2":
                for name in (*required, "image_width", "image_height"):
                    if _metric(metrics, name, remote=remote) is None:
                        raise ValueError(f"missing {name}: {row['trial_id']}")
            exchange_ms = metrics.get("network_round_trip_ms") if success else None
            server_ms = metrics.get("server_total_ms") if success else None
            transport_residual = (
                float(exchange_ms) - float(server_ms)
                if exchange_ms is not None and server_ms is not None
                else None
            )
            if success and remote and transport_residual is None:
                raise ValueError(f"missing remote timing: {row['trial_id']}")
            request_wire = metrics.get("request_wire_bytes") if success else None
            response_wire = metrics.get("response_wire_bytes") if success else None
            payload = (
                metrics.get("projected_payload_bytes")
                if mode == "distributed"
                else metrics.get("image_bytes")
                if mode == "cloud"
                else None
            )
            if (
                success
                and remote
                and (
                    not isinstance(request_wire, int)
                    or not isinstance(response_wire, int)
                    or not isinstance(payload, int)
                    or request_wire < payload
                    or response_wire <= 0
                )
            ):
                raise ValueError(f"invalid wire-byte accounting: {row['trial_id']}")
            if (
                success
                and mode == "cloud"
                and payload != item.image_path.stat().st_size
            ):
                raise ValueError(
                    f"raw image payload differs from manifest: {row['trial_id']}"
                )
            if (
                success
                and remote
                and metrics.get("total_wire_bytes") is not None
                and metrics["total_wire_bytes"]
                != cast(int, request_wire) + cast(int, response_wire)
            ):
                raise ValueError(
                    f"total wire-byte accounting differs: {row['trial_id']}"
                )
            if (
                success
                and mode == "distributed"
                and config["model"]["adapter"] == "moondream2"
                and payload != 729 * 2048 * 2
            ):
                raise ValueError(f"projected tensor payload differs: {row['trial_id']}")
            token_ids = row.get("output_token_ids", [])
            token_hash = (
                hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode()
                ).hexdigest()
                if success
                else None
            )
            stage_components = (
                [
                    metrics.get("edge_vision_end_to_end_ms"),
                    metrics.get("tensor_serialization_ms"),
                    exchange_ms,
                ]
                if mode == "distributed"
                else [metrics.get("image_read_ms"), exchange_ms]
                if mode == "cloud"
                else [metrics.get("end_to_end_ms")]
            )
            stage_sum = (
                sum(float(value) for value in stage_components)
                if success and all(value is not None for value in stage_components)
                else None
            )
            observed = float(row["observed_end_to_end_ms"])
            fields: dict[str, Any] = {
                "image_width": _metric(metrics, "image_width", remote=remote),
                "image_height": _metric(metrics, "image_height", remote=remote),
                "image_bytes": item.image_path.stat().st_size,
                "prompt_tokens": _metric(metrics, "prompt_tokens", remote=remote),
                "actual_output_tokens": len(token_ids) if success else None,
                "edge_vision_ms": metrics.get("vision_encoder_ms")
                if mode != "cloud"
                else None,
                "edge_projector_ms": metrics.get("projection_ms")
                if mode != "cloud"
                else None,
                "edge_total_ms": metrics.get("edge_vision_end_to_end_ms")
                if mode == "distributed"
                else metrics.get("end_to_end_ms")
                if mode == "edge"
                else None,
                "serialization_ms": metrics.get("tensor_serialization_ms"),
                "request_payload_bytes": payload,
                "request_wire_bytes": request_wire,
                "response_wire_bytes": response_wire,
                "network_exchange_ms": exchange_ms,
                "network_connect_ms": metrics.get("network_connect_ms"),
                "network_send_ms": metrics.get("network_send_ms"),
                "network_receive_ms": metrics.get("network_receive_ms"),
                "server_processing_ms": server_ms,
                "server_receive_queue_ms": None,
                "transport_residual_ms": transport_residual,
                "deserialization_ms": metrics.get(
                    "server_model_tensor_deserialization_ms"
                ),
                "image_prefill_ms": _metric(metrics, "image_prefill_ms", remote=remote),
                "image_encode_and_prefill_ms": _metric(
                    metrics, "image_encode_and_prefill_ms", remote=remote
                ),
                "prompt_prepare_ms": _metric(
                    metrics, "prompt_prepare_ms", remote=remote
                ),
                "prompt_prefill_ms": _metric(
                    metrics, "prompt_prefill_ms", remote=remote
                ),
                "decode_ms": _metric(metrics, "decode_ms", remote=remote),
                "edge_peak_allocated_mib": metrics.get("vision_cuda_peak_allocated_mib")
                if mode != "cloud"
                else None,
                "edge_peak_reserved_mib": metrics.get("vision_cuda_peak_reserved_mib")
                if mode != "cloud"
                else None,
                "edge_process_rss_mib": metrics.get("rss_after_mib")
                if mode != "cloud"
                else None,
                "cloud_peak_allocated_mib": metrics.get(
                    "server_model_text_cuda_peak_allocated_mib"
                ),
                "cloud_peak_reserved_mib": metrics.get(
                    "server_model_text_cuda_peak_reserved_mib"
                ),
                "cloud_process_rss_mib": metrics.get("server_model_rss_after_mib"),
                "stage_sum_ms": stage_sum,
                "scheduler_overhead_ms": None,
                "stage_residual_ms": observed - stage_sum
                if stage_sum is not None
                else None,
            }
            missing: dict[str, str] = {}
            for name, value in fields.items():
                if value is not None:
                    continue
                if not success:
                    reason = "failed_trial"
                elif name == "scheduler_overhead_ms":
                    reason = "fixed_policy_benchmark_no_scheduler"
                elif name == "server_receive_queue_ms":
                    reason = "sequential_server_queue_not_observable"
                elif "peak_" in name and (
                    (
                        name.startswith("edge_")
                        and placement.get("vision_device") == "cpu"
                    )
                    or (name.startswith("cloud_") and spec.get("cloud_device") == "cpu")
                ):
                    reason = "cpu_placement_no_cuda_peak"
                elif name.startswith("cloud_") and mode == "edge":
                    reason = "no_cloud_execution"
                elif name.startswith("edge_") and mode == "cloud":
                    reason = "no_edge_model_execution"
                elif (
                    name
                    in {
                        "serialization_ms",
                        "request_payload_bytes",
                        "request_wire_bytes",
                        "response_wire_bytes",
                        "network_exchange_ms",
                        "network_connect_ms",
                        "network_send_ms",
                        "network_receive_ms",
                        "server_processing_ms",
                        "transport_residual_ms",
                        "deserialization_ms",
                    }
                    and mode == "edge"
                ):
                    reason = "local_execution_no_transport"
                elif name == "deserialization_ms" and mode == "cloud":
                    reason = "raw_image_no_tensor_deserialization"
                elif name == "image_prefill_ms" and mode == "cloud":
                    reason = "monolithic_image_encode_and_prefill_combined"
                elif name == "image_encode_and_prefill_ms" and mode != "cloud":
                    reason = "decomposed_path_separate_vision_and_prefill"
                else:
                    reason = "metric_unavailable"
                missing[name] = reason
            rows_out.append(
                {
                    "schema_version": 1,
                    "campaign_id": spec["campaign_id"],
                    "measured_at_utc": row.get("started_at_utc"),
                    "trial_id": row["trial_id"],
                    "request_id": row["request_id"],
                    "configuration_id": configuration_id,
                    "run_id": row["run_id"],
                    "plan_sha256": row["plan_sha256"],
                    "config_sha256": row["config_sha256"],
                    "model_contract_sha256": row["model_contract_sha256"],
                    "dataset_manifest_sha256": row["dataset_manifest_sha256"],
                    "host_pair_id": spec["host_pair_id"],
                    "placement_profile": placement,
                    "network_state": network,
                    "task_bucket": spec["task_bucket"],
                    "task_type": item.scoring,
                    "requested_max_new_tokens": 32,
                    "warmup": row["warmup"],
                    "status": row["status"],
                    "failure_reason": row.get("error_message") if not success else None,
                    "output_token_hash": token_hash,
                    "quality_score": row.get("quality_score"),
                    "observed_end_to_end_ms": observed,
                    "fields": fields,
                    "null_reasons": missing,
                    "raw_metrics": metrics,
                }
            )
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    request_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows_out:
        if row["warmup"]:
            continue
        key = row["configuration_id"]
        counts[key][row["status"]] += 1
        if row["status"] == "ok":
            request_ids[key].add(row["request_id"])
    coverage = {
        key: {
            "status_counts": dict(statuses),
            "successful_request_ids": len(request_ids[key]),
            "covered": statuses["ok"] >= 10 and len(request_ids[key]) >= 3,
            "failure_rate": (
                sum(count for status, count in statuses.items() if status != "ok")
                / sum(statuses.values())
            ),
        }
        for key, statuses in counts.items()
    }
    summary = {
        "schema_version": 1,
        "campaign_id": spec["campaign_id"],
        "specification_sha256": hashlib.sha256(specification.read_bytes()).hexdigest(),
        "dataset_manifest_sha256": frozen.sha256,
        "model_contract_sha256": reference["model_contract_sha256"],
        "row_count": len(rows_out),
        "measurement_call_overhead_ms_median_cpu": median(
            measure_call(lambda: None).elapsed_ms for _ in range(1000)
        ),
        "warmup_rows": sum(bool(row["warmup"]) for row in rows_out),
        "coverage": coverage,
        "local_gate_passed": len(coverage) == 3
        and all(value["covered"] for value in coverage.values())
        and all(
            row["fields"]["stage_residual_ms"] is not None
            for row in rows_out
            if row["status"] == "ok"
        ),
        "controlled_network_gate_passed": bool(network["controlled"])
        and len(coverage) == 3
        and all(value["covered"] for value in coverage.values()),
        "negative_transport_residuals": sum(
            row["fields"]["transport_residual_ms"] is not None
            and row["fields"]["transport_residual_ms"] < 0
            for row in rows_out
        ),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "profile.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows_out),
        encoding="utf-8",
    )
    write_json_atomic(output_dir / "summary.json", summary)
    return summary
