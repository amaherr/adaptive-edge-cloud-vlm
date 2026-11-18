"""Predictor V2 data from a G2 v2 campaign (protocol sections 5-9).

G2 v1 crossed every request ID with every network block; G2 v2 pairs them:
each network condition belongs to one split and is measured only on that
split's request IDs. This module

1. profiles each valid sub-block (``profile_campaign``) with ``build_profile``;
2. builds isolated partitions (``prepare_campaign_partitions``): training
   rows with targets, validation inputs with targets sealed under ``000``
   permissions, and evaluation sub-blocks only counted from their plans,
   never opened.

A profile row whose request ID belongs to a different split than its network
condition is rejected ("mixed-split row"): that can only come from a corrupt
or mislabelled campaign, and accepting it would leak data across splits.

Input snapshots extend schema 2 with the edge-load level and the edge state
read before the timer started (``pre_request_state``). Cloud ``server_state_*``
values arrive with the response, so they are outcomes and never inputs of
that request. G2 v2+ (amendment A2a rule 9) adds the cloud-load level and the
previous remote response's cloud state in the same sub-block
(``previous_cloud_state``, null for the first), which the runner recorded
before the timer started.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.profile import build_profile
from adaptive_vlm.campaign.grid import PRIMARY_TASK, Grid, SubBlock, load_grid
from adaptive_vlm.campaign.subblock import check_completed
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.prediction.v2.dataset.contract import (
    validate_target,
    validate_task_features,
)
from adaptive_vlm.prediction.v2.dataset.features import (
    fake_task_features,
    pinned_task_features,
)
from adaptive_vlm.prediction.v2.dataset.parsing import (
    ROUTING_KEYS,
    digest,
    number,
    routing_metadata,
    strict_json,
)
from adaptive_vlm.scheduler.configuration import SPEC_BY_PLACEMENT

PARTITIONS = {
    "training": "train",
    "validation": "validation",
    "evaluation": "final_test_protected",
}
INT8 = "distributed/projected/int8"
CONFIGURATION_BY_MODE = {
    "edge": "edge/full",
    "cloud": "cloud/raw_image",
    "distributed": "distributed/projected/synchronous",
}
EDGE_STATE_KEYS = (
    "edge_gpu_temperature_c",
    "edge_gpu_power_w",
    "edge_gpu_clock_graphics_mhz",
    "edge_gpu_clock_sm_mhz",
    "edge_gpu_clock_memory_mhz",
    "edge_gpu_utilization_percent",
    "edge_gpu_memory_utilization_percent",
    "edge_gpu_memory_used_mib",
    "edge_gpu_memory_free_mib",
    "edge_gpu_performance_state",
    "edge_gpu_clock_event_reasons",
    "edge_host_memory_available_mib",
    "edge_host_swap_used_mib",
    "edge_windows_gpu_dedicated_mib",
    "edge_windows_gpu_shared_mib",
    "edge_windows_gpu_sample_age_s",
    "edge_load_alive",
    "edge_load_duty",
    "edge_load_heartbeat_age_s",
)
# The AWS GRID vGPU reports no temperature, power or clock-event reasons, and
# an AWS edge has no Windows counters (phase D5): those inputs would always be
# null there, so they are not inputs on that platform.
_NOT_ON_AWS = {
    "edge_gpu_temperature_c",
    "edge_gpu_power_w",
    "edge_gpu_clock_event_reasons",
    "edge_windows_gpu_dedicated_mib",
    "edge_windows_gpu_shared_mib",
    "edge_windows_gpu_sample_age_s",
}
EDGE_STATE_KEYS_BY_PLATFORM = {
    "physical-wsl": EDGE_STATE_KEYS,
    "aws-linux": tuple(k for k in EDGE_STATE_KEYS if k not in _NOT_ON_AWS),
}
_INPUT_KEYS = ROUTING_KEYS | {
    "edge_load_level",
    "pre_request_state",
    "cloud_load_level",
    "previous_cloud_state",
}
PINNED_TOKENIZER = Path(
    ".model-cache/hub/models--moondream--starmie-v1/snapshots/"
    "35192e10a54e36eabe0a7cc57a2c1aab371cafc5/tokenizer.json"
)
PINNED_TEMPLATE = Path(".model-cache/audit/moondream2-9a7d4024/config.py")


def task_bucket(contract: dict[str, Any], grid: Grid, task: str) -> str:
    """The task bucket the grid declares, cross-checked with the contract."""

    bucket = grid.tasks[task].task_bucket
    allowed = contract.get("task_buckets") or {
        PRIMARY_TASK: contract["supported_task_bucket"]
    }
    if allowed.get(task) != bucket:
        raise ValueError(f"contract and grid disagree on the {task} task bucket")
    return bucket


def _reference_for(contract: dict[str, Any], task: str) -> str | None:
    value = contract.get("same_device_reference_evidence")
    if isinstance(value, dict):
        reference = value.get(task)
        return None if reference is None else str(reference)
    return str(value) if value and task == PRIMARY_TASK else None


def _campaign_grid(campaign: Path) -> tuple[dict[str, Any], Grid]:
    plan = strict_json((campaign / "campaign_plan.json").read_bytes())
    if plan.get("acceptance_drill"):
        raise ValueError("an acceptance-drill campaign is not campaign data")
    grid = load_grid(
        Path(plan["settings"]["grid"]),
        rate_fallback=bool(plan["settings"].get("rate_fallback")),
    )
    if grid.sha256 != plan["grid_sha256"]:
        raise ValueError("grid file changed since the campaign ran")
    if [b.subblock_id for b in grid.execution_order()] != plan["subblock_order"]:
        raise ValueError("grid order differs from the campaign plan")
    return plan, grid


def _valid_subblocks(
    campaign: Path, grid: Grid, splits: Iterable[str]
) -> list[tuple[SubBlock, Path]]:
    wanted = set(splits)
    result = []
    for block in grid.execution_order():
        if block.condition.split not in wanted:
            continue
        directory = campaign / "subblocks" / block.subblock_id
        report = check_completed(directory)
        if report is None or not report["valid"]:
            raise ValueError(f"sub-block is missing or invalid: {block.subblock_id}")
        result.append((block, directory))
    return result


def _network_state(
    block: SubBlock, directory: Path, spec_dir: Path, controlled: bool
) -> dict[str, Any]:
    if not controlled:
        return {
            "bucket_id": block.condition.condition_id,
            "source": "loopback_uncontrolled",
            "controlled": False,
            "uplink_bps": None,
            "downlink_bps": None,
            "rtt_ms": None,
        }
    evidence = directory / "network_pre.json"
    probe = strict_json(evidence.read_bytes())
    return {
        "bucket_id": block.condition.condition_id,
        "source": "controlled_netem_pre_subblock_probe",
        "controlled": True,
        "uplink_bps": probe["edge_to_cloud_receiver_bps"],
        "downlink_bps": probe["cloud_to_edge_receiver_bps"],
        "rtt_ms": probe["echo_median_rtt_ms"],
        "measurement_evidence_path": os.path.relpath(evidence, spec_dir),
        "measurement_evidence_sha256": digest(evidence),
    }


def profile_campaign(
    *,
    campaign: Path,
    contract_path: Path,
    output: Path,
    splits: Iterable[str] = ("training", "validation"),
) -> dict[str, Any]:
    """Profile every valid sub-block of the chosen splits into ``output``."""

    contract = strict_json(contract_path.read_bytes())
    plan, grid = _campaign_grid(campaign)
    controlled = plan["mode"] == "physical"
    if contract["require_controlled_network"] and not controlled:
        raise ValueError("this contract requires a physical (controlled) campaign")
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for block, directory in _valid_subblocks(campaign, grid, splits):
        target = output / block.subblock_id
        if (target / "profile" / "summary.json").exists():
            summaries[block.subblock_id] = strict_json(
                (target / "profile" / "summary.json").read_bytes()
            )
            continue
        target.mkdir(parents=True, exist_ok=True)
        spec: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": f"{grid.payload['campaign_id']}:{block.subblock_id}",
            "host_pair_id": contract["host_pair_id"],
            "cloud_device": contract["cloud_device"],
            "task_bucket": task_bucket(contract, grid, block.task),
            "requested_max_new_tokens": 32,
            "correctness_protocol": contract["correctness_protocol"],
            "network_state": _network_state(block, directory, target, controlled),
        }
        reference = _reference_for(contract, block.task)
        if reference:
            path = Path(reference).resolve(strict=True)
            spec["same_device_reference_evidence_path"] = os.path.relpath(path, target)
            spec["same_device_reference_evidence_sha256"] = digest(path)
        spec_path = target / "profile_specification.json"
        write_json_atomic(spec_path, spec)
        summaries[block.subblock_id] = build_profile(
            run_dirs=[directory / "run" / p for p in grid.placements],
            specification=spec_path,
            output_dir=target / "profile",
        )
    index = {
        "schema_version": 1,
        "campaign_plan_sha256": digest(campaign / "campaign_plan.json"),
        "contract_sha256": digest(contract_path),
        "profiled_splits": sorted(set(splits)),
        "subblocks": {
            sid: {
                "profile_sha256": digest(output / sid / "profile" / "profile.jsonl"),
                "row_count": summary["row_count"],
            }
            for sid, summary in summaries.items()
        },
    }
    write_json_atomic(output / "profiles_index.json", index)
    return index


def _cloud_state(state: Any) -> dict[str, Any] | None:
    """The previous response's cloud state: numeric/boolean fields, or None."""

    if not isinstance(state, dict):
        return None
    return {
        key: value
        for key, value in sorted(state.items())
        if value is None or isinstance(value, (bool, int, float))
    }


def edge_state_keys(plan: dict[str, Any]) -> tuple[str, ...]:
    """The edge-state inputs for the campaign's platform."""

    platform = plan["settings"].get("platform", "physical-wsl")
    if platform not in EDGE_STATE_KEYS_BY_PLATFORM:
        raise ValueError(f"no edge-state inputs defined for platform {platform}")
    return EDGE_STATE_KEYS_BY_PLATFORM[platform]


def _edge_state(state: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    source = state if isinstance(state, dict) else {}
    values: dict[str, float | bool | None] = {}
    for key in keys:
        value = source.get(key)
        if isinstance(value, bool) or value is None:
            values[key] = value
        else:
            values[key] = number(value, key)
    return values


def _probe_snapshot(
    network: dict[str, Any], probe: dict[str, Any] | None, started: str, max_age: float
) -> dict[str, Any]:
    if not network.get("controlled"):
        return {
            "network_bucket_id": network["bucket_id"],
            "network_source": network["source"],
            "uplink_bps": None,
            "downlink_bps": None,
            "rtt_ms": None,
            "probe_age_seconds": None,
            "probe_scope": "uncontrolled",
        }
    assert probe is not None
    start = datetime.fromisoformat(started.replace("Z", "+00:00"))
    measured = datetime.fromisoformat(probe["collected_at_utc"])
    age = (start - measured).total_seconds()
    if not 0 <= age <= max_age:
        raise ValueError("pre-sub-block probe is future-dated or too old")
    return {
        "network_bucket_id": network["bucket_id"],
        "network_source": network["source"],
        "uplink_bps": number(network["uplink_bps"], "uplink", positive=True),
        "downlink_bps": number(network["downlink_bps"], "downlink", positive=True),
        "rtt_ms": number(network["rtt_ms"], "rtt"),
        "probe_age_seconds": age,
        "probe_scope": "controlled_subblock_preflight",
    }


def prepare_campaign_partitions(
    *,
    campaign: Path,
    profiles: Path,
    contract_path: Path,
    feature_dir: Path,
    output: Path,
    sealed_root: Path = Path("/tmp"),
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite a partition bundle")
    contract = strict_json(contract_path.read_bytes())
    plan, grid = _campaign_grid(campaign)
    state_keys = edge_state_keys(plan)
    # A2a rules 4-5: the scheduler may see int8 only if its quality was admitted.
    admitted = contract.get("int8_admitted")
    if INT8 in grid.configurations and admitted is not True:
        raise ValueError(
            "the grid measures int8 but the dataset contract does not record "
            "int8_admitted: true (amendment A2a rule 4)"
        )
    splits: dict[str, dict[str, Any]] = {}
    features: dict[str, dict[str, Any]] = {}
    ids: dict[str, dict[str, list[str]]] = {}
    every: list[str] = []
    for name, task_spec in grid.tasks.items():
        split = strict_json(task_spec.split.read_bytes())
        if split["model_contract_sha256"] != contract["model_contract_sha256"]:
            raise ValueError(f"{name} split and dataset contract name different models")
        bundle = strict_json(feature_path(feature_dir, name).read_bytes())
        if (
            bundle["model_contract_sha256"] != contract["model_contract_sha256"]
            or bundle["dataset_manifest_sha256"] != split["dataset_manifest_sha256"]
            or bundle["feature_source"] != contract["feature_source"]
            or bundle["task_bucket"] != task_bucket(contract, grid, name)
        ):
            raise ValueError(f"{name} pre-execution feature provenance mismatch")
        ids[name] = {part: list(split[f"{part}_request_ids"]) for part in PARTITIONS}
        task_ids = [rid for group in ids[name].values() for rid in group]
        if set(bundle["tasks"]) != set(task_ids):
            raise ValueError(f"{name} task features must cover exactly its split's IDs")
        for value in bundle["tasks"].values():
            validate_task_features(value)
        every += task_ids
        splits[name], features[name] = split, bundle
    if len(every) != len(set(every)):
        raise ValueError("request partitions overlap")
    index = strict_json((profiles / "profiles_index.json").read_bytes())
    if index["campaign_plan_sha256"] != digest(campaign / "campaign_plan.json"):
        raise ValueError("profiles were built from a different campaign plan")

    classified: Counter[str] = Counter()
    training: list[dict[str, Any]] = []
    validation_inputs: list[dict[str, Any]] = []
    validation_targets: list[bytes] = []
    seen: set[tuple[str, str, str]] = set()
    opened: list[str] = []
    for block in grid.execution_order():
        split_name = block.condition.split
        partition = PARTITIONS[split_name]
        block_ids = ids[block.task][split_name]
        directory = campaign / "subblocks" / block.subblock_id
        report = check_completed(directory)
        if report is None or not report["valid"]:
            raise ValueError(f"sub-block is missing or invalid: {block.subblock_id}")
        schedule = strict_json((directory / "run" / "edge" / "plan.json").read_bytes())
        measured = [t for t in schedule["schedule"] if not t["warmup"]]
        if {t["request_id"] for t in measured} != set(block_ids):
            raise ValueError(f"{block.subblock_id} plan does not cover its split")
        classified[partition] += len(measured) * len(grid.placements)
        if partition == "final_test_protected":
            continue  # Never open an evaluation profile or journal.
        if digest(profiles / block.subblock_id / "profile" / "profile.jsonl") != index[
            "subblocks"
        ].get(block.subblock_id, {}).get("profile_sha256"):
            raise ValueError(f"profile changed or missing: {block.subblock_id}")
        spec = strict_json(
            (profiles / block.subblock_id / "profile_specification.json").read_bytes()
        )
        network = spec["network_state"]
        probe = (
            strict_json((directory / "network_pre.json").read_bytes())
            if network.get("controlled")
            else None
        )
        if (
            network.get("controlled")
            and digest(directory / "network_pre.json")
            != (network["measurement_evidence_sha256"])
        ):
            raise ValueError("probe evidence changed")
        plans = {
            SPEC_BY_PLACEMENT[placement].id.value: strict_json(
                (directory / "run" / placement / "plan.json").read_bytes()
            )
            for placement in grid.placements
        }
        member = profiles / block.subblock_id / "profile" / "profile.jsonl"
        opened.append(str(member.relative_to(profiles)))
        expected = {(rid, cid) for rid in block_ids for cid in grid.configurations}
        found: set[tuple[str, str]] = set()
        for line in member.read_bytes().splitlines():
            meta = routing_metadata(line, keys=_INPUT_KEYS)
            if meta.get("warmup") is True:
                continue
            request_id = meta["request_id"]
            if request_id not in block_ids:
                raise ValueError(
                    f"mixed-split row: {request_id} in {split_name} condition "
                    f"{block.condition.condition_id} ({block.task} task)"
                )
            cid = meta["configuration_id"]
            if cid not in plans:
                raise ValueError(f"unrecognized configuration: {cid}")
            _check_envelope(
                meta,
                contract,
                splits[block.task],
                plans[cid],
                block,
                task_bucket(contract, grid, block.task),
            )
            key = (request_id, block.subblock_id, cid)
            if key in seen:
                raise ValueError("duplicate measured trial key")
            seen.add(key)
            found.add((request_id, cid))
            task = features[block.task]["tasks"][request_id]
            snapshot = {
                "schema_version": 3,
                "request_id": request_id,
                "task_name": block.task,
                "subblock_id": block.subblock_id,
                "cell_id": block.cell_id,
                "condition_id": block.condition.condition_id,
                "split": split_name,
                "configuration_id": cid,
                "repetition": meta["repetition"],
                "trial_id": meta["trial_id"],
                "task": task,
                "network": _probe_snapshot(
                    network,
                    probe,
                    meta["measured_at_utc"],
                    float(contract["max_probe_age_seconds"]),
                ),
                "edge_load_level": meta["edge_load_level"],
                "edge_state": _edge_state(meta.get("pre_request_state"), state_keys),
                "cloud_load_level": meta.get("cloud_load_level", "C0"),
                "previous_cloud_state": _cloud_state(meta.get("previous_cloud_state")),
            }
            if snapshot["cloud_load_level"] != block.cloud_load:
                raise ValueError("profile cloud-load level differs from the sub-block")
            if meta["edge_load_level"] != block.load_level:
                raise ValueError("profile load level differs from the sub-block")
            if partition == "train":
                row = strict_json(line)
                validate_target(row, task)
                training.append({"inputs": snapshot, "target": row})
            else:
                validation_inputs.append(snapshot)
                validation_targets.append(line)
        if found != expected:
            raise ValueError(f"{block.subblock_id} profile does not cover its plan")

    expected_counts = {
        PARTITIONS[name]: sum(
            len(ids[b.task][name]) * len(grid.placements)
            for b in grid.subblocks()
            if b.condition.split == name
        )
        for name in PARTITIONS
    }
    if (
        dict(classified) != expected_counts
        or len(training) != expected_counts["train"]
        or len(validation_inputs) != expected_counts["validation"]
    ):
        raise ValueError("incomplete campaign or partition coverage")

    output.mkdir(parents=True)

    def save_lines(name: str, rows: list[dict[str, Any]]) -> None:
        (output / name).write_text(
            "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows)
        )

    save_lines("training.jsonl", training)
    save_lines("validation_inputs.jsonl", validation_inputs)
    sealed_root.mkdir(parents=True, exist_ok=True)
    sealed_dir = Path(tempfile.mkdtemp(prefix="predictor-v2-sealed-", dir=sealed_root))
    sealed = sealed_dir / "validation_targets.jsonl"
    payload = b"\n".join(validation_targets) + b"\n" if validation_targets else b""
    sealed.write_bytes(payload)
    sealed.chmod(0o000)
    if sealed.stat().st_mode & 0o777:
        raise RuntimeError("validation outcome permission seal was not enforced")
    receipt = {
        "schema_version": 3,
        "campaign_id": grid.payload["campaign_id"],
        "campaign_mode": plan["mode"],
        "campaign_plan_sha256": digest(campaign / "campaign_plan.json"),
        "grid_sha256": grid.sha256,
        "split_sha256": {n: digest(t.split) for n, t in grid.tasks.items()},
        "features_sha256": {
            n: digest(feature_path(feature_dir, n)) for n in grid.tasks
        },
        "contract_sha256": digest(contract_path),
        "configurations": list(grid.configurations),
        "int8_admitted": admitted,
        "cloud_load_level": grid.cloud_load_level,
        "classification": dict(classified),
        "training_attempts": len(training),
        "validation_attempts": len(validation_inputs),
        "source_profiles_opened": opened,
        "evaluation_profiles_opened": [],
        "sealed_validation_path": str(sealed),
        "training_status_counts": dict(
            Counter(r["target"]["status"] for r in training)
        ),
        "training_load_levels": dict(
            Counter(r["inputs"]["edge_load_level"] for r in training)
        ),
        "training_tasks": dict(Counter(r["inputs"]["task_name"] for r in training)),
        "files": {
            "training.jsonl": digest(output / "training.jsonl"),
            "validation_inputs.jsonl": digest(output / "validation_inputs.jsonl"),
            "validation_targets.jsonl": hashlib.sha256(payload).hexdigest(),
        },
    }
    write_json_atomic(output / "partition_audit.json", receipt)
    write_json_atomic(output / "contract.json", contract)
    for name, task_spec in grid.tasks.items():
        (output / f"split-{name}.json").write_bytes(task_spec.split.read_bytes())
    return receipt


def _check_envelope(
    meta: dict[str, Any],
    contract: dict[str, Any],
    split: dict[str, Any],
    plan: dict[str, Any],
    block: SubBlock,
    bucket: str,
) -> None:
    expected = {
        "schema_version": 1,
        "host_pair_id": contract["host_pair_id"],
        "model_contract_sha256": split["model_contract_sha256"],
        "dataset_manifest_sha256": split["dataset_manifest_sha256"],
        "requested_max_new_tokens": 32,
        "plan_sha256": plan["plan_sha256"],
        "task_bucket": bucket,
    }
    mismatched = [k for k, v in expected.items() if meta.get(k) != v]
    if mismatched:
        raise ValueError(f"profile envelope mismatch: {mismatched}")
    if contract["require_clean_source"] and meta.get("git_dirty") is not False:
        raise ValueError("profile rows come from a dirty source tree")
    if meta.get("network_state", {}).get("bucket_id") != block.condition.condition_id:
        raise ValueError("profile network bucket differs from the sub-block")
    scheduled = next(
        (t for t in plan["schedule"] if t["trial_id"] == meta.get("trial_id")), None
    )
    if scheduled is None or any(
        meta.get(k) != scheduled[k] for k in ("request_id", "warmup", "repetition")
    ):
        raise ValueError("profile row differs from the frozen schedule")
    if meta["repetition"] != block.repetition:
        raise ValueError("profile repetition differs from the sub-block")


def feature_path(feature_dir: Path, task: str) -> Path:
    return feature_dir / f"task_features-{task}.json"


def write_feature_bundle(
    campaign: Path,
    contract_path: Path,
    feature_dir: Path,
    *,
    tokenizer: Path = PINNED_TOKENIZER,
    template_source: Path = PINNED_TEMPLATE,
) -> list[Path]:
    """Pre-execution task features for every task's split, one file per task."""

    contract = strict_json(contract_path.read_bytes())
    _, grid = _campaign_grid(campaign)
    written = []
    for name, task_spec in grid.tasks.items():
        output = feature_path(feature_dir, name)
        if output.exists():
            raise FileExistsError(f"feature file exists: {output}")
        split = strict_json(task_spec.split.read_bytes())
        manifest = load_frozen_manifest(
            task_spec.split.parent / split["dataset_manifest_path"]
        )
        if manifest.sha256 != split["dataset_manifest_sha256"]:
            raise ValueError(f"{name} manifest differs from its split")
        bucket = task_bucket(contract, grid, name)
        payload: dict[str, Any] = {
            "schema_version": 3,
            "task": name,
            "task_bucket": bucket,
            "model_contract_sha256": contract["model_contract_sha256"],
            "dataset_manifest_sha256": manifest.sha256,
            "feature_source": contract["feature_source"],
        }
        if contract["feature_source"] == "pinned_tokenizer":
            if (
                digest(tokenizer) != contract["tokenizer_sha256"]
                or digest(template_source) != contract["template_source_sha256"]
            ):
                raise ValueError("pinned tokenizer/template SHA mismatch")
            tasks = pinned_task_features(
                manifest,
                tokenizer=tokenizer,
                template_source=template_source,
                bucket=bucket,
            )
            payload["tokenizer_sha256"] = digest(tokenizer)
        elif contract["feature_source"] == "fake_adapter":
            tasks = fake_task_features(manifest, bucket=bucket)
        else:
            raise ValueError(f"unknown feature source: {contract['feature_source']}")
        wanted = {
            rid
            for key in ("training", "validation", "evaluation")
            for rid in split[f"{key}_request_ids"]
        }
        payload["tasks"] = {rid: tasks[rid] for rid in sorted(wanted)}
        feature_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output, payload)
        written.append(output)
    return written
