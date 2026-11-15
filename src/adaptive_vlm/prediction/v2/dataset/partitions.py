"""Isolate physical G2 rows into sealed training/validation partitions."""

from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.prediction.v1.calibration import canonical_sha256
from adaptive_vlm.prediction.v2.dataset.contract import (
    BLOCKS,
    CONFIGURATION_IDS,
    IDS_KEYS,
    FrozenSplitV2,
    validate_target,
    validate_task_features,
)
from adaptive_vlm.prediction.v2.dataset.parsing import (
    digest,
    number,
    routing_metadata,
    strict_json,
)


def _archive_read(
    tar: tarfile.TarFile, member: str, inventory: dict[str, Any]
) -> bytes:
    handle = tar.extractfile(member)
    if handle is None:
        raise ValueError(f"missing archive member: {member}")
    payload = handle.read()
    if (
        member not in inventory
        or hashlib.sha256(payload).hexdigest() != inventory[member]["sha256"]
    ):
        raise ValueError(f"archive member hash mismatch: {member}")
    return payload


def _probe_snapshot(
    probe: dict[str, Any], timestamp: str, contract: dict[str, Any], block: str
) -> dict[str, Any]:
    start = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    measured = datetime.fromisoformat(probe["collected_at_utc"])
    age = (start - measured).total_seconds()
    if not 0 <= age <= contract["max_controlled_block_probe_age_seconds"]:
        raise ValueError(
            "pre-block probe is future-dated or outside static-block envelope"
        )
    return {
        "network_bucket_id": block,
        "network_source": contract["network_source"],
        "uplink_bps": number(
            probe["edge_to_cloud_receiver_bps"], "uplink", positive=True
        ),
        "downlink_bps": number(
            probe["cloud_to_edge_receiver_bps"], "downlink", positive=True
        ),
        "rtt_ms": number(probe["echo_median_rtt_ms"], "rtt"),
        "probe_age_seconds": age,
        "probe_scope": "controlled_block_preflight",
    }


def _validate_envelope(
    row: dict[str, Any],
    contract: dict[str, Any],
    split: FrozenSplitV2,
    plan: dict[str, Any],
    configuration: str,
    block: str,
) -> None:
    expected = {
        "host_pair_id": contract["host_pair_id"],
        "model_contract_sha256": split.value["model_contract_sha256"],
        "dataset_manifest_sha256": split.value["dataset_manifest_sha256"],
        "requested_max_new_tokens": 32,
        "configuration_id": configuration,
        "plan_sha256": plan["plan_sha256"],
        "git_commit": contract["inference_git_commit"],
    }
    if row.get("schema_version") != 1 or row.get("git_dirty") is not False:
        raise ValueError("profile schema or clean source provenance mismatch")
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("profile host/model/dataset/configuration/source mismatch")
    if row.get("network_state", {}).get("bucket_id") != block:
        raise ValueError("profile network bucket mismatch")
    scheduled = next(
        (t for t in plan["schedule"] if t["trial_id"] == row.get("trial_id")), None
    )
    if scheduled is None or any(
        row.get(k) != scheduled[k] for k in ("request_id", "warmup", "repetition")
    ):
        raise ValueError("profile row differs from frozen schedule")
    if (
        canonical_sha256(row["placement_profile"])
        != contract["placement_profile_sha256"][configuration]
    ):
        raise ValueError("placement profile mismatch")


def prepare_partitions(
    *,
    archive: Path,
    inventory_path: Path,
    split_path: Path,
    manifest_path: Path,
    feature_path: Path,
    contract_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite a partition bundle")
    contract = strict_json(contract_path.read_bytes())
    if digest(archive) != contract["archive_sha256"]:
        raise ValueError("source archive hash mismatch")
    if contract.get("environment_kind") != "physical_wsl" or any(
        s in contract["host_pair_id"].lower() for s in ("aws", "vultr", "g6f")
    ):
        raise ValueError("only the approved physical pair is allowed")
    split = FrozenSplitV2.load(split_path, manifest_path)
    if split.sha256 != contract["frozen_split_sha256"]:
        raise ValueError("frozen split hash mismatch")
    feature_bundle = strict_json(feature_path.read_bytes())
    if (
        feature_bundle["model_contract_sha256"] != split.value["model_contract_sha256"]
        or feature_bundle["dataset_manifest_sha256"]
        != split.value["dataset_manifest_sha256"]
        or feature_bundle["tokenizer_sha256"] != contract["tokenizer_sha256"]
    ):
        raise ValueError("pre-execution feature provenance mismatch")
    tasks = feature_bundle["tasks"]
    if set(tasks) != set().union(*(set(split.value[k]) for k in IDS_KEYS.values())):
        raise ValueError("pre-execution features must cover exactly the frozen IDs")
    for feature in tasks.values():
        validate_task_features(feature)
    inventory = strict_json(inventory_path.read_bytes())
    prefix = contract["archive_prefix"]
    classified: Counter[str] = Counter()
    training: list[dict[str, Any]] = []
    validation_targets: list[bytes] = []
    validation_inputs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    opened_profiles: list[str] = []
    with tarfile.open(archive, "r:gz") as tar:
        plans: dict[tuple[str, str], dict[str, Any]] = {}

        def get_plan(block: str, mode: str) -> dict[str, Any]:
            key = (block, mode)
            if key not in plans:
                plans[key] = strict_json(
                    _archive_read(tar, f"{prefix}/{block}/{mode}/plan.json", inventory)
                )
            return plans[key]

        for i in range(1, 10):
            block = f"N{i}"
            for configuration in sorted(CONFIGURATION_IDS):
                mode = configuration.split("/")[0]
                plan = get_plan(block, mode)
                fingerprint = plan["plan_sha256"]
                if (
                    canonical_sha256(
                        {k: v for k, v in plan.items() if k != "plan_sha256"}
                    )
                    != fingerprint
                ):
                    raise ValueError("plan hash mismatch")
                config = plan["config"]
                model = config["model"]
                if (
                    model["precision"] != "bf16"
                    or model["max_new_tokens"] != 32
                    or model["revision"] != contract["model_revision"]
                    or model["artifact_sha256"] != contract["model_artifact_sha256"]
                    or canonical_sha256(model) != contract["model_config_sha256"]
                ):
                    raise ValueError("model identity/precision/budget drift")
                trials = [r for r in plan["schedule"] if not r["warmup"]]
                if (
                    len(plan["schedule"]) != 301
                    or len(trials) != 300
                    or len({r["trial_id"] for r in plan["schedule"]}) != 301
                    or {(r["request_id"], r["repetition"]) for r in trials}
                    != {(rid, rep) for rid in tasks for rep in range(3)}
                ):
                    raise ValueError("incomplete or duplicate planned schedule")
                for trial in trials:
                    classified[split.partition(trial["request_id"], block)] += 1
                if block in BLOCKS["final_test_protected"]:
                    continue  # Crucially, never extract a final-test profile member.
                probe = strict_json(
                    _archive_read(tar, f"{prefix}/{block}/network_pre.json", inventory)
                )
                member = f"{prefix}/{block}/profile/profile.jsonl"
                # Each shared profile is opened once, while mode plans are metadata.
                if member in opened_profiles:
                    continue
                opened_profiles.append(member)
                data = _archive_read(tar, member, inventory)
                for line in data.splitlines():
                    meta = routing_metadata(line)
                    if meta.get("warmup") is True:
                        continue
                    part = split.partition(meta["request_id"], block)
                    if part == "excluded":
                        continue  # Outcome fields were only skipped, never decoded.
                    cid = meta["configuration_id"]
                    if cid not in CONFIGURATION_IDS:
                        raise ValueError("unrecognized configuration")
                    selected_mode = cid.split("/")[0]
                    selected_plan = get_plan(block, selected_mode)
                    _validate_envelope(meta, contract, split, selected_plan, cid, block)
                    key = (meta["request_id"], block, cid, meta["repetition"])
                    if key in seen:
                        raise ValueError("duplicate measured trial key")
                    seen.add(key)
                    task = tasks[meta["request_id"]]
                    snapshot = {
                        "schema_version": 2,
                        "request_id": meta["request_id"],
                        "block": block,
                        "configuration_id": cid,
                        "repetition": meta["repetition"],
                        "trial_id": meta["trial_id"],
                        "task": task,
                        "network": _probe_snapshot(
                            probe, meta["measured_at_utc"], contract, block
                        ),
                    }
                    if part == "train":
                        row = strict_json(line)
                        validate_target(row, task)
                        training.append({"inputs": snapshot, "target": row})
                    elif part == "validation":
                        validation_inputs.append(snapshot)
                        validation_targets.append(line)
                    else:
                        raise ValueError("protected outcome reached partition decoder")
    if (
        dict(classified)
        != {
            "train": 2700,
            "validation": 360,
            "final_test_protected": 360,
            "excluded": 4680,
        }
        or len(training) != 2700
        or len(validation_inputs) != 360
    ):
        raise ValueError("incomplete campaign or partition coverage")
    output.mkdir(parents=True)

    def save_lines(name: str, rows: list[dict[str, Any]]) -> None:
        (output / name).write_text(
            "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows)
        )

    save_lines("training.jsonl", training)
    save_lines("validation_inputs.jsonl", validation_inputs)
    # DrvFS may report chmod success without removing read permissions. Use a
    # private Linux filesystem directory, and verify the actual permission bits.
    sealed_directory = Path(tempfile.mkdtemp(prefix="predictor-v2-sealed-", dir="/tmp"))
    sealed = sealed_directory / "validation_targets.jsonl"
    sealed.write_bytes(b"\n".join(validation_targets) + b"\n")
    sealed.chmod(0o000)
    if sealed.stat().st_mode & 0o777:
        raise RuntimeError("validation outcome permission seal was not enforced")
    receipt = {
        "schema_version": 2,
        "archive_sha256": contract["archive_sha256"],
        "split_sha256": split.sha256,
        "features_sha256": digest(feature_path),
        "contract_sha256": digest(contract_path),
        "classification": dict(classified),
        "training_attempts": len(training),
        "validation_attempts": len(validation_inputs),
        "source_profiles_opened": opened_profiles,
        "final_profiles_opened": [],
        "final_targets_materialized": False,
        "prior_final_inspection_disclosed": True,
        "sealed_validation_path": str(sealed),
        "validation_permissions": "000 on Linux /tmp; restore/repartition after reboot",
        "training_status_counts": dict(
            Counter(r["target"]["status"] for r in training)
        ),
        "files": {
            "training.jsonl": digest(output / "training.jsonl"),
            "validation_inputs.jsonl": digest(output / "validation_inputs.jsonl"),
            "validation_targets.jsonl": hashlib.sha256(
                b"\n".join(validation_targets) + b"\n"
            ).hexdigest(),
        },
    }
    write_json_atomic(output / "partition_audit.json", receipt)
    write_json_atomic(output / "contract.json", contract)
    (output / "split.json").write_bytes(split_path.read_bytes())
    return receipt


def load_training(bundle: Path) -> list[dict[str, Any]]:
    audit = strict_json((bundle / "partition_audit.json").read_bytes())
    path = bundle / "training.jsonl"
    if digest(path) != audit["files"][path.name]:
        raise ValueError("training partition hash mismatch")
    rows = [strict_json(line) for line in path.read_bytes().splitlines()]
    split = strict_json((bundle / "split.json").read_bytes())
    for row in rows:
        snapshot = row["inputs"]
        if (
            snapshot["request_id"] not in split["training_request_ids"]
            or snapshot["block"] not in BLOCKS["train"]
        ):
            raise ValueError("forbidden fit request/network row")
        validate_task_features(snapshot["task"])
        validate_target(row["target"], snapshot["task"])
    return rows
