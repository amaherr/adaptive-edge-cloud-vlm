"""Positive-allowlist physical G2 partitions; no final-outcome reader."""

from __future__ import annotations

import hashlib
import json
import math
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.calibrate import canonical_sha256
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS

BLOCKS = {
    "train": frozenset({"N1", "N3", "N5", "N6", "N8"}),
    "validation": frozenset({"N2", "N7"}),
    "final_test_protected": frozenset({"N4", "N9"}),
}
IDS_KEYS = {
    "train": "training_request_ids",
    "validation": "validation_request_ids",
    "final_test_protected": "evaluation_request_ids",
}
CONFIGURATION_IDS = frozenset(c.id.value for c in CONFIGURATIONS)
ROUTING_KEYS = frozenset(
    {
        "schema_version",
        "request_id",
        "configuration_id",
        "network_state",
        "warmup",
        "repetition",
        "trial_id",
        "measured_at_utc",
        "host_pair_id",
        "model_contract_sha256",
        "dataset_manifest_sha256",
        "placement_profile",
        "plan_sha256",
        "requested_max_new_tokens",
        "task_bucket",
        "git_commit",
        "git_dirty",
    }
)
TASK_KEYS = frozenset(
    {
        "task_bucket",
        "image_bytes",
        "image_width",
        "image_height",
        "prompt_tokens",
        "requested_max_new_tokens",
    }
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return result


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(data: bytes | str) -> dict[str, Any]:
    value = json.loads(data, object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _skip_value(text: str, position: int) -> int:
    """Skip opaque outcome values without decoding their numbers/objects."""
    start = position
    if text[position] in "{[":
        stack: list[str] = []
        quoted = escaped = False
        while position < len(text):
            char = text[position]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack or (stack.pop(), char) not in {("{", "}"), ("[", "]")}:
                    raise ValueError("malformed opaque JSON value")
                if not stack:
                    return position + 1
            position += 1
    elif text[position] == '"':
        position += 1
        escaped = False
        while position < len(text):
            char = text[position]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return position + 1
            position += 1
    else:
        while position < len(text) and text[position] not in ",} \t\r\n":
            position += 1
        if position > start:
            return position
    raise ValueError("unterminated JSON value")


def routing_metadata(line: bytes) -> dict[str, Any]:
    """Decode only allowlisted envelope keys, never fields/raw_metrics/outcomes."""
    text = line.decode("utf-8").strip()
    if not text.startswith("{"):
        raise ValueError("profile row must be an object")
    position = 1
    result: dict[str, Any] = {}
    seen: set[str] = set()
    decoder = json.JSONDecoder(object_pairs_hook=_pairs)
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            raise ValueError("unterminated profile envelope")
        if text[position] == "}":
            if text[position + 1 :].strip():
                raise ValueError("trailing profile bytes")
            return result
        key, position = decoder.raw_decode(text, position)
        if not isinstance(key, str) or key in seen:
            raise ValueError("invalid or duplicate envelope key")
        seen.add(key)
        while position < len(text) and text[position].isspace():
            position += 1
        if text[position] != ":":
            raise ValueError("missing JSON colon")
        position += 1
        while text[position].isspace():
            position += 1
        if key in ROUTING_KEYS:
            result[key], position = decoder.raw_decode(text, position)
        else:
            position = _skip_value(text, position)
        while position < len(text) and text[position].isspace():
            position += 1
        if text[position] == ",":
            position += 1
        elif text[position] != "}":
            raise ValueError("missing JSON delimiter")


@dataclass(frozen=True)
class FrozenSplitV2:
    value: dict[str, Any]
    sha256: str

    @classmethod
    def load(cls, path: Path, manifest: Path) -> FrozenSplitV2:
        value = strict_json(path.read_bytes())
        groups = [value.get(k) for k in IDS_KEYS.values()]
        if any(
            not isinstance(g, list) or len(g) != n or len(set(g)) != n
            for g, n in zip(groups, (60, 20, 20), strict=True)
        ):
            raise ValueError("frozen split must have exact unique 60/20/20 IDs")
        sets = [set(g) for g in groups if isinstance(g, list)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("request partitions overlap")
        entries = [strict_json(line) for line in manifest.read_bytes().splitlines()]
        if len(entries) != 100 or set.union(*sets) != {
            r["request_id"] for r in entries
        }:
            raise ValueError("split must exactly cover the frozen 100-ID manifest")
        if digest(manifest) != value.get("dataset_manifest_sha256"):
            raise ValueError("dataset manifest hash mismatch")
        for name, prefix in (
            ("train", "training"),
            ("validation", "validation"),
            ("final_test_protected", "evaluation"),
        ):
            actual = value.get(prefix + "_network_block_ids")
            if (
                not isinstance(actual, list)
                or len(actual) != len(set(actual))
                or set(actual) != BLOCKS[name]
            ):
                raise ValueError("frozen network membership mismatch")
        if value.get("max_new_tokens") != 32:
            raise ValueError("frozen generation budget must be 32")
        return cls(value, digest(path))

    def partition(self, request_id: str, block: str) -> str:
        if block not in set.union(*(set(b) for b in BLOCKS.values())):
            raise ValueError("unknown network block")
        if not any(request_id in self.value[k] for k in IDS_KEYS.values()):
            raise ValueError("unknown request ID")
        for name, key in IDS_KEYS.items():
            if request_id in self.value[key] and block in BLOCKS[name]:
                return name
        return "excluded"


def validate_task_features(value: dict[str, Any]) -> None:
    if set(value) != TASK_KEYS:
        raise ValueError("task snapshot contains missing or forbidden features")
    if (
        value["task_bucket"] != "vqa-short-answer-32-token-budget"
        or value["requested_max_new_tokens"] != 32
    ):
        raise ValueError("task/generation contract mismatch")
    for name in TASK_KEYS - {"task_bucket"}:
        n = number(value[name], name, positive=True)
        if n != int(n):
            raise ValueError("task sizes must be integral")


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


def validate_target(row: dict[str, Any], features: dict[str, Any]) -> None:
    if row.get("status") not in {
        "ok",
        "oom",
        "timeout",
        "protocol_error",
        "model_error",
    }:
        raise ValueError("unrecognized trial status")
    if row["status"] != "ok":
        if not row.get("failure_reason"):
            raise ValueError("failed measured attempt requires failure_reason")
        return
    observed = number(row.get("observed_end_to_end_ms"), "observed", positive=True)
    fields = row["fields"]
    for key in ("image_bytes", "image_width", "image_height", "prompt_tokens"):
        if fields.get(key) != features[key]:
            raise ValueError("profile/input feature mismatch")
    for key in ("decode_ms", "stage_residual_ms", "actual_output_tokens"):
        number(fields.get(key), key)
    if fields["actual_output_tokens"] > 32:
        raise ValueError("output exceeds frozen token budget")
    for key, value in fields.items():
        if key.endswith("_ms") and value is not None:
            number(value, key)
    mode = row["configuration_id"].split("/")[0]
    parts = (
        [fields["edge_total_ms"]]
        if mode == "edge"
        else [row["raw_metrics"]["image_read_ms"], fields["network_exchange_ms"]]
        if mode == "cloud"
        else [
            fields["edge_total_ms"],
            fields["serialization_ms"],
            fields["network_exchange_ms"],
        ]
    )
    if abs(sum(parts) + fields["stage_residual_ms"] - observed) > 1e-4:
        raise ValueError("stage accounting does not close")
    if mode != "edge":
        residual = fields["network_exchange_ms"] - fields["server_processing_ms"]
        if abs(residual - fields["transport_residual_ms"]) > 1e-4 or residual < 0:
            raise ValueError("invalid transport accounting")


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
