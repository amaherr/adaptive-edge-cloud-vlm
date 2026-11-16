"""Frozen V2 split contract, task features and target validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_vlm.prediction.v2.dataset.parsing import digest, number, strict_json
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


# G2 v2 adds a timed-only long-answer bucket (protocol section 3.5).
TASK_BUCKETS = frozenset(
    {"vqa-short-answer-32-token-budget", "vqa-long-answer-32-token-budget"}
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
        value["task_bucket"] not in TASK_BUCKETS
        or value["requested_max_new_tokens"] != 32
    ):
        raise ValueError("task/generation contract mismatch")
    for name in TASK_KEYS - {"task_bucket"}:
        n = number(value[name], name, positive=True)
        if n != int(n):
            raise ValueError("task sizes must be integral")


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
