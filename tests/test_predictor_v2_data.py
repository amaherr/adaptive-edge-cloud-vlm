from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.benchmark.predictor_v2_data import (
    FrozenSplitV2,
    _probe_snapshot,
    routing_metadata,
    strict_json,
    validate_target,
    validate_task_features,
)

ROOT = Path(__file__).resolve().parents[1]


def split() -> FrozenSplitV2:
    return FrozenSplitV2.load(
        ROOT / "configs/evaluation/physical_vqa_split_v1.json",
        ROOT / "data/manifests/physical_vqa_v1.jsonl",
    )


def task() -> dict[str, Any]:
    return {
        "task_bucket": "vqa-short-answer-32-token-budget",
        "image_bytes": 10,
        "image_width": 4,
        "image_height": 4,
        "prompt_tokens": 7,
        "requested_max_new_tokens": 32,
    }


def test_exact_frozen_cartesian_partitions() -> None:
    frozen = split()
    ids = [
        rid
        for key in (
            "training_request_ids",
            "validation_request_ids",
            "evaluation_request_ids",
        )
        for rid in frozen.value[key]
    ]
    counts = Counter(
        frozen.partition(rid, f"N{n}")
        for rid in ids
        for n in range(1, 10)
        for _ in range(9)
    )
    assert counts == {
        "train": 2700,
        "validation": 360,
        "final_test_protected": 360,
        "excluded": 4680,
    }
    with pytest.raises(ValueError, match="unknown request"):
        frozen.partition("missing", "N1")
    with pytest.raises(ValueError, match="unknown network"):
        frozen.partition(ids[0], "N10")


def test_frozen_split_rejects_membership_drift(tmp_path: Path) -> None:
    frozen = split()
    frozen.value["training_request_ids"][0] = frozen.value["validation_request_ids"][0]
    path = tmp_path / "split.json"
    path.write_text(json.dumps(frozen.value))
    with pytest.raises(ValueError, match="overlap"):
        FrozenSplitV2.load(path, ROOT / "data/manifests/physical_vqa_v1.jsonl")


def test_router_skips_outcomes_and_nested_decoys() -> None:
    line = json.dumps(
        {
            "fields": {
                "request_id": "decoy",
                "decode_ms": 1e200,
                "nested": ['}\\"', {"x": 1}],
            },
            "request_id": "allowed",
            "observed_end_to_end_ms": 1e300,
            "status": "oom",
            "network_state": {"bucket_id": "N1"},
            "raw_metrics": {"actual_output_tokens": 32},
            "repetition": 0,
        }
    ).encode()
    result = routing_metadata(line)
    assert result == {
        "request_id": "allowed",
        "network_state": {"bucket_id": "N1"},
        "repetition": 0,
    }
    changed = line.replace(b"1e+300", b"9e+300").replace(b"1e+200", b"7e+200")
    assert routing_metadata(changed) == result
    with pytest.raises(ValueError, match="duplicate"):
        routing_metadata(b'{"request_id":"one","request_id":"two"}')
    with pytest.raises(ValueError, match="duplicate"):
        strict_json('{"x":1,"x":2}')


@pytest.mark.parametrize(
    "forbidden",
    [
        "actual_output_tokens",
        "observed_end_to_end_ms",
        "decode_ms",
        "response_wire_bytes",
        "output_token_hash",
        "status",
    ],
)
def test_features_reject_outcomes(forbidden: str) -> None:
    snapshot = task()
    snapshot[forbidden] = 1
    with pytest.raises(ValueError, match="forbidden"):
        validate_task_features(snapshot)


def test_exact_preflight_probe_not_post_or_nominal() -> None:
    probe = {
        "collected_at_utc": "2026-11-10T14:00:00+00:00",
        "edge_to_cloud_receiver_bps": 1_900_000,
        "cloud_to_edge_receiver_bps": 1_800_000,
        "echo_median_rtt_ms": 23,
    }
    contract = {
        "max_controlled_block_probe_age_seconds": 14400,
        "network_source": "physical_wsl_netem_measured",
    }
    value = _probe_snapshot(probe, "2026-11-10T14:00:10Z", contract, "N1")
    assert value["uplink_bps"] == 1_900_000
    assert value["probe_age_seconds"] == 10
    assert value["probe_scope"] == "controlled_block_preflight"
    with pytest.raises(ValueError, match="future-dated"):
        _probe_snapshot(probe, "2026-11-10T13:59:59Z", contract, "N1")
    with pytest.raises(ValueError, match="envelope"):
        _probe_snapshot(probe, "2026-11-10T19:00:00Z", contract, "N1")


def test_targets_keep_failures_and_reject_negative_accounting() -> None:
    validate_target({"status": "oom", "failure_reason": "CUDA out of memory"}, task())
    with pytest.raises(ValueError, match="failure_reason"):
        validate_target({"status": "oom"}, task())
    row = {
        "status": "ok",
        "observed_end_to_end_ms": 12,
        "configuration_id": "edge/full",
        "fields": {
            **{
                k: v
                for k, v in task().items()
                if k not in {"task_bucket", "requested_max_new_tokens"}
            },
            "decode_ms": 1,
            "actual_output_tokens": 1,
            "stage_residual_ms": 2,
            "edge_total_ms": 10,
        },
    }
    validate_target(row, task())
    row["fields"]["stage_residual_ms"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        validate_target(row, task())


def test_real_archive_isolation_never_opens_final_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tarfile

    from adaptive_vlm.benchmark.predictor_v2_data import (
        load_training,
        prepare_partitions,
    )

    opened: list[str] = []
    original = tarfile.TarFile.extractfile

    def checked_extract(self: tarfile.TarFile, member: Any) -> Any:
        name = member if isinstance(member, str) else member.name
        if any(f"/{b}/" in name for b in ("N4", "N9")) and not name.endswith(
            "/plan.json"
        ):
            raise AssertionError("protected final outcome was opened")
        opened.append(name)
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", checked_extract)
    output = tmp_path / "partitions"
    audit = prepare_partitions(
        archive=ROOT / "results/physical-g2-20261110/evidence.tar.gz",
        inventory_path=ROOT / "results/physical-g2-20261110/archive_inventory.json",
        split_path=ROOT / "configs/evaluation/physical_vqa_split_v1.json",
        manifest_path=ROOT / "data/manifests/physical_vqa_v1.jsonl",
        feature_path=ROOT / "configs/predictor/physical_vqa_v2_input_features.json",
        contract_path=ROOT / "configs/predictor/physical_vqa_v2.json",
        output=output,
    )
    assert audit["classification"] == {
        "train": 2700,
        "validation": 360,
        "excluded": 4680,
        "final_test_protected": 360,
    }
    protected = Path(audit["sealed_validation_path"])
    assert protected.stat().st_mode & 0o777 == 0
    with pytest.raises(PermissionError):
        protected.read_bytes()
    assert len(load_training(output)) == 2700
    assert not any("network_post" in name for name in opened)
    assert audit["final_profiles_opened"] == []
    assert audit["training_status_counts"] == {"ok": 2700}
    assert not list(output.glob("*final*"))
    assert {
        name.split("/")[-3] for name in opened if name.endswith("profile/profile.jsonl")
    } == {"N1", "N2", "N3", "N5", "N6", "N7", "N8"}
