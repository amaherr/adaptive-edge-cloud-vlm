from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from test_predictor_v2 import ROOT, synthetic_training

from adaptive_vlm.benchmark.predictor_v2_data import digest
from adaptive_vlm.benchmark.predictor_v2_diagnose import (
    correlation,
    crop_count,
    descriptive_features,
    diagnose,
    join_diagnostics,
    metrics,
    observed_components,
    repeat_changes,
)


def fixture_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, split, _policy = synthetic_training()
    rows = copy.deepcopy(
        [
            r
            for r in rows
            if r["inputs"]["block"] == "N1"
            and r["inputs"]["request_id"] in split["training_request_ids"][:3]
        ]
    )
    predictions = []
    for row in rows:
        row["inputs"]["trial_id"] = f"test-{row['inputs']['repetition']}"
        row["target"]["measured_at_utc"] = "2026-11-10T00:00:00Z"
        for key in [
            "edge_process_rss_mib",
            "cloud_process_rss_mib",
            "edge_vision_ms",
            "edge_projector_ms",
            "image_encode_and_prefill_ms",
            "image_prefill_ms",
            "prompt_prefill_ms",
            "prompt_prepare_ms",
            "deserialization_ms",
        ]:
            row["target"]["fields"][key] = None
        components = observed_components(row)
        components["decode_ms"] += 10
        components["client_residual_ms"] += 2
        predictions.append(
            {
                **{
                    k: row["inputs"][k]
                    for k in ["request_id", "block", "configuration_id", "repetition"]
                },
                "variant": "v2_a",
                "diagnostic": "synthetic-held-fold",
                "observed_ms": row["target"]["observed_end_to_end_ms"],
                "prediction": {
                    "components_ms": components,
                    "total_ms": sum(components.values()),
                    "valid": True,
                    "reason": None,
                },
            }
        )
    return rows, predictions


def test_component_closure_and_oracle_replacement() -> None:
    rows, predictions = fixture_rows()
    joined = join_diagnostics(rows, predictions)
    assert metrics(joined)["mae_ms"] == pytest.approx(12)
    assert metrics(joined, "decode_ms")["mae_ms"] == pytest.approx(2)
    assert metrics(joined)["independent_request_ids"] == 3
    for row in joined:
        assert sum(row["component_errors_ms"].values()) == pytest.approx(12)
    bad = copy.deepcopy(predictions)
    bad[0]["prediction"]["total_ms"] += 1
    with pytest.raises(ValueError, match="residual accounting"):
        join_diagnostics(rows, bad)
    rows[0]["target"]["observed_end_to_end_ms"] += 1
    with pytest.raises(ValueError, match="component accounting"):
        observed_components(rows[0])


@pytest.mark.parametrize("change", ["forbidden", "duplicate", "missing", "target"])
def test_bad_diagnostic_schedule_is_rejected(change: str) -> None:
    rows, predictions = fixture_rows()
    if change == "forbidden":
        predictions[0]["block"] = "N4"
    elif change == "duplicate":
        predictions.append(copy.deepcopy(predictions[0]))
    elif change == "missing":
        predictions.pop()
    else:
        predictions[0]["observed_ms"] += 1
    with pytest.raises(ValueError):
        join_diagnostics(rows, predictions)


def test_rank_ties_and_constant_features_are_not_fake_associations() -> None:
    assert correlation([1, 1, 3], [3, 3, 1]) == pytest.approx(-1)
    assert correlation([1, 1, 1], [1, 2, 3]) is None
    assert correlation([1, 2], [2, 1]) is None


def test_matched_block_and_repeat_differences_keep_requests_paired() -> None:
    rows, predictions = fixture_rows()
    joined = join_diagnostics(rows, predictions)
    cid = "cloud/raw_image"
    first = [r for r in joined if r["configuration_id"] == cid]
    reference = copy.deepcopy(first)
    for row in reference:
        row["block"] = "N3"
        row["observed_components_ms"]["decode_ms"] += 50
        row["observed_ms"] += 50
        row["predicted_ms"] += 50
    report = descriptive_features(first + reference)
    shift = report["matched_request_block_shift_relative_to_N3"]["N1"]["decode_ms"]
    assert shift["paired_requests"] == 3
    assert shift["median_shift_ms"] == -50
    for row in first:
        if row["repetition"] == 2:
            row["observed_components_ms"]["decode_ms"] += 3
            row["observed_ms"] += 3
    assert (
        repeat_changes(first)["observed_total_ms"]["median_third_minus_first_ms"] == 3
    )


def test_component_cancellation_does_not_imply_additive_oracle_gains() -> None:
    rows, predictions = fixture_rows()
    joined = join_diagnostics(rows, predictions)
    for row in joined:
        row["component_errors_ms"]["decode_ms"] = 10
        row["component_errors_ms"]["client_residual_ms"] = -9
        row["predicted_ms"] = row["observed_ms"] + 1
    assert metrics(joined)["mae_ms"] == 1
    assert metrics(joined, "decode_ms")["mae_ms"] == 9


def test_crop_geometry_boundary_and_maximum_tile_scaling() -> None:
    # Reference cases checked against the pinned model; no model cache in CI.
    for height, width, expected in [
        (375, 500, 2),
        (480, 640, 9),
        (378, 640, 2),
        (379, 640, 9),
        (10000, 10000, 10),
    ]:
        assert crop_count(height, width) == expected
    assert crop_count(378, 640) == 2
    assert crop_count(379, 640) > 2


def test_training_only_execution_tripwire_and_hash_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, predictions = fixture_rows()
    bundle, fit = tmp_path / "bundle", tmp_path / "fit"
    bundle.mkdir()
    fit.mkdir()
    (bundle / "training.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (bundle / "partition_audit.json").write_text(
        json.dumps({"files": {"training.jsonl": digest(bundle / "training.jsonl")}})
    )
    (bundle / "split.json").write_bytes(
        (ROOT / "configs/evaluation/physical_vqa_split_v1.json").read_bytes()
    )
    for name in ["training_oof.jsonl", "training_lobo.jsonl"]:
        (fit / name).write_text("".join(json.dumps(r) + "\n" for r in predictions))
    (fit / "file_hashes.json").write_text(
        json.dumps({p.name: digest(p) for p in fit.iterdir()})
    )
    original = Path.read_bytes
    allowed = {
        bundle / name
        for name in ["training.jsonl", "partition_audit.json", "split.json"]
    }
    allowed |= {
        fit / name
        for name in ["file_hashes.json", "training_oof.jsonl", "training_lobo.jsonl"]
    }

    def guarded(path: Path) -> bytes:
        assert path in allowed, f"unexpected outcome access: {path}"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    report = diagnose(bundle, fit, tmp_path / "report")
    assert report["training_attempts"] == 27
    assert report["validation_outcomes_read"] is False
    assert report["final_outcomes_read"] is False
    with pytest.raises(FileExistsError):
        diagnose(bundle, fit, tmp_path / "report")
    with (fit / "training_oof.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="prediction hash"):
        diagnose(bundle, fit, tmp_path / "tampered")
