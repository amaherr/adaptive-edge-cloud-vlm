"""Training-only internal review of V2 cross-validation forecasts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.prediction.v1.calibration import canonical_sha256
from adaptive_vlm.prediction.v2.dataset.contract import (
    BLOCKS,
)
from adaptive_vlm.prediction.v2.dataset.parsing import digest, strict_json
from adaptive_vlm.prediction.v2.evaluation.scoring import scores
from adaptive_vlm.prediction.v2.evaluation.sealing import trial_key
from adaptive_vlm.prediction.v2.predictor import (
    VARIANTS,
)
from adaptive_vlm.scheduler.configuration import CONFIGURATIONS


def review_training(bundle: Path, fit_dir: Path, output: Path) -> dict[str, Any]:
    """Training-only residual/cross-validation review; does not alter any fit."""
    import csv

    from adaptive_vlm.prediction.v2.dataset.partitions import load_training

    if output.exists():
        raise FileExistsError("refusing to overwrite training review")
    training = load_training(bundle)
    split = strict_json((bundle / "split.json").read_bytes())
    lookup = {trial_key(r["inputs"]): r for r in training}
    receipts = strict_json((fit_dir / "file_hashes.json").read_bytes())
    reports = {}
    residuals = []
    manifests = []
    for filename in ("training_oof.jsonl", "training_lobo.jsonl"):
        path = fit_dir / filename
        if digest(path) != receipts[filename]:
            raise ValueError("training diagnostic hash changed")
        rows = [strict_json(line) for line in path.read_bytes().splitlines()]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                row["request_id"],
                row["block"],
                row["configuration_id"],
                row["repetition"],
            )
            source = lookup[key]
            groups[row["diagnostic"]].append(row)
            if row["prediction"]["total_ms"] is None or row["status"] != "ok":
                continue
            before = source["inputs"]
            residuals.append(
                {
                    "diagnostic": filename,
                    "fold": row["diagnostic"],
                    "variant": row["variant"],
                    "request_id": row["request_id"],
                    "block": row["block"],
                    "configuration_id": row["configuration_id"],
                    "repetition": row["repetition"],
                    "predicted_ms": row["prediction"]["total_ms"],
                    "observed_ms": row["observed_ms"],
                    "signed_error_ms": row["prediction"]["total_ms"]
                    - row["observed_ms"],
                    "uplink_bps": before["network"]["uplink_bps"],
                    "rtt_ms": before["network"]["rtt_ms"],
                    "image_bytes": before["task"]["image_bytes"],
                    "image_pixels": before["task"]["image_width"]
                    * before["task"]["image_height"],
                    "prompt_tokens": before["task"]["prompt_tokens"],
                    "predicted_expected_output_tokens": row["prediction"][
                        "estimated_output_tokens"
                    ],
                    "actual_output_tokens_training_diagnostic_only": source["target"][
                        "fields"
                    ]["actual_output_tokens"],
                }
            )
        by_model: dict[str, Any] = {}
        for variant in VARIANTS:
            by_model[variant] = {}
            for spec in CONFIGURATIONS:
                chosen = [
                    r
                    for r in rows
                    if r["variant"] == variant
                    and r["configuration_id"] == spec.id.value
                ]
                by_model[variant][spec.id.value] = {
                    "valid_only": scores(chosen),
                    "all_numeric_diagnostics": scores(chosen, diagnostic=True),
                    "by_block": {
                        b: scores(
                            [r for r in chosen if r["block"] == b], diagnostic=True
                        )
                        for b in sorted(BLOCKS["train"])
                    },
                }
        reports[filename] = by_model
        ids = sorted(split["training_request_ids"])
        for label, observed in groups.items():
            if label.startswith("grouped_request_fold_"):
                fold = int(label.rsplit("_", 1)[1])
                held_ids = {rid for i, rid in enumerate(ids) if i % 5 == fold}
                kept = [
                    r for r in training if r["inputs"]["request_id"] not in held_ids
                ]
                assert {r["request_id"] for r in observed} == held_ids
                held_blocks = []
            else:
                held_block = label.removeprefix("leave_training_block_").removesuffix(
                    "_out"
                )
                held_blocks = [held_block]
                held_ids = set()
                kept = [r for r in training if r["inputs"]["block"] != held_block]
                assert {r["block"] for r in observed} == {held_block}
            manifests.append(
                {
                    "label": label,
                    "training_requests": sorted(
                        {r["inputs"]["request_id"] for r in kept}
                    ),
                    "held_training_request_ids": sorted(held_ids),
                    "training_blocks": sorted({r["inputs"]["block"] for r in kept}),
                    "held_training_blocks": held_blocks,
                    "training_rows_sha256": canonical_sha256(kept),
                    "fold_calibration_ids": {
                        v: sorted(
                            {
                                r["prediction"]["calibration_id"]
                                for r in observed
                                if r["variant"] == v
                            }
                        )
                        for v in VARIANTS
                    },
                }
            )
    output.mkdir(parents=True)
    write_json_atomic(
        output / "training_internal_review.json",
        {
            "schema_version": 2,
            "reports": reports,
            "independent_requests": 60,
            "training_network_blocks": 5,
            "validation_outcomes_read": False,
            "final_outcomes_read": False,
            "uncertainty_note": (
                "OOF/LOBO point predictions precede interval fitting; their "
                "zero-width interval fields are not calibrated coverage."
            ),
        },
    )
    write_json_atomic(output / "training_cv_manifests.json", {"folds": manifests})
    if residuals:
        with (output / "training_residuals.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(residuals[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(residuals)
    return {
        "training_requests": 60,
        "training_network_blocks": 5,
        "cross_validation_manifests": len(manifests),
        "residual_diagnostics": len(residuals),
        "fits_modified": False,
        "validation_outcomes_read": False,
    }
