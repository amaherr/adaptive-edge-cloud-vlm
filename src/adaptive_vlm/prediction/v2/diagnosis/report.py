"""Training-only stage-attribution report for saved V2 forecasts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.prediction.v2.dataset.parsing import digest, strict_json
from adaptive_vlm.prediction.v2.dataset.partitions import load_training
from adaptive_vlm.prediction.v2.diagnosis.analysis import (
    associations,
    descriptive_features,
    repeat_changes,
    summarize,
)
from adaptive_vlm.prediction.v2.diagnosis.components import join_diagnostics


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def diagnose(bundle: Path, fit: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    training = load_training(bundle)
    receipts = strict_json((fit / "file_hashes.json").read_bytes())
    reports: dict[str, Any] = {}
    residuals: list[dict[str, Any]] = []
    assoc_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    for filename in ["training_oof.jsonl", "training_lobo.jsonl"]:
        path = fit / filename
        if digest(path) != receipts[filename]:
            raise ValueError("diagnostic prediction hash mismatch")
        predictions = [strict_json(line) for line in path.read_bytes().splitlines()]
        joined = join_diagnostics(training, predictions)
        reports[filename] = {}
        for cid in sorted({r["configuration_id"] for r in joined}):
            rows = [r for r in joined if r["configuration_id"] == cid]
            report = summarize(rows)
            report["feature_diagnostics"] = descriptive_features(rows)
            report["by_block"] = {
                b: summarize([r for r in rows if r["block"] == b])
                for b in sorted({r["block"] for r in rows})
            }
            report["paired_repeat_changes_by_block"] = {
                b: repeat_changes([r for r in rows if r["block"] == b])
                for b in report["by_block"]
            }
            report["by_block_repetition"] = [
                {
                    "block": b,
                    "repetition": rep,
                    **summarize(
                        [r for r in rows if r["block"] == b and r["repetition"] == rep]
                    ),
                }
                for b in sorted({r["block"] for r in rows})
                for rep in range(3)
            ]
            reports[filename][cid] = report
            for a in associations(rows):
                assoc_rows.append(
                    {"diagnostic": filename, "configuration_id": cid, **a}
                )
            for scope, value in [
                ("all_training_blocks", report),
                *report["by_block"].items(),
            ]:
                for stage, stats in value["components"].items():
                    stage_rows.append(
                        {
                            "diagnostic": filename,
                            "configuration_id": cid,
                            "block": scope,
                            "component": stage,
                            **{
                                k: v
                                for k, v in stats.items()
                                if k != "oracle_replacement"
                            },
                            **{
                                "oracle_" + k: v
                                for k, v in stats["oracle_replacement"].items()
                            },
                        }
                    )
            for sort_key in ["ape", "absolute_error_ms"]:
                for rank, row in enumerate(
                    sorted(
                        rows,
                        key=lambda r: (
                            r["ape"] if sort_key == "ape" else abs(r["signed_error_ms"])
                        ),
                        reverse=True,
                    )[:15],
                    1,
                ):
                    outliers.append(
                        {"ranking": sort_key, "rank": rank, "source": filename, **row}
                    )
        residuals.extend({"source": filename, **r} for r in joined)
    output.mkdir(parents=True)
    report = {
        "scope": (
            "Physical training data only; AWS/Vultr excluded; "
            "no fit or candidate selection"
        ),
        "validation_outcomes_read": False,
        "final_outcomes_read": False,
        "training_attempts": len(training),
        "independent_requests": len({r["inputs"]["request_id"] for r in training}),
        "training_blocks": sorted({r["inputs"]["block"] for r in training}),
        "interpretation": (
            "Descriptive associations, not causal effects or independent "
            "repeated-trial tests. Oracle replacements use measured outcomes "
            "and cannot be deployed. Component absolute-error shares are not "
            "additive contributions to total MAE."
        ),
        "reports": reports,
    }
    write_json_atomic(output / "diagnosis.json", report)
    write_json_atomic(output / "largest_outliers.json", {"outliers": outliers})
    with (output / "component_residuals.jsonl").open("w") as handle:
        for row in residuals:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _csv(output / "component_summary.csv", stage_rows)
    _csv(output / "associations.csv", assoc_rows)
    return report
