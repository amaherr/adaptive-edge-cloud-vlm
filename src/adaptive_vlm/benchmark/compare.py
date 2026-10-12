"""Cross-policy token and quality comparison from raw benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic


def compare_runs(run_dirs: list[Path], *, output: Path | None = None) -> dict[str, Any]:
    """Compare first measured repetition for every request across policies."""

    if len(run_dirs) < 2:
        raise ValueError("at least two run directories are required")
    plans = [
        json.loads((directory / "plan.json").read_text())
        for directory in run_dirs
    ]
    dataset_hashes = {plan["dataset_manifest_sha256"] for plan in plans}
    model_hashes = {plan["model_contract_sha256"] for plan in plans}
    if len(dataset_hashes) != 1 or len(model_hashes) != 1:
        raise ValueError("runs differ in request manifest or model contract")
    modes = [plan["mode"] for plan in plans]
    if len(set(modes)) != len(modes):
        raise ValueError("comparison requires distinct execution modes")
    by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    failures: dict[str, list[dict[str, Any]]] = {}
    for mode, directory in zip(modes, run_dirs, strict=True):
        rows = [json.loads(line) for line in
                (directory / "requests.jsonl").read_text().splitlines()]
        if any(row.get("plan_sha256") != plans[modes.index(mode)]["plan_sha256"]
               for row in rows):
            raise ValueError(f"raw results do not match plan for {mode}")
        measured = [row for row in rows if not row["warmup"]]
        failures[mode] = [
            {"trial_id": row["trial_id"], "request_id": row["request_id"],
             "status": row["status"], "error_type": row.get("error_type")}
            for row in measured if row["status"] != "ok"
        ]
        first: dict[str, dict[str, Any]] = {}
        for row in measured:
            if row["repetition"] == 0:
                first[row["request_id"]] = row
        by_mode[mode] = first
    request_ids = set.intersection(*(set(rows) for rows in by_mode.values()))
    expected_ids = {
        trial["request_id"] for trial in plans[0]["schedule"]
        if trial["repetition"] == 0
    }
    mismatches: list[dict[str, Any]] = []
    matched = 0
    for request_id in sorted(request_ids):
        request_rows = {mode: by_mode[mode][request_id] for mode in modes}
        if any(row["status"] != "ok" for row in request_rows.values()):
            continue
        tokens = {
            mode: row["output_token_ids"] for mode, row in request_rows.items()
        }
        if len({tuple(value) for value in tokens.values()}) == 1:
            matched += 1
        else:
            mismatches.append(
                {"request_id": request_id, "tokens": tokens,
                 "outputs": {
                     mode: row["output_text"] for mode, row in request_rows.items()
                 }}
            )
    report = {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset_hashes.pop(),
        "model_contract_sha256": model_hashes.pop(),
        "modes": modes,
        "common_first_repetition_requests": len(request_ids),
        "exact_token_matches": matched,
        "mismatches": mismatches,
        "failures": failures,
        "complete_and_equal": (
            request_ids == expected_ids and matched == len(expected_ids)
            and not mismatches and not any(failures.values())
        ),
    }
    if output is not None:
        write_json_atomic(output, report)
    return report
