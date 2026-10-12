"""Cross-policy token and quality comparison from raw benchmark artifacts."""

from __future__ import annotations

import json
from itertools import zip_longest
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.runner import semantic_config_sha256


def compare_runs(run_dirs: list[Path], *, output: Path | None = None) -> dict[str, Any]:
    """Compare all measured request/repetition pairs across policies."""

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
    if len({semantic_config_sha256(plan["config"]) for plan in plans}) != 1:
        raise ValueError("runs differ in resolved experiment configuration")
    if any(
        [
            (trial["repetition"], trial["request_id"], trial["warmup"])
            for trial in plan["schedule"]
        ] != [
            (trial["repetition"], trial["request_id"], trial["warmup"])
            for trial in plans[0]["schedule"]
        ] for plan in plans[1:]
    ):
        raise ValueError("runs differ in trial schedule")
    modes = [plan["mode"] for plan in plans]
    if len(set(modes)) != len(modes):
        raise ValueError("comparison requires distinct execution modes")
    by_mode: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    failures: dict[str, list[dict[str, Any]]] = {}
    quality_by_mode: dict[str, dict[str, Any]] = {}
    for mode, directory in zip(modes, run_dirs, strict=True):
        summary = json.loads((directory / "summary.json").read_text())
        quality_by_mode[mode] = {
            "quality_mean_success": summary["quality_mean_success"],
            "quality_scored_attempts": summary["quality_scored_attempts"],
            "quality_metric": summary["quality_metric"],
        }
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
        observed: dict[tuple[int, str], dict[str, Any]] = {}
        for row in measured:
            key = (row["repetition"], row["request_id"])
            if key in observed:
                raise ValueError(f"duplicate measured pair in {mode}: {key}")
            observed[key] = row
        by_mode[mode] = observed
    common_pairs = set.intersection(*(set(rows) for rows in by_mode.values()))
    expected_pairs = {
        (trial["repetition"], trial["request_id"])
        for trial in plans[0]["schedule"] if not trial["warmup"]
    }
    mismatches: list[dict[str, Any]] = []
    text_mismatches: list[dict[str, Any]] = []
    matched = 0
    for repetition, request_id in sorted(common_pairs):
        request_rows = {
            mode: by_mode[mode][(repetition, request_id)] for mode in modes
        }
        if any(row["status"] != "ok" for row in request_rows.values()):
            continue
        tokens = {
            mode: row["output_token_ids"] for mode, row in request_rows.items()
        }
        outputs = {mode: row["output_text"] for mode, row in request_rows.items()}
        if len(set(outputs.values())) != 1:
            text_mismatches.append(
                {"request_id": request_id, "repetition": repetition,
                 "outputs": outputs}
            )
        if len({tuple(value) for value in tokens.values()}) == 1:
            matched += 1
        else:
            reference_tokens = tokens[modes[0]]
            mismatches.append(
                {"request_id": request_id, "repetition": repetition,
                 "tokens": tokens,
                 "outputs": outputs,
                 "first_differing_token_index": {
                     mode: next(
                         (
                             index for index, pair in enumerate(
                                 zip_longest(reference_tokens, mode_tokens)
                             ) if pair[0] != pair[1]
                         ),
                         None,
                     )
                     for mode, mode_tokens in tokens.items() if mode != modes[0]
                 }}
            )
    report = {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset_hashes.pop(),
        "model_contract_sha256": model_hashes.pop(),
        "modes": modes,
        "common_measured_pairs": len(common_pairs),
        "expected_measured_pairs": len(expected_pairs),
        "exact_token_matches": matched,
        "mismatches": mismatches,
        "text_mismatches": text_mismatches,
        "quality_by_mode": quality_by_mode,
        "failures": failures,
        "complete_and_equal": (
            common_pairs == expected_pairs and matched == len(expected_pairs)
            and not mismatches and not text_mismatches
            and not any(failures.values())
        ),
    }
    if output is not None:
        write_json_atomic(output, report)
    return report
