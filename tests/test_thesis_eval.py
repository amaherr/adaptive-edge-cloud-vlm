"""Phase 7 split, failure accounting, and official-format export contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.benchmark.thesis_eval import analyze_thesis_campaign
from adaptive_vlm.benchmark.vqa_export import (
    export_vqa_predictions,
    prepare_vqa_eval_subset,
)
from adaptive_vlm.data.manifest import load_frozen_manifest


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _campaign(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    spec = tmp_path / "spec.json"
    _write_json(
        spec,
        {
            "schema_version": 1,
            "campaign_id": "test",
            "design": "retrospective_diagnostic",
            "dataset_manifest_sha256": "a" * 64,
            "model_contract_sha256": "b" * 64,
            "calibration_id": "c" * 64,
            "predictor_training_request_ids": ["val"],
            "validation_request_ids": ["val"],
            "evaluation_request_ids": ["test-1", "test-2"],
            "validation_network_block_ids": [],
            "evaluation_network_block_ids": [],
            "sla_ms": 100,
            "bootstrap_seed": 7,
            "bootstrap_resamples": 100,
            "confidence_level": 0.95,
            "task_type": "fixture",
            "network_regime": "uncontrolled_loopback",
        },
    )
    schedule = [
        {
            "trial_id": f"r{i}",
            "request_id": request_id,
            "warmup": False,
            "repetition": 0,
        }
        for i, request_id in enumerate(("val", "test-1", "test-2"))
    ]
    fixed: list[Path] = []
    for mode in ("adaptive", "edge", "cloud", "distributed"):
        directory = tmp_path / mode
        directory.mkdir()
        plan = {
            "mode": mode,
            "plan_sha256": mode,
            "schedule": schedule,
            "config": {"dataset": {"manifest_path": "fixture"}},
            "dataset_manifest_sha256": "a" * 64,
            "model_contract_sha256": "b" * 64,
            "calibration_id": "c" * 64,
        }
        _write_json(directory / "plan.json", plan)
        _write_json(
            directory / "run_manifest.json",
            {"run_id": mode, "dataset_manifest_sha256": "a" * 64},
        )
        durations = {
            "adaptive": (12, 40, 120),
            "edge": (10, 30, 45),
            "cloud": (20, 20, 20),
            "distributed": (15, 10, 15),
        }[mode]
        rows = []
        for i, entry in enumerate(schedule):
            status = "timeout" if mode == "adaptive" and i == 2 else "ok"
            rows.append(
                {
                    **entry,
                    "plan_sha256": mode,
                    "run_id": mode,
                    "dataset_manifest_sha256": "a" * 64,
                    "model_contract_sha256": "b" * 64,
                    "status": status,
                    "output_token_ids": [1] if status == "ok" else [],
                    "observed_end_to_end_ms": durations[i],
                    "metrics": {"request_wire_bytes": 0, "response_wire_bytes": 0},
                    "selected_configuration_id": "edge/full"
                    if mode == "adaptive"
                    else None,
                    "selection_overhead_ms": 1,
                    "system_state": {"network_bucket_id": "local"},
                    "attempts": [],
                }
            )
        (directory / "requests.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        if mode != "adaptive":
            fixed.append(directory)
    return spec, tmp_path / "adaptive", fixed


def test_thesis_report_uses_validation_choice_and_counts_failed_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, adaptive, fixed = _campaign(tmp_path)
    monkeypatch.setattr(
        "adaptive_vlm.benchmark.thesis_eval.analyze_adaptive_run",
        lambda *_args: {
            "decision_trace_errors": [],
            "wire_accounting_errors": [],
            "placement_mismatch_trials": [],
            "same_placement_token_mismatches": [],
        },
    )
    report = analyze_thesis_campaign(spec, adaptive, fixed, output_dir=tmp_path / "out")
    assert report["best_static_validation_choice"] == "edge"
    assert report["evaluation"]["adaptive"]["status_counts"] == {"ok": 1, "timeout": 1}
    assert report["evaluation"]["adaptive"]["sla_success_rate"] == 0.5
    assert report["paired"]["edge"]["latency_ms"]["paired_trials"] == 1
    assert report["paired"]["edge"]["sla_success"]["paired_trials"] == 2
    assert report["measured_oracle_regret_ms"]["paired_trials"] == 1
    assert report["g7_passed"] is False
    assert len((tmp_path / "out" / "paired_trials.jsonl").read_text().splitlines()) == 2


def test_thesis_report_rejects_split_leakage_and_missing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, adaptive, fixed = _campaign(tmp_path)
    monkeypatch.setattr(
        "adaptive_vlm.benchmark.thesis_eval.analyze_adaptive_run",
        lambda *_args: {
            "decision_trace_errors": [],
            "wire_accounting_errors": [],
            "placement_mismatch_trials": [],
            "same_placement_token_mismatches": [],
        },
    )
    value = json.loads(spec.read_text())
    value["predictor_training_request_ids"].append("test-1")
    _write_json(spec, value)
    with pytest.raises(ValueError, match="leaks"):
        analyze_thesis_campaign(spec, adaptive, fixed, output_dir=tmp_path / "out")
    value["predictor_training_request_ids"].pop()
    _write_json(spec, value)
    missing = fixed[0] / "requests.jsonl"
    missing.write_text(missing.read_text().splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unexpected"):
        analyze_thesis_campaign(spec, adaptive, fixed, output_dir=tmp_path / "out")


def test_vqa_export_keeps_failures_in_official_denominator(tmp_path: Path) -> None:
    manifest_path = tmp_path / "vqa.jsonl"
    items = []
    for question_id in (7, 8):
        image = tmp_path / f"{question_id}.jpg"
        image.write_bytes(bytes([question_id]))
        items.append(
            {
                "request_id": f"vqa-v2-val2014-{question_id}",
                "image_path": str(image),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "prompt": "What is visible?",
                "references": ["cat"] * 10,
                "scoring": "vqa_consensus",
                "source": "vqa-v2-val2014",
                "revision": "v2-test",
            }
        )
    manifest_path.write_text(
        "\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8"
    )
    manifest = load_frozen_manifest(manifest_path)
    assert all(item.task_type == "vqa_short_answer" for item in manifest.items)
    run = tmp_path / "run"
    run.mkdir()
    schedule = [
        {
            "trial_id": f"r{i}",
            "request_id": item["request_id"],
            "warmup": False,
            "repetition": 0,
        }
        for i, item in enumerate(items)
    ]
    _write_json(
        run / "plan.json",
        {
            "plan_sha256": "p",
            "dataset_manifest_sha256": manifest.sha256,
            "schedule": schedule,
        },
    )
    rows = [
        {
            **entry,
            "plan_sha256": "p",
            "dataset_manifest_sha256": manifest.sha256,
            "status": "ok" if i == 0 else "timeout",
            "output_text": "cat" if i == 0 else None,
        }
        for i, entry in enumerate(schedule)
    ]
    (run / "requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    output = tmp_path / "predictions.json"
    metadata = export_vqa_predictions(manifest_path, run, output)
    assert json.loads(output.read_text()) == [
        {"question_id": 7, "answer": "cat"},
        {"question_id": 8, "answer": ""},
    ]
    assert metadata["failed_answers_exported_empty"] == 1
    with pytest.raises(FileExistsError):
        export_vqa_predictions(manifest_path, run, output)

    questions = tmp_path / "questions.json"
    annotations = tmp_path / "annotations.json"
    _write_json(
        questions,
        {
            "info": {"version": "v2"},
            "questions": [{"question_id": i, "image_id": i} for i in (7, 8, 9)],
        },
    )
    _write_json(
        annotations,
        {
            "info": {"version": "v2"},
            "annotations": [{"question_id": i, "image_id": i} for i in (7, 8, 9)],
        },
    )
    subset_dir = tmp_path / "subset"
    subset = prepare_vqa_eval_subset(manifest_path, questions, annotations, subset_dir)
    assert subset["question_count"] == 2
    assert {
        row["question_id"]
        for row in json.loads((subset_dir / "annotations.json").read_text())[
            "annotations"
        ]
    } == {7, 8}
