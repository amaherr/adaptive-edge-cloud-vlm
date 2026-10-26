"""Public CLI smoke checks across extracted command groups."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_vlm.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_benchmark_route_records_fake_trials(tmp_path: Path) -> None:
    run_dir = tmp_path / "fixed"
    assert (
        main(
            [
                "benchmark",
                "--config",
                str(ROOT / "configs/experiments/chunk6_fake_benchmark.yaml"),
                "--mode",
                "edge",
                "--run-dir",
                str(run_dir),
                "--warmups",
                "0",
                "--repetitions",
                "1",
            ]
        )
        == 0
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["measured_attempts"] == 12
    assert summary["status_counts"] == {"ok": 12}
    assert len((run_dir / "requests.jsonl").read_text().splitlines()) == 12


def test_network_trace_route_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "trace"
    assert (
        main(
            [
                "replay-network-trace",
                "--calibration",
                str(ROOT / "configs/network_traces/phase6_synthetic_calibration.json"),
                "--trace",
                str(ROOT / "configs/network_traces/phase6_synthetic_trace.json"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads((output / "evaluation.json").read_text())[
        "synthetic_trace_gate_passed"
    ]
    assert len((output / "decisions.jsonl").read_text().splitlines()) == 9


def test_vqa_routes_freeze_matching_official_subset(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "COCO_val2014_000000000001.jpg").write_bytes(b"image")
    questions = tmp_path / "questions.json"
    annotations = tmp_path / "annotations.json"
    questions.write_text(
        json.dumps(
            {"questions": [{"question_id": 1, "image_id": 1, "question": "What?"}]}
        ),
        encoding="utf-8",
    )
    annotations.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "question_id": 1,
                        "image_id": 1,
                        "answers": [{"answer": "cat"} for _ in range(10)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "vqa.jsonl"
    assert (
        main(
            [
                "import-vqa-v2",
                "--questions",
                str(questions),
                "--annotations",
                str(annotations),
                "--images-dir",
                str(images),
                "--output",
                str(manifest),
                "--count",
                "1",
                "--revision",
                "vqa-test",
            ]
        )
        == 0
    )
    subset = tmp_path / "subset"
    assert (
        main(
            [
                "prepare-vqa-eval-subset",
                "--manifest",
                str(manifest),
                "--questions",
                str(questions),
                "--annotations",
                str(annotations),
                "--output-dir",
                str(subset),
            ]
        )
        == 0
    )
    assert json.loads((subset / "metadata.json").read_text())["question_count"] == 1
