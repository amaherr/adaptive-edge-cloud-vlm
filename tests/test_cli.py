from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_vlm.cli import main

ROOT = Path(__file__).parents[1]


def test_config_validate_prints_resolved_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["config", "validate", str(ROOT / "configs/experiments/baseline_edge.yaml")]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["mode"] == "edge"


def test_fake_infer_writes_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "fake-infer",
            "--config",
            str(ROOT / "configs/experiments/chunk1_smoke.yaml"),
            "--image",
            str(ROOT / "tests/fixtures/smoke_image.txt"),
            "--prompt",
            "Describe this fixture.",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    run_directory = next(tmp_path.iterdir())
    assert record["status"] == "ok"
    assert (run_directory / "run_manifest.json").is_file()
    assert (run_directory / "request.json").is_file()


def test_fake_infer_rejects_real_model_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "fake-infer",
            "--config",
            str(ROOT / "configs/experiments/baseline_edge.yaml"),
            "--image",
            str(ROOT / "tests/fixtures/smoke_image.txt"),
            "--prompt",
            "Describe this fixture.",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "requires" in captured.err

