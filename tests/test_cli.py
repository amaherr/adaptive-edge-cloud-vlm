from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_vlm.cli import main
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.models.moondream import MoondreamAdapter

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


def test_edge_infer_uses_local_policy_and_writes_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_loader(*_args: object, **_kwargs: object) -> FakeModelAdapter:
        return FakeModelAdapter()

    monkeypatch.setattr(MoondreamAdapter, "from_pretrained", fake_loader)
    exit_code = main(
        [
            "edge-infer",
            "--config",
            str(ROOT / "configs/experiments/baseline_edge.yaml"),
            "--artifact-manifest",
            str(ROOT / "configs/model/moondream2_artifact.yaml"),
            "--image",
            str(ROOT / "tests/fixtures/moondream_smoke.ppm"),
            "--prompt",
            "Describe this fixture.",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    record = json.loads(capsys.readouterr().out)
    assert record["mode"] == "edge"
    assert record["metrics"]["model_load_ms"] >= 0
    run_directory = next(tmp_path.iterdir())
    assert (run_directory / "run_manifest.json").is_file()
    assert (run_directory / "request.json").is_file()

