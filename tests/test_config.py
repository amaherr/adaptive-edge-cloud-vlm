from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from adaptive_vlm.config import ConfigError, ExecutionMode, load_experiment_config

ROOT = Path(__file__).parents[1]
EXPERIMENTS = ROOT / "configs" / "experiments"


@pytest.mark.parametrize(
    ("filename", "mode"),
    [
        ("baseline_edge.yaml", ExecutionMode.EDGE),
        ("baseline_cloud.yaml", ExecutionMode.CLOUD),
        ("baseline_distributed.yaml", ExecutionMode.DISTRIBUTED),
    ],
)
def test_loads_baseline_configs(filename: str, mode: ExecutionMode) -> None:
    config = load_experiment_config(EXPERIMENTS / filename)

    assert config.mode is mode
    assert config.model.revision == "2025-06-21"
    assert config.edge.gpu_vram_mib == 4096
    assert config.cloud.status == "pending"
    assert config.runtime.measured_repetitions == 30
    assert config.artifacts_dir == ROOT / "artifacts"


def test_resolved_config_is_json_compatible() -> None:
    config = load_experiment_config(EXPERIMENTS / "baseline_edge.yaml")

    resolved = config.to_dict()

    assert resolved["mode"] == "edge"
    assert resolved["model"]["precision"] == "bf16"
    assert resolved["source_path"].endswith("baseline_edge.yaml")


def test_rejects_moving_model_revision(tmp_path: Path) -> None:
    model = yaml.safe_load((ROOT / "configs/model/moondream2.yaml").read_text())
    model["revision"] = "main"
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(model), encoding="utf-8")
    experiment = _experiment_with_model(model_path)
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    with pytest.raises(ConfigError, match="must be pinned"):
        load_experiment_config(experiment_path)


def test_rejects_unknown_fields(tmp_path: Path) -> None:
    experiment = _experiment_with_model(ROOT / "configs/model/moondream2.yaml")
    experiment["typo_field"] = True
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    with pytest.raises(ConfigError, match="typo_field"):
        load_experiment_config(experiment_path)


def _experiment_with_model(model_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "test",
        "mode": "edge",
        "model_config": str(model_path),
        "edge_hardware_config": str(ROOT / "configs/hardware/edge_rtx3050.yaml"),
        "cloud_hardware_config": str(
            ROOT / "configs/hardware/cloud_pending.yaml"
        ),
        "dataset": {"manifest_path": "manifest.jsonl", "revision": "test-v1"},
        "runtime": {
            "seed": 42,
            "warmup_requests": 0,
            "measured_repetitions": 1,
        },
        "transport": {
            "host": "127.0.0.1",
            "port": 5050,
            "timeout_seconds": 1.0,
            "protocol_version": 1,
        },
        "artifacts_dir": "artifacts",
    }

