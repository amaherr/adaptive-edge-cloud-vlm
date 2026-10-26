from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_vlm.cli import main
from adaptive_vlm.config import (
    ExecutionMode,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.execution.cloud import HealthStatus
from adaptive_vlm.models.base import ModelMetadata
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


def test_adaptive_benchmark_routes_to_command_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[object] = []

    def run_command(args: object, *, rss_reader: object) -> int:
        received.extend((args, rss_reader))
        return 0

    monkeypatch.setattr("adaptive_vlm.cli.run_adaptive_benchmark_command", run_command)
    assert (
        main(
            [
                "adaptive-benchmark",
                "--config",
                "config.yaml",
                "--manifest",
                "tasks.jsonl",
                "--calibration",
                "fit.json",
                "--replay",
                "replay.json",
                "--run-dir",
                "run",
                "--artifact-manifest",
                "model.yaml",
            ]
        )
        == 0
    )
    assert received[0].config == Path("config.yaml")
    assert callable(received[1])


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
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert "Loading and verifying the pinned Moondream2 model" in captured.err
    assert "Running edge inference" in captured.err
    assert record["mode"] == "edge"
    assert record["metrics"]["model_load_ms"] >= 0
    run_directory = next(tmp_path.iterdir())
    manifest_path = run_directory / "run_manifest.json"
    assert manifest_path.is_file()
    assert (run_directory / "request.json").is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["environment"]["execution_placement"] == {
        "vision_device": "cpu",
        "text_device": "cpu",
        "offline_model_cache": False,
    }


def test_cloud_infer_uses_remote_policy_and_writes_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeModelAdapter()

    class StubCloudPolicy:
        mode = ExecutionMode.CLOUD

        def __init__(self, **_: object) -> None:
            pass

        def health(self) -> HealthStatus:
            return HealthStatus(True, adapter.metadata.fingerprint)

        def infer(self, request: object) -> object:
            return adapter.generate(request)  # type: ignore[arg-type]

    monkeypatch.setattr("adaptive_vlm.cli.CloudExecutionPolicy", StubCloudPolicy)
    exit_code = main(
        [
            "cloud-infer",
            "--config",
            str(ROOT / "configs/experiments/chunk4_fake_cloud.yaml"),
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
    assert "Checking cloud server" in captured.err
    assert "Running cloud inference" in captured.err
    assert record["mode"] == "cloud"
    run_directory = next(tmp_path.iterdir())
    assert (run_directory / "run_manifest.json").is_file()
    assert (run_directory / "request.json").is_file()


def test_cloud_server_starts_fake_adapter(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    instances: list[object] = []

    class StubServer:
        def __init__(self, *, host: str, port: int, **_: object) -> None:
            self.host = host
            self.port = port
            self.started = False
            self.served = False
            self.stopped = False
            instances.append(self)

        def start(self) -> None:
            self.started = True

        def serve_forever(self) -> None:
            self.served = True

        def shutdown(self) -> None:
            self.stopped = True

    monkeypatch.setattr("adaptive_vlm.cli.CloudServer", StubServer)
    exit_code = main(
        [
            "cloud-server",
            "--config",
            str(ROOT / "configs/experiments/chunk4_fake_cloud.yaml"),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    ready = json.loads(captured.out)
    assert "Cloud server ready" in captured.err
    assert "Waiting for requests" in captured.err
    assert ready["ready"] is True
    config = load_experiment_config(ROOT / "configs/experiments/chunk4_fake_cloud.yaml")
    assert ready["model_artifact_sha256"] == FakeModelAdapter().metadata.fingerprint
    assert ready["model_contract_fingerprint"] == model_contract_fingerprint(
        config.model
    )
    server = instances[0]
    assert server.started  # type: ignore[attr-defined]
    assert server.served  # type: ignore[attr-defined]
    assert server.stopped  # type: ignore[attr-defined]


def test_cloud_commands_reject_pending_hardware(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "cloud-server",
            "--config",
            str(ROOT / "configs/experiments/baseline_cloud.yaml"),
        ]
    )

    assert exit_code == 2
    assert "pending" in capsys.readouterr().err


def test_edge_verbose_reports_model_details(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MoondreamAdapter,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeModelAdapter(),
    )

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
            "--verbose",
        ]
    )

    assert exit_code == 0
    assert "vikhyatk/moondream2@9a7d402" in capsys.readouterr().err


def test_distributed_server_loads_only_text_stage(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    loader_kwargs: dict[str, object] = {}

    class StubTextAdapter:
        component_role = "text"
        metadata = ModelMetadata("model", "revision", "a" * 64)

        @staticmethod
        def prepare_prompt(_prompt: str) -> tuple[int, ...]:
            return (1,)

        @staticmethod
        def generate_from_payload(_request: object) -> object:
            raise AssertionError("the stub server should not receive a request")

    class StubServer:
        def __init__(self, *, host: str, port: int, **_: object) -> None:
            self.host = host
            self.port = port

        def start(self) -> None:
            pass

        def serve_forever(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

    def fake_loader(*_args: object, **kwargs: object) -> StubTextAdapter:
        loader_kwargs.update(kwargs)
        return StubTextAdapter()

    monkeypatch.setattr(MoondreamAdapter, "from_pretrained", fake_loader)
    monkeypatch.setattr("adaptive_vlm.cli.CloudServer", StubServer)

    exit_code = main(
        [
            "distributed-server",
            "--config",
            str(ROOT / "configs/experiments/chunk5_local_distributed.yaml"),
            "--artifact-manifest",
            str(ROOT / "configs/model/moondream2_artifact.yaml"),
        ]
    )

    assert exit_code == 0
    assert loader_kwargs["components"] == "text"
    ready = json.loads(capsys.readouterr().out)
    assert ready["component_role"] == "text"


def test_distributed_infer_uses_vision_stage_and_writes_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_kwargs: dict[str, object] = {}
    policy_kwargs: dict[str, object] = {}
    fake = FakeModelAdapter()

    class StubVisionAdapter:
        component_role = "vision"

    class StubDistributedPolicy:
        mode = ExecutionMode.DISTRIBUTED

        def __init__(self, **kwargs: object) -> None:
            policy_kwargs.update(kwargs)

        @staticmethod
        def health() -> HealthStatus:
            return HealthStatus(True, "contract")

        @staticmethod
        def infer(request: object) -> object:
            return fake.generate(request)  # type: ignore[arg-type]

    def fake_loader(*_args: object, **kwargs: object) -> StubVisionAdapter:
        loader_kwargs.update(kwargs)
        return StubVisionAdapter()

    monkeypatch.setattr(MoondreamAdapter, "from_pretrained", fake_loader)
    monkeypatch.setattr(
        "adaptive_vlm.cli.DistributedExecutionPolicy", StubDistributedPolicy
    )

    exit_code = main(
        [
            "distributed-infer",
            "--config",
            str(ROOT / "configs/experiments/chunk5_local_distributed.yaml"),
            "--artifact-manifest",
            str(ROOT / "configs/model/moondream2_artifact.yaml"),
            "--image",
            str(ROOT / "tests/fixtures/moondream_smoke.ppm"),
            "--prompt",
            "Describe this fixture.",
            "--output-dir",
            str(tmp_path),
            "--pipeline",
            "prompt_first",
        ]
    )

    assert exit_code == 0
    assert loader_kwargs["components"] == "vision"
    assert policy_kwargs["pipeline"] == "prompt_first"
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert record["mode"] == "distributed"
    assert "Loading the Moondream vision stage" in captured.err
    assert "Running distributed inference" in captured.err
    run_directory = next(tmp_path.iterdir())
    assert (run_directory / "run_manifest.json").is_file()
    assert (run_directory / "request.json").is_file()


def test_distributed_commands_reject_pending_cloud(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "distributed-infer",
            "--config",
            str(ROOT / "configs/experiments/baseline_distributed.yaml"),
            "--artifact-manifest",
            str(ROOT / "configs/model/moondream2_artifact.yaml"),
            "--image",
            str(ROOT / "tests/fixtures/moondream_smoke.ppm"),
            "--prompt",
            "Describe this fixture.",
        ]
    )

    assert exit_code == 2
    assert "pending" in capsys.readouterr().err
