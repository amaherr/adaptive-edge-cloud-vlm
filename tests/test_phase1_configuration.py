from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_vlm.cli import main
from adaptive_vlm.config import (
    ExecutionMode,
    TransportConfig,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.execution.adaptive import load_phase1_policies
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.remote import (
    RemoteCapabilities,
    check_remote_capabilities,
)
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.moondream import MoondreamAdapter
from adaptive_vlm.scheduler.configuration import (
    CONFIGURATIONS,
    CapabilityRegistry,
    ConfigurationId,
)
from adaptive_vlm.scheduler.dispatch import ConfigurationDispatcher
from adaptive_vlm.scheduler.state import SystemState, TaskState
from adaptive_vlm.transport.client import ExchangeResult
from adaptive_vlm.transport.protocol import (
    Frame,
    MessageType,
    ProtocolError,
    ReceivedFrame,
    encode_frame,
)
from adaptive_vlm.transport.server import CloudRequestProcessor
from adaptive_vlm.transport.tensors import TensorInferenceRequest

ROOT = Path(__file__).parents[1]
FINGERPRINT = "a" * 64
REMOTE = RemoteCapabilities(
    True,
    FINGERPRINT,
    frozenset({"raw_image", "projected_synchronous", "projected_prompt_first"}),
)


class _Policy(ExecutionPolicy):
    def __init__(self, mode: ExecutionMode) -> None:
        self._mode = mode
        self.calls: list[str] = []

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self.calls.append(request.request_id)
        return InferenceResult(request.request_id, self.mode.value, (1, 2))


def _policies() -> dict[ConfigurationId, ExecutionPolicy]:
    return {
        spec.id: _Policy(spec.mode) for spec in CONFIGURATIONS
    }


def _received(frame: Frame) -> ReceivedFrame:
    return ReceivedFrame(frame, len(encode_frame(frame)))


def test_dual_server_reports_paths_and_routes_both_requests() -> None:
    calls: list[str] = []

    def image_infer(request: InferenceRequest) -> InferenceResult:
        calls.append("image")
        return InferenceResult(request.request_id, "same", (7,))

    def tensor_infer(request: TensorInferenceRequest) -> InferenceResult:
        calls.append("tensor")
        return InferenceResult(request.request_id, "same", (7,))

    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        inference_function=image_infer,
        tensor_inference_function=tensor_infer,
    )
    capabilities = processor.handle(_received(Frame(
        1, MessageType.CAPABILITY_REQUEST, "caps", {},
    )))
    assert capabilities.metadata["paths"] == [
        "raw_image", "projected_synchronous"
    ]
    image_result = processor.handle(_received(Frame(
        1, MessageType.INFERENCE_REQUEST, "image", {
            "prompt": "Describe", "model_fingerprint": FINGERPRINT,
            "image_suffix": ".png",
        }, b"image bytes",
    )))
    tensor_result = processor.handle(_received(Frame(
        1, MessageType.TENSOR_REQUEST, "tensor", {
            "prompt": "Describe", "model_fingerprint": FINGERPRINT,
            "shape": [1], "dtype": "float32", "byte_order": "little",
            "layout": "contiguous", "nbytes": 4, "pipeline": "synchronous",
        }, bytes(4),
    )))
    assert calls == ["image", "tensor"]
    assert image_result.metadata["output_token_ids"] == [7]
    assert tensor_result.metadata["output_token_ids"] == [7]


def test_capability_client_rejects_model_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=FINGERPRINT,
        inference_function=lambda request: InferenceResult(
            request.request_id, "ok", (1,)
        ),
    )

    def exchange(_: TransportConfig, frame: Frame) -> ExchangeResult:
        return ExchangeResult(processor.handle(_received(frame)), 1, 1)

    monkeypatch.setattr("adaptive_vlm.execution.remote.exchange_frame", exchange)
    transport = TransportConfig("127.0.0.1", 5050, 1.0, 1)
    correct = ModelMetadata("model", "revision", FINGERPRINT)
    assert check_remote_capabilities(transport, correct).paths == {"raw_image"}
    wrong = ModelMetadata("model", "revision", "b" * 64)
    with pytest.raises(ProtocolError, match="different model fingerprint"):
        check_remote_capabilities(transport, wrong)


def test_registry_routes_exactly_one_loaded_policy(tmp_path: Path) -> None:
    policies = _policies()
    registry = CapabilityRegistry(
        policies=policies,
        expected_model_fingerprint=FINGERPRINT,
        remote=REMOTE,
        edge_resident_memory_mib=500,
        edge_required_peak_mib=550,
        edge_memory_budget_mib=600,
    )
    assert all(status.available for status in registry.statuses())
    request = InferenceRequest("one", tmp_path / "image.png", "Describe")
    request.image_path.write_bytes(b"image")
    outcome = ConfigurationDispatcher(registry).infer(
        request,
        configuration_id=ConfigurationId.DISTRIBUTED_PROJECTED,
        system_state=SystemState("manual", uplink_bps=10_000_000, rtt_ms=20),
        task_state=TaskState.from_request(request, expected_output_tokens=8),
    )
    assert outcome.result.output_text == "distributed"
    assert outcome.decision.configuration_id is ConfigurationId.DISTRIBUTED_PROJECTED
    assert [policy.calls for policy in policies.values()] == [[], [], ["one"]]
    assert outcome.decision.to_dict()["system_state"]["uplink_bps"] == 10_000_000
    assert outcome.selection_overhead_ms >= 0


def test_registry_excludes_missing_path_model_and_memory(tmp_path: Path) -> None:
    policies = _policies()
    request = InferenceRequest("one", tmp_path / "image.png", "Describe")
    no_split = CapabilityRegistry(
        policies=policies,
        expected_model_fingerprint=FINGERPRINT,
        remote=RemoteCapabilities(True, FINGERPRINT, frozenset({"raw_image"})),
    )
    with pytest.raises(ValueError, match="remote_path_missing"):
        ConfigurationDispatcher(no_split).infer(
            request,
            configuration_id=ConfigurationId.DISTRIBUTED_PROJECTED,
            system_state=SystemState("manual"),
            task_state=TaskState(5, 8),
        )
    assert all(not policy.calls for policy in policies.values())
    mismatch = CapabilityRegistry(
        policies=policies,
        expected_model_fingerprint=FINGERPRINT,
        remote=RemoteCapabilities(True, "b" * 64, REMOTE.paths),
    )
    assert mismatch.status(CONFIGURATIONS[1]).reason == "remote_model_mismatch"
    low_memory = CapabilityRegistry(
        policies=policies,
        expected_model_fingerprint=FINGERPRINT,
        remote=REMOTE,
        edge_resident_memory_mib=700,
        edge_required_peak_mib=700,
        edge_memory_budget_mib=600,
    )
    assert all(not status.available for status in low_memory.statuses())
    unknown_memory = CapabilityRegistry(
        policies=policies,
        expected_model_fingerprint=FINGERPRINT,
        remote=REMOTE,
        edge_resident_memory_mib=500,
        edge_memory_budget_mib=600,
    )
    assert all(status.reason == "edge_memory_requirement_unknown"
               for status in unknown_memory.statuses())
    offline = CapabilityRegistry(
        policies=policies,
        expected_model_fingerprint=FINGERPRINT,
        remote=None,
        remote_error="connection refused",
    )
    assert offline.status(CONFIGURATIONS[0]).available
    assert offline.status(CONFIGURATIONS[1]).reason == (
        "remote_unavailable: connection refused"
    )


def test_adaptive_infer_writes_decision_and_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policies = _policies()
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_local_campaign.yaml"
    )
    remote = RemoteCapabilities(True, model_contract_fingerprint(config.model),
                                REMOTE.paths)
    monkeypatch.setattr(
        "adaptive_vlm.cli.load_phase1_policies", lambda *_args, **_kwargs: policies
    )
    monkeypatch.setattr(
        "adaptive_vlm.cli.check_remote_capabilities",
        lambda *_args, **_kwargs: remote,
    )
    monkeypatch.setattr("adaptive_vlm.cli._current_rss_mib", lambda: 500.0)
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    exit_code = main([
        "adaptive-infer", "--config",
        str(ROOT / "configs/experiments/chunk6_local_campaign.yaml"),
        "--configuration-id", ConfigurationId.CLOUD_RAW_IMAGE.value,
        "--image", str(image), "--prompt", "Describe",
        "--edge-memory-budget-mib", "600", "--edge-required-peak-mib", "550",
        "--output-dir", str(tmp_path / "runs"),
    ])
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decision"]["configuration_id"] == "cloud/raw_image"
    assert output["request"]["mode"] == "cloud"
    assert [policy.calls for policy in policies.values()] == [[], ["adaptive-0001"], []]
    run_directory = next((tmp_path / "runs").iterdir())
    assert (run_directory / "decision.json").is_file()
    assert (run_directory / "request.json").is_file()


def test_adaptive_infer_requires_peak_estimate_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def must_not_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("memory preflight should run before model loading")

    monkeypatch.setattr("adaptive_vlm.cli.load_phase1_policies", must_not_load)
    code = main([
        "adaptive-infer", "--config",
        str(ROOT / "configs/experiments/chunk6_local_campaign.yaml"),
        "--configuration-id", ConfigurationId.EDGE_FULL.value,
        "--image", str(ROOT / "tests/fixtures/moondream_smoke.ppm"),
        "--prompt", "Describe", "--edge-memory-budget-mib", "600",
    ])
    assert code == 2
    assert "requires --edge-required-peak-mib" in capsys.readouterr().err


def test_phase1_loader_shares_one_resident_full_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[dict[str, object]] = []

    class Adapter:
        component_role = "full"

    def loader(*_args: object, **kwargs: object) -> Adapter:
        loaded.append(kwargs)
        return Adapter()

    monkeypatch.setattr(MoondreamAdapter, "from_pretrained", loader)
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_local_campaign.yaml"
    )
    policies = load_phase1_policies(
        config,
        artifact_manifest=ROOT / "configs/model/moondream2_artifact.yaml",
        cache_dir=ROOT / ".model-cache",
        offline=True,
        vision_device="cpu",
        text_device="cpu",
        edge_components="full",
    )
    assert set(policies) == set(ConfigurationId)
    assert len(loaded) == 1
    assert loaded[0]["components"] == "full"


def test_adaptive_server_exposes_all_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    processors: list[CloudRequestProcessor] = []

    class Adapter:
        metadata = ModelMetadata("model", "revision", "b" * 64)

        def run_monolithic(self, _request: InferenceRequest) -> object:
            raise AssertionError("no request expected")

        def generate_from_payload(self, _request: TensorInferenceRequest) -> object:
            raise AssertionError("no request expected")

        def prepare_prompt(self, _prompt: str) -> tuple[int, ...]:
            return (1,)

    class Server:
        def __init__(self, *, host: str, port: int,
                     processor: CloudRequestProcessor, **_: object) -> None:
            self.host = host
            self.port = port
            processors.append(processor)

        def start(self) -> None:
            pass

        def serve_forever(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(MoondreamAdapter, "from_pretrained",
                        lambda *_args, **_kwargs: Adapter())
    monkeypatch.setattr("adaptive_vlm.cli.CloudServer", Server)
    assert main([
        "adaptive-server", "--config",
        str(ROOT / "configs/experiments/chunk4_local_cloud.yaml"),
        "--artifact-manifest",
        str(ROOT / "configs/model/moondream2_artifact.yaml"),
    ]) == 0
    ready = json.loads(capsys.readouterr().out)
    assert ready["supported_paths"] == [
        "raw_image", "projected_synchronous", "projected_prompt_first"
    ]
    assert processors[0].supported_paths == tuple(ready["supported_paths"])
