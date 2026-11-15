"""Interleaved three-placement runner, resume, device failures and profiles."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.interleaved import (
    DEVICE_RESTART_EXIT_CODE,
    PLACEMENTS,
    DeviceContextFailure,
    interleave_parameters,
    is_device_context_failure,
    make_interleaved_order,
    placement_schedule,
    run_interleaved_benchmark,
    run_with_device_restarts,
    verify_interleave_parameters,
)
from adaptive_vlm.benchmark.profile import build_profile
from adaptive_vlm.cli import main
from adaptive_vlm.config import ExecutionMode, load_experiment_config
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.base import ExecutionPolicy
from adaptive_vlm.execution.remote import RemoteInferenceError
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer

ROOT = Path(__file__).resolve().parents[2]
IDS = [f"correctness-{index:03d}" for index in range(1, 7)]


class StubPolicy(ExecutionPolicy):
    """Profile-complete metrics; optional failure injection by call number."""

    def __init__(
        self,
        mode: ExecutionMode,
        log: list[str],
        *,
        fail_on: dict[int, BaseException] | None = None,
    ) -> None:
        self._mode = mode
        self.log = log
        self.calls = 0
        self.fail_on = fail_on or {}

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self.calls += 1
        self.log.append(f"{self._mode.value}:{request.request_id}")
        if self.calls in self.fail_on:
            raise self.fail_on[self.calls]
        model = {
            "prompt_tokens": 12,
            "prompt_prefill_ms": 0.000001,
            "decode_ms": 0.000001,
            "output_tokens": 1,
            "image_width": 16,
            "image_height": 16,
        }
        if self._mode is ExecutionMode.EDGE:
            metrics: dict[str, Any] = {**model, "end_to_end_ms": 0.000002}
        else:
            metrics = {
                **{f"server_model_{name}": value for name, value in model.items()},
                "network_round_trip_ms": 0.000002,
                "server_total_ms": 0.000001,
                "request_wire_bytes": 3_000_000,
                "server_request_wire_bytes": 3_000_000,
                "response_wire_bytes": 30,
            }
            if self._mode is ExecutionMode.CLOUD:
                metrics.update(
                    image_bytes=request.image_path.stat().st_size,
                    image_read_ms=0.000001,
                )
            else:
                metrics.update(
                    projected_payload_bytes=729 * 2048 * 2,
                    edge_vision_end_to_end_ms=0.000001,
                    tensor_serialization_ms=0.000001,
                    image_width=16,
                    image_height=16,
                )
        return InferenceResult(request.request_id, "red", (1,), metrics)


def _setup() -> tuple[Any, Any]:
    config = load_experiment_config(ROOT / "configs/experiments/local_campaign.yaml")
    return config, load_frozen_manifest(config.dataset.manifest_path)


def _run(
    tmp_path: Path,
    *,
    policies: dict[str, ExecutionPolicy],
    prepare: Any = None,
    resume: bool = False,
    max_failures: int = 0,
) -> dict[str, Any]:
    config, manifest = _setup()
    return run_interleaved_benchmark(
        config=config,
        manifest=manifest,
        policies=policies,
        run_manifests={
            p: create_run_manifest(config, {}, dataset_manifest_sha256=manifest.sha256)
            for p in policies
        },
        run_dir=tmp_path / "run",
        request_ids=IDS,
        repetitions=[1, 2],
        warmups=1,
        seed=5113,
        prepare_stage=prepare,
        resume=resume,
        max_consecutive_failures=max_failures,
    )


def _policies(log: list[str]) -> dict[str, ExecutionPolicy]:
    return {p: StubPolicy(ExecutionMode(p), log) for p in PLACEMENTS}


def _journal(tmp_path: Path, placement: str) -> list[dict[str, Any]]:
    path = tmp_path / "run" / placement / "requests.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_order_runs_every_placement_once_per_request_back_to_back() -> None:
    order = make_interleaved_order(
        IDS, repetitions=[0, 2], warmups=2, placements=PLACEMENTS, seed=1
    )
    assert order == make_interleaved_order(
        IDS, repetitions=[0, 2], warmups=2, placements=PLACEMENTS, seed=1
    )
    assert len(order) == 3 * (2 + 2 * len(IDS))
    assert Counter((p, t.trial_id) for p, t in order).most_common(1)[0][1] == 1
    assert all(t.warmup for _, t in order[:6])
    for start in range(0, len(order), 3):
        group = order[start : start + 3]
        assert {p for p, _ in group} == set(PLACEMENTS)
        assert len({t.trial_id for _, t in group}) == 1
    orders = {tuple(p for p, _ in order[i : i + 3]) for i in range(0, len(order), 3)}
    assert len(orders) > 1, "placement order should vary between requests"
    assert {t.repetition for _, t in order if not t.warmup} == {0, 2}
    different = make_interleaved_order(
        IDS, repetitions=[0, 2], warmups=2, placements=PLACEMENTS, seed=2
    )
    assert different != order


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"request_ids": []}, "request IDs"),
        ({"request_ids": ["a", "a"]}, "request IDs"),
        ({"repetitions": []}, "repetition"),
        ({"repetitions": [-1]}, "non-negative"),
        ({"warmups": -1}, "warmups"),
        ({"placements": ["edge", "edge"]}, "placements"),
        ({"placements": ["moon"]}, "unknown placements"),
    ],
)
def test_order_rejects_invalid_parameters(kwargs: dict[str, Any], message: str) -> None:
    base: dict[str, Any] = {
        "request_ids": IDS,
        "repetitions": [0],
        "warmups": 0,
        "placements": PLACEMENTS,
        "seed": 1,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        make_interleaved_order(base.pop("request_ids"), **base)


def test_parameters_reproduce_their_schedule_and_detect_tampering() -> None:
    parameters = interleave_parameters(
        request_ids=IDS, repetitions=[0], warmups=1, placements=PLACEMENTS, seed=9
    )
    verify_interleave_parameters(parameters)
    schedule = placement_schedule(parameters)
    assert len(schedule) == 1 + len(IDS)
    tampered = {**parameters, "seed": 10}
    with pytest.raises(ValueError, match="reproduce"):
        verify_interleave_parameters(tampered)


def test_runner_prepares_stage_untimed_and_writes_profile_ready_runs(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    prepared: list[str] = []

    def prepare(stage: str) -> float:
        prepared.append(stage)
        log.append(f"prepare:{stage}")
        return 12.5

    summary = _run(tmp_path, policies=_policies(log), prepare=prepare)
    assert summary["attempts"] == 3 * (1 + 2 * len(IDS))
    for placement in PLACEMENTS:
        rows = _journal(tmp_path, placement)
        assert len(rows) == 1 + 2 * len(IDS)
        assert all(row["status"] == "ok" for row in rows)
        expected_stage = {"edge": "text", "distributed": "vision", "cloud": None}
        assert {row["prepared_stage"] for row in rows} == {expected_stage[placement]}
        assert {row["stage_prepare_ms"] for row in rows} == (
            {None} if placement == "cloud" else {12.5}
        )
        assert summary["placements"][placement]["measured_attempts"] == 2 * len(IDS)
    # Preparation happens immediately before its own placement's inference.
    for index, entry in enumerate(log):
        if entry == "prepare:text":
            assert log[index + 1].startswith("edge:")
        if entry == "prepare:vision":
            assert log[index + 1].startswith("distributed:")
    assert "cloud" not in "".join(prepared)
    profile = build_profile(
        run_dirs=[tmp_path / "run" / p for p in PLACEMENTS],
        specification=ROOT / "configs/profiles/local_fixture.json",
        output_dir=tmp_path / "profile",
    )
    rows = (tmp_path / "profile" / "profile.jsonl").read_text().splitlines()
    assert len(rows) == 3 * (1 + 2 * len(IDS))
    assert profile["specification_sha256"]


def test_profile_rejects_a_consistently_rewritten_interleave(tmp_path: Path) -> None:
    _run(tmp_path, policies=_policies([]))
    for placement in PLACEMENTS:
        directory = tmp_path / "run" / placement
        plan = json.loads((directory / "plan.json").read_text())
        old_hash = plan.pop("plan_sha256")
        plan["interleave"] = {**plan["interleave"], "seed": 1}
        plan["plan_sha256"] = hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (directory / "plan.json").write_text(json.dumps(plan))
        journal = directory / "requests.jsonl"
        journal.write_text(journal.read_text().replace(old_hash, plan["plan_sha256"]))
    with pytest.raises(ValueError, match="reproduce"):
        build_profile(
            run_dirs=[tmp_path / "run" / p for p in PLACEMENTS],
            specification=ROOT / "configs/profiles/local_fixture.json",
            output_dir=tmp_path / "profile",
        )


def test_kill_and_resume_completes_without_gaps_or_duplicates(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    policies = _policies(log)
    killer = StubPolicy(
        ExecutionMode.CLOUD, log, fail_on={4: KeyboardInterrupt("killed")}
    )
    policies["cloud"] = killer
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, policies=policies)
    before = {p: len(_journal(tmp_path, p)) for p in PLACEMENTS}
    assert sum(before.values()) < 3 * (1 + 2 * len(IDS))
    with pytest.raises(FileExistsError):
        _run(tmp_path, policies=_policies([]))
    _run(tmp_path, policies=_policies([]), resume=True)
    for placement in PLACEMENTS:
        rows = _journal(tmp_path, placement)
        trial_ids = [row["trial_id"] for row in rows]
        assert len(trial_ids) == len(set(trial_ids)) == 1 + 2 * len(IDS)
    positions = sorted(
        row["interleave_position"] for p in PLACEMENTS for row in _journal(tmp_path, p)
    )
    assert positions == list(range(1, 3 * (1 + 2 * len(IDS)) + 1))


def test_resume_rejects_a_changed_plan(tmp_path: Path) -> None:
    killer = _policies([])
    killer["edge"] = StubPolicy(
        ExecutionMode.EDGE, [], fail_on={2: KeyboardInterrupt("killed")}
    )
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, policies=killer)
    config, manifest = _setup()
    with pytest.raises(ValueError, match="interleaved plan differs"):
        run_interleaved_benchmark(
            config=config,
            manifest=manifest,
            policies=_policies([]),
            run_manifests={
                p: create_run_manifest(
                    config, {}, dataset_manifest_sha256=manifest.sha256
                )
                for p in PLACEMENTS
            },
            run_dir=tmp_path / "run",
            request_ids=IDS,
            repetitions=[1, 2],
            warmups=1,
            seed=1,
            resume=True,
        )


def test_local_cuda_failure_is_saved_then_stops_for_restart(tmp_path: Path) -> None:
    policies = _policies([])
    policies["edge"] = StubPolicy(
        ExecutionMode.EDGE, [], fail_on={3: RuntimeError("CUDA error: unknown error")}
    )
    with pytest.raises(DeviceContextFailure, match="restart"):
        _run(tmp_path, policies=policies)
    edge = _journal(tmp_path, "edge")
    assert edge[-1]["status"] == "model_error"
    assert edge[-1]["device_context_failure"] is True
    _run(tmp_path, policies=_policies([]), resume=True)
    edge = _journal(tmp_path, "edge")
    assert len(edge) == 1 + 2 * len(IDS)
    assert sum(row["status"] != "ok" for row in edge) == 1


def test_remote_errors_do_not_restart_and_count_toward_the_limit(
    tmp_path: Path,
) -> None:
    remote = RemoteInferenceError(
        "model_error", "CUDA error on the cloud", retryable=False
    )
    assert not is_device_context_failure(remote)
    assert is_device_context_failure(RuntimeError("CUDA error: unknown error"))
    assert not is_device_context_failure(RuntimeError("shape mismatch"))
    failing = {
        n: RemoteInferenceError("model_error", "boom", retryable=False)
        for n in range(1, 99)
    }
    policies = _policies([])
    policies["cloud"] = StubPolicy(ExecutionMode.CLOUD, [], fail_on=failing)
    policies["distributed"] = StubPolicy(
        ExecutionMode.DISTRIBUTED, [], fail_on=dict(failing)
    )
    policies["edge"] = StubPolicy(ExecutionMode.EDGE, [], fail_on=dict(failing))
    with pytest.raises(RuntimeError, match="3 consecutive failures"):
        _run(tmp_path, policies=policies, max_failures=3)


def test_restart_helper_resumes_only_after_device_failures() -> None:
    calls: list[list[str]] = []
    codes = iter([DEVICE_RESTART_EXIT_CODE, DEVICE_RESTART_EXIT_CODE, 0])

    def fake_run(argv: list[str]) -> int:
        calls.append(list(argv))
        return next(codes)

    assert run_with_device_restarts(["bench"], max_restarts=3, run=fake_run) == (0, 2)
    assert calls == [["bench"], ["bench", "--resume"], ["bench", "--resume"]]
    stopped = iter([DEVICE_RESTART_EXIT_CODE] * 5)
    assert run_with_device_restarts(
        ["bench"], max_restarts=1, run=lambda _: next(stopped)
    ) == (DEVICE_RESTART_EXIT_CODE, 1)
    assert run_with_device_restarts(["bench"], max_restarts=2, run=lambda _: 1) == (
        1,
        0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        run_with_device_restarts(["bench"], max_restarts=-1, run=lambda _: 0)


def test_cli_interleaves_fake_placements_over_real_loopback(tmp_path: Path) -> None:
    adapter = FakeModelAdapter()
    model_yaml = tmp_path / "fake32.yaml"
    model_yaml.write_text(
        (ROOT / "configs/model/fake.yaml")
        .read_text()
        .replace("max_new_tokens: 8", "max_new_tokens: 32")
    )
    from adaptive_vlm.config import model_contract_fingerprint

    probe = tmp_path / "probe.yaml"
    config_text = f"""schema_version: 1
name: interleaved-fake-loopback
mode: distributed
model_config: {model_yaml.as_posix()}
edge_hardware_config: {(ROOT / "configs/hardware/edge_rtx3050.yaml").as_posix()}
cloud_hardware_config: {(ROOT / "configs/hardware/local_test_cloud.yaml").as_posix()}
dataset:
  manifest_path: {(ROOT / "data/manifests/correctness.jsonl").as_posix()}
  revision: synthetic-fixtures-v1
runtime:
  seed: 42
  warmup_requests: 1
  measured_repetitions: 2
transport:
  host: 127.0.0.1
  port: PORT
  timeout_seconds: 5.0
  protocol_version: 1
artifacts_dir: {tmp_path.as_posix()}
"""
    probe.write_text(config_text.replace("PORT", "5050"))
    fingerprint = model_contract_fingerprint(load_experiment_config(probe).model)
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=fingerprint,
        inference_function=adapter.generate,
        tensor_inference_function=adapter.generate_from_payload,
    )
    server = CloudServer(
        host="127.0.0.1", port=0, timeout_seconds=5.0, processor=processor
    )
    try:
        server.start()
    except OSError as exc:  # pragma: no cover - sandbox dependent
        pytest.skip(f"loopback binding unavailable: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = tmp_path / "interleaved.yaml"
        config.write_text(config_text.replace("PORT", str(server.port)))
        split = tmp_path / "split.json"
        split.write_text(json.dumps({"training_request_ids": IDS[:4]}))
        argv = [
            "interleaved-benchmark",
            "--config",
            str(config),
            "--run-dir",
            str(tmp_path / "run"),
            "--split",
            str(split),
            "--split-name",
            "training",
            "--order-seed",
            "7",
            "--repetition-indices",
            "1",
        ]
        assert main(argv) == 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
    for placement in PLACEMENTS:
        rows = _journal(tmp_path, placement)
        assert len(rows) == 1 + 4
        assert all(row["status"] == "ok" for row in rows), rows
        manifest = json.loads(
            (tmp_path / "run" / placement / "run_manifest.json").read_text()
        )
        assert manifest["environment"]["execution_placement"]["interleaved"]
    edge = {r["trial_id"]: r["output_text"] for r in _journal(tmp_path, "edge")}
    cloud = {r["trial_id"]: r["output_text"] for r in _journal(tmp_path, "cloud")}
    assert edge == cloud
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                **json.loads(
                    (ROOT / "configs/profiles/local_fixture.json").read_text()
                ),
                "campaign_id": "interleaved-fake-loopback",
            }
        )
    )
    summary = build_profile(
        run_dirs=[tmp_path / "run" / p for p in PLACEMENTS],
        specification=spec,
        output_dir=tmp_path / "profile",
    )
    assert summary
