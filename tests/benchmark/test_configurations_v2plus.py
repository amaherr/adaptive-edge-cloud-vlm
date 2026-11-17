"""G2 v2+ configuration space: five configurations over a real loopback server."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.benchmark.heterogeneous_correctness import evaluate_rows
from adaptive_vlm.benchmark.interleaved import ALL_PLACEMENTS, PLACEMENTS
from adaptive_vlm.benchmark.profile import build_profile
from adaptive_vlm.campaign.grid import load_grid
from adaptive_vlm.campaign.subblock import benchmark_arguments
from adaptive_vlm.cli import main
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.scheduler.configuration import SPEC_BY_PLACEMENT
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer

ROOT = Path(__file__).resolve().parents[2]
IDS = [f"correctness-{index:03d}" for index in range(1, 7)]


def _journal(run: Path, placement: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run / placement / "requests.jsonl").read_text().splitlines()
    ]


def _config_text(tmp_path: Path) -> str:
    model_yaml = tmp_path / "fake32.yaml"
    model_yaml.write_text(
        (ROOT / "configs/model/fake.yaml")
        .read_text()
        .replace("max_new_tokens: 8", "max_new_tokens: 32")
    )
    return f"""schema_version: 1
name: interleaved-fake-v2plus
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


def test_five_configurations_run_interleaved_over_loopback(tmp_path: Path) -> None:
    adapter = FakeModelAdapter()
    text = _config_text(tmp_path)
    probe = tmp_path / "probe.yaml"
    probe.write_text(text.replace("PORT", "5050"))
    processor = CloudRequestProcessor(
        protocol_version=1,
        model_fingerprint=model_contract_fingerprint(
            load_experiment_config(probe).model
        ),
        inference_function=adapter.generate,
        tensor_inference_function=adapter.generate_from_payload,
        prompt_prepare_function=adapter.prepare_prompt,
        prepared_tensor_inference_function=adapter.generate_from_payload,
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
    run = tmp_path / "run"
    try:
        config = tmp_path / "interleaved.yaml"
        config.write_text(text.replace("PORT", str(server.port)))
        split = tmp_path / "split.json"
        split.write_text(json.dumps({"training_request_ids": IDS[:4]}))
        argv = [
            "interleaved-benchmark",
            "--config",
            str(config),
            "--run-dir",
            str(run),
            "--split",
            str(split),
            "--split-name",
            "training",
            "--order-seed",
            "7",
            "--repetition-indices",
            "0",
            "1",
            "--placements",
            *ALL_PLACEMENTS,
        ]
        assert main(argv) == 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert processor.pending_requests == 0  # no prompt-first state left behind

    rows = {placement: _journal(run, placement) for placement in ALL_PLACEMENTS}
    for placement, journal in rows.items():
        assert len(journal) == 1 + 8
        assert all(row["status"] == "ok" for row in journal), journal
        plan = json.loads((run / placement / "plan.json").read_text())
        spec = SPEC_BY_PLACEMENT[placement]
        assert plan["configuration_id"] == spec.id.value
        assert all(row["configuration_id"] == spec.id.value for row in journal)
        assert all(row["mode"] == spec.mode.value for row in journal)

    def outputs(placement: str) -> dict[str, str]:
        return {r["trial_id"]: r["output_text"] for r in rows[placement]}

    assert outputs("distributed_prompt_first") == outputs("distributed")
    for row in rows["distributed_prompt_first"]:
        assert row["metrics"]["prompt_round_trip_ms"] > 0
        assert row["metrics"]["distributed_pipeline"] == "prompt_first"
    bf16 = rows["distributed"][0]["metrics"]["projected_payload_bytes"]
    for row in rows["distributed_int8"]:
        metrics = row["metrics"]
        assert metrics["payload_format"] == "int8"
        assert metrics["projected_payload_bytes"] == 729 * 2048 + 729 * 2
        assert abs(metrics["projected_payload_bytes"] / bf16 - 0.5) < 0.01
        assert metrics["quantize_ms"] >= 0 and metrics["server_dequantize_ms"] >= 0
        assert (
            metrics["request_wire_bytes"]
            < 0.51 * (rows["distributed"][0]["metrics"]["request_wire_bytes"])
        )

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                **json.loads(
                    (ROOT / "configs/profiles/local_fixture.json").read_text()
                ),
                "campaign_id": "interleaved-fake-v2plus",
            }
        )
    )
    summary = build_profile(
        run_dirs=[run / p for p in ALL_PLACEMENTS],
        specification=spec_path,
        output_dir=tmp_path / "profile",
    )
    assert set(summary["coverage"]) == {
        SPEC_BY_PLACEMENT[p].id.value for p in ALL_PLACEMENTS
    }

    manifest = load_frozen_manifest(ROOT / "data/manifests/correctness.jsonl")
    subset = replace(
        manifest,
        items=tuple(item for item in manifest.items if item.request_id in IDS[:4]),
    )
    expected = {
        reference: {
            r["request_id"]: (tuple(r["output_token_ids"]), r["output_text"])
            for r in rows[placement]
            if not r["warmup"]
        }
        for reference, placement in (
            ("edge", "edge"),
            ("cloud", "cloud"),
            ("distributed", "distributed"),
            ("distributed_int8", "distributed_int8"),
        )
    }
    report = evaluate_rows(rows, subset, expected_outputs=expected)
    assert report["reference_output_mismatches"] == {}
    assert set(report["lossy_quality_mean"]) <= {"distributed_int8"}
    swapped = {**expected, "distributed_int8": expected["distributed"]}
    mismatched = evaluate_rows(rows, subset, expected_outputs=swapped)
    assert "distributed_int8" in mismatched["reference_output_mismatches"]


def _grid(tmp_path: Path, **changes: Any) -> Path:
    source = ROOT / "configs/profiles/rehearsal_grid_v2.json"
    payload = json.loads(source.read_text())
    for key in ("protocol_path", "split_path", "experiment_config_path"):
        payload[key] = str((source.parent / payload[key]).resolve())
    payload["edge_load_levels_path"] = str(
        (source.parent / payload["edge_load_levels_path"]).resolve()
    )
    for task in payload["extra_tasks"].values():
        for key in ("split_path", "experiment_config_path"):
            task[key] = str((source.parent / task[key]).resolve())
    payload.update(changes)
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(payload))
    return path


def test_grid_configurations_set_placements_and_counts(tmp_path: Path) -> None:
    default = load_grid(_grid(tmp_path))
    assert default.placements == PLACEMENTS
    block = default.execution_order()[0]
    assert "--placements" not in benchmark_arguments(
        default, block, tmp_path / "run", resume=False
    )
    five = load_grid(
        _grid(
            tmp_path,
            configurations=[
                "distributed/projected/int8",
                "edge/full",
                "cloud/raw_image",
                "distributed/projected/synchronous",
                "distributed/projected/prompt_first",
            ],
        )
    )
    assert five.placements == ALL_PLACEMENTS  # canonical order
    assert five.configurations[-1] == "distributed/projected/int8"
    arguments = benchmark_arguments(five, block, tmp_path / "run", resume=False)
    assert arguments[arguments.index("--placements") + 1 :] == list(ALL_PLACEMENTS)
    splits = {name: five.load_split(name) for name in five.tasks}
    assert five.expected_attempts(splits)["measured"] == (
        default.expected_attempts(splits)["measured"] * 5 // 3
    )
    with pytest.raises(ValueError, match="unknown configuration"):
        load_grid(_grid(tmp_path, configurations=["edge/full", "edge/int4"]))
    with pytest.raises(ValueError, match="must include"):
        load_grid(_grid(tmp_path, configurations=["edge/full", "cloud/raw_image"]))
