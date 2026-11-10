"""Fail-closed checks for the explicitly selected physical v2 criterion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.heterogeneous import (
    PROTOCOL,
    evaluate_rows,
    implementation_hashes,
    inference_hashes,
    semantic_manifest_sha256,
    verify_reference_evidence,
)
from adaptive_vlm.benchmark.profile import build_profile
from adaptive_vlm.benchmark.runner import run_benchmark
from adaptive_vlm.config import ExecutionMode, load_experiment_config
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.data.manifest import FrozenManifest, load_frozen_manifest
from adaptive_vlm.execution.base import ExecutionPolicy

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PlacementPolicy(ExecutionPolicy):
    def __init__(self, mode: ExecutionMode) -> None:
        self._mode = mode

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def infer(self, request: InferenceRequest) -> InferenceResult:
        token = {"edge": 1, "cloud": 2, "distributed": 3}[self.mode.value]
        common = {
            "prompt_tokens": 12,
            "prompt_prefill_ms": 0.000001,
            "decode_ms": 0.000001,
            "output_tokens": 1,
            "image_width": 16,
            "image_height": 16,
        }
        if self.mode is ExecutionMode.EDGE:
            metrics = {**common, "end_to_end_ms": 0.000001}
        else:
            metrics = {
                **{f"server_model_{k}": v for k, v in common.items()},
                "network_round_trip_ms": 0.000002,
                "server_total_ms": 0.000001,
                "request_wire_bytes": 3_000_000,
                "server_request_wire_bytes": 3_000_000,
                "response_wire_bytes": 40,
            }
            if self.mode is ExecutionMode.CLOUD:
                metrics.update(
                    image_bytes=request.image_path.stat().st_size,
                    image_read_ms=0.000001,
                )
            else:
                metrics.update(
                    image_width=16,
                    image_height=16,
                    projected_payload_bytes=729 * 2048 * 2,
                    edge_vision_end_to_end_ms=0.000001,
                    tensor_serialization_ms=0.000001,
                )
        return InferenceResult(
            request.request_id, f"output-{self.mode.value}", (token,), metrics
        )


@pytest.fixture
def evidence(tmp_path: Path) -> tuple[FrozenManifest, dict[str, Path], Path, Path]:
    config = load_experiment_config(
        ROOT / "configs/experiments/chunk6_local_campaign.yaml"
    )
    original = config.dataset.manifest_path
    records = [json.loads(line) for line in original.read_text().splitlines()]
    records = [r for r in records if r["scoring"] != "none"][:4]
    for row in records:
        row["image_path"] = str((original.parent / row["image_path"]).resolve())
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    config = replace(
        config, dataset=replace(config.dataset, manifest_path=manifest_path)
    )
    frozen = load_frozen_manifest(manifest_path)
    dirs = {}
    for mode in ExecutionMode:
        directory = tmp_path / "runs" / mode.value
        run_benchmark(
            config=config,
            manifest=frozen,
            policy=PlacementPolicy(mode),
            run_manifest=create_run_manifest(
                config, {}, dataset_manifest_sha256=frozen.sha256
            ),
            run_dir=directory,
            warmups=1,
            repetitions=3,
            pipeline="synchronous",
        )
        dirs[mode.value] = directory
    plan = json.loads((dirs["edge"] / "plan.json").read_text())
    base = tmp_path / "reference"
    script = base / "producer.py"
    script.parent.mkdir()
    script.write_text("# unit-test evidence producer\n")
    artifacts = {"producer.py": digest(script)}
    hosts = {}
    for host, token in [("edge", 1), ("cloud", 2)]:
        report_name = f"{host}/report.json"
        journal_name = f"{host}/requests.jsonl"
        rows = [
            {
                "request_id": i.request_id,
                "image_sha256": i.image_sha256,
                "comparison": {
                    "projected_tensor_equal": True,
                    "token_ids_equal": True,
                    "text_equal": True,
                },
                "reference_tokens": [token],
                "candidate_tokens": [token],
                "reference_text": f"output-{host}",
                "candidate_text": f"output-{host}",
            }
            for i in frozen.items
        ]
        save(
            base / report_name,
            {
                "passed": True,
                "completed": len(rows),
                "requests": len(rows),
                "mismatches": 0,
                "model_contract_sha256": plan["model_contract_sha256"],
                "input_semantic_sha256": semantic_manifest_sha256(frozen),
                "manifest_sha256": frozen.sha256,
                "source_files_sha256": implementation_hashes(),
                "script_sha256": digest(script),
                "device": config.to_dict()[host].get("gpu") or f"unit-{host}",
            },
        )
        (base / journal_name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        hosts[host] = {"report": report_name, "journal": journal_name}
        artifacts.update(
            {name: digest(base / name) for name in [report_name, journal_name]}
        )
    control = {}
    for key, filename in [("plan", "plan.json"), ("journal", "requests.jsonl")]:
        name = "mixed/" + filename
        target = base / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((dirs["distributed"] / filename).read_bytes())
        artifacts[name] = digest(target)
        control[key] = name
    index = base / "index.json"
    save(
        index,
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "dataset_manifest_sha256": frozen.sha256,
            "model_contract_sha256": plan["model_contract_sha256"],
            "input_semantic_sha256": semantic_manifest_sha256(frozen),
            "reference_manifest_sha256": frozen.sha256,
            "reference_script": "producer.py",
            "artifacts_sha256": artifacts,
            "hosts": hosts,
            "distributed_control": control,
            "inference_source_sha256": inference_hashes(),
            "runtime_versions": {
                host: {"python": "unit-runtime", "packages": {"torch": "unit"}}
                for host in ("edge", "cloud")
            },
        },
    )
    spec = json.loads((ROOT / "configs/profiles/phase2_local_fixture.json").read_text())
    network = tmp_path / "network.json"
    save(network, {"measured": True})
    spec["network_state"].update(
        controlled=True,
        uplink_bps=1_000_000,
        downlink_bps=1_000_000,
        rtt_ms=5,
        measurement_evidence_path=network.name,
        measurement_evidence_sha256=digest(network),
    )
    spec.update(
        correctness_protocol=PROTOCOL,
        same_device_reference_evidence_path="reference/index.json",
        same_device_reference_evidence_sha256=digest(index),
    )
    specification = tmp_path / "spec.json"
    save(specification, spec)
    return frozen, dirs, index, specification


def checked_proof(evidence: tuple) -> dict:
    frozen, dirs, index, _ = evidence
    plan = json.loads((dirs["edge"] / "plan.json").read_text())
    return verify_reference_evidence(
        index, digest(index), frozen, plan["model_contract_sha256"]
    )


def raw_rows(dirs: dict[str, Path]) -> dict[str, list[dict]]:
    return {
        mode: [
            json.loads(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
        ]
        for mode, directory in dirs.items()
    }


def test_explicit_v2_passes_without_claiming_cross_gpu_identity(
    evidence: tuple, tmp_path: Path
) -> None:
    frozen, dirs, _, spec = evidence
    proof = checked_proof(evidence)
    result = evaluate_rows(
        raw_rows(dirs), frozen, expected_outputs=proof["_expected_outputs"]
    )
    assert result["passed"] and result["task_quality_equal"]
    assert len(result["cross_gpu_output_disagreement_request_ids"]) == len(frozen.items)
    summary = build_profile(
        run_dirs=list(dirs.values()), specification=spec, output_dir=tmp_path / "v2"
    )
    assert summary["controlled_network_gate_passed"]
    assert not summary["legacy_strict_gate_passed"]
    assert not summary["complete_triplets_output_tokens_equal"]
    original = json.loads(spec.read_text())
    original.pop("correctness_protocol")
    strict = tmp_path / "strict.json"
    save(strict, original)
    assert not build_profile(
        run_dirs=list(dirs.values()),
        specification=strict,
        output_dir=tmp_path / "strict",
    )["local_gate_passed"]


@pytest.mark.parametrize(
    "change", ["quality", "unscored", "local_tokens", "reference", "coverage"]
)
def test_quality_repeatability_and_reference_changes_block(
    evidence: tuple, change: str
) -> None:
    frozen, dirs, _, _ = evidence
    proof = checked_proof(evidence)
    rows = raw_rows(dirs)
    measured = [r for r in rows["edge"] if not r["warmup"]]
    first = measured[0]
    if change == "quality":
        first["quality_score"] = 0.5
    elif change == "unscored":
        first["quality_score"] = None
    elif change in {"local_tokens", "reference"}:
        affected = (
            [first]
            if change == "local_tokens"
            else [r for r in measured if r["request_id"] == first["request_id"]]
        )
        for row in affected:
            row["output_token_ids"] = [999]
    else:
        rows["edge"] = [
            r
            for r in rows["edge"]
            if r["warmup"]
            or r["request_id"] != first["request_id"]
            or r["repetition"] == first["repetition"]
        ]
    assert not evaluate_rows(rows, frozen, expected_outputs=proof["_expected_outputs"])[
        "passed"
    ]


def test_unknown_placement_and_request_rejected(evidence: tuple) -> None:
    frozen, dirs, _, _ = evidence
    rows = raw_rows(dirs)
    with pytest.raises(ValueError, match="all three"):
        evaluate_rows({"edge": rows["edge"]}, frozen)
    rows["edge"][1]["request_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown request"):
        evaluate_rows(rows, frozen)


@pytest.mark.parametrize(
    "change",
    [
        "artifact",
        "contract",
        "source",
        "report",
        "row",
        "control",
        "script",
        "gpu",
        "torn",
        "runtime",
    ],
)
def test_bound_reference_proof_fails_closed(evidence: tuple, change: str) -> None:
    frozen, dirs, path, _ = evidence
    index = json.loads(path.read_text())
    devices = None
    if change == "artifact":
        (path.parent / "edge/requests.jsonl").write_text("changed")
    elif change == "contract":
        index["input_semantic_sha256"] = "0" * 64
    elif change == "source":
        index["inference_source_sha256"] = {}
    elif change == "runtime":
        index.pop("runtime_versions")
    elif change == "script":
        index["reference_script"] = "missing.py"
    elif change == "gpu":
        devices = {"edge": "different GPU"}
    else:
        name = "edge/report.json" if change == "report" else "edge/requests.jsonl"
        if change == "control":
            name = "mixed/plan.json"
        target = path.parent / name
        if change == "report":
            value = json.loads(target.read_text())
            value["passed"] = False
            save(target, value)
        elif change == "control":
            value = json.loads(target.read_text())
            value["mode"] = "edge"
            save(target, value)
        elif change == "torn":
            target.write_bytes(target.read_bytes().rstrip())
        else:
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            rows[0]["comparison"]["projected_tensor_equal"] = False
            target.write_text("".join(json.dumps(row) + "\n" for row in rows))
        index["artifacts_sha256"][name] = digest(target)
    save(path, index)
    plan = json.loads((dirs["edge"] / "plan.json").read_text())
    with pytest.raises(ValueError):
        verify_reference_evidence(
            path,
            digest(path),
            frozen,
            plan["model_contract_sha256"],
            expected_devices=devices,
        )


def test_v2_requires_evidence_and_known_protocol(
    evidence: tuple, tmp_path: Path
) -> None:
    _, dirs, _, path = evidence
    spec = json.loads(path.read_text())
    spec.pop("same_device_reference_evidence_path")
    save(path, spec)
    with pytest.raises(ValueError, match="requires same-device"):
        build_profile(
            run_dirs=list(dirs.values()),
            specification=path,
            output_dir=tmp_path / "missing",
        )
    spec["correctness_protocol"] = "unknown"
    save(path, spec)
    with pytest.raises(ValueError, match="unsupported correctness"):
        build_profile(
            run_dirs=list(dirs.values()),
            specification=path,
            output_dir=tmp_path / "unknown",
        )
