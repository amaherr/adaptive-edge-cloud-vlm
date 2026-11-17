"""Same-device reference proofs and bundle, checked by the real verifier."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from adaptive_vlm.artifacts import create_run_manifest
from adaptive_vlm.benchmark.runner import run_benchmark
from adaptive_vlm.campaign.reference import build_bundle, same_device_proof
from adaptive_vlm.config import ExecutionMode, load_experiment_config
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.base import ExecutionPolicy

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/local_campaign.yaml"
SCRIPT = ROOT / "scripts/campaign/build_g2v2_reference.py"


def _tokens(request: InferenceRequest) -> tuple[int, ...]:
    return tuple(ord(c) % 97 for c in request.request_id[-4:])


class StubAdapter:
    def __init__(self, mismatch: str | None = None) -> None:
        self.mismatch = mismatch

    def _run(self, request: InferenceRequest, salt: str = "") -> Any:
        return SimpleNamespace(
            result=InferenceResult(
                request.request_id,
                "answer" + salt,
                _tokens(request),
                {"decode_ms": 1.0},
            )
        )

    def run_monolithic(self, request: InferenceRequest) -> Any:
        return self._run(request)

    def run_decomposed(self, request: InferenceRequest) -> Any:
        salt = "!" if request.request_id == self.mismatch else ""
        return self._run(request, salt)

    def compare_runs(self, reference: Any, candidate: Any) -> dict[str, bool]:
        same = reference.result.output_text == candidate.result.output_text
        return {
            "projected_tensor_equal": True,
            "token_ids_equal": True,
            "text_equal": same,
        }


class StubDistributed(ExecutionPolicy):
    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.DISTRIBUTED

    def infer(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(request.request_id, "answer", _tokens(request), {})


def _evidence(tmp_path: Path, mismatch: str | None = None) -> dict[str, Path]:
    config = load_experiment_config(CONFIG)
    manifest = load_frozen_manifest(config.dataset.manifest_path)
    paths = {}
    for host, device in (
        ("edge", config.to_dict()["edge"]["gpu"]),
        ("cloud", config.to_dict()["cloud"].get("gpu") or "stub cloud GPU"),
    ):
        report = same_device_proof(
            config_path=CONFIG,
            output=tmp_path / host,
            script_path=SCRIPT,
            adapter=StubAdapter(mismatch if host == "cloud" else None),
            device_name=device,
            log=lambda message: None,
        )
        assert report["completed"] == len(manifest.items)
        paths[host] = tmp_path / host
        runtime = tmp_path / f"{host}-runtime.json"
        runtime.write_text(
            json.dumps({"python": "3.11", "packages": {"torch": "2.7.1"}})
        )
        paths[f"{host}_runtime"] = runtime
    run_benchmark(
        config=config,
        manifest=manifest,
        policy=StubDistributed(),
        run_manifest=create_run_manifest(
            config, {}, dataset_manifest_sha256=manifest.sha256
        ),
        run_dir=tmp_path / "control",
        warmups=1,
        repetitions=3,
        pipeline="synchronous",
    )
    paths["control"] = tmp_path / "control"
    return paths


def test_bundle_from_proofs_passes_the_real_verifier(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    index, proof = build_bundle(
        config_path=CONFIG,
        edge=paths["edge"],
        cloud=paths["cloud"],
        control=paths["control"],
        edge_runtime=paths["edge_runtime"],
        cloud_runtime=paths["cloud_runtime"],
        script_path=SCRIPT,
        output=tmp_path / "bundle",
    )
    assert proof["verified"] and proof["requests_per_gpu"] == 12
    saved = json.loads(index.read_text())
    assert saved["reference_script"] == SCRIPT.name
    assert SCRIPT.name in saved["artifacts_sha256"]


def test_a_single_mismatch_fails_the_proof_and_the_bundle(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG)
    first = load_frozen_manifest(config.dataset.manifest_path).items[3].request_id
    paths = _evidence(tmp_path, mismatch=first)
    report = json.loads((paths["cloud"] / "report.json").read_text())
    assert report["mismatches"] == 1 and report["passed"] is False
    with pytest.raises(ValueError, match="cloud same-device proof did not pass"):
        build_bundle(
            config_path=CONFIG,
            edge=paths["edge"],
            cloud=paths["cloud"],
            control=paths["control"],
            edge_runtime=paths["edge_runtime"],
            cloud_runtime=paths["cloud_runtime"],
            script_path=SCRIPT,
            output=tmp_path / "bundle",
        )


class StubInt8(StubDistributed):
    def infer(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(request.request_id, "answer8", _tokens(request), {})


def test_bundle_with_an_int8_control_binds_its_outputs(tmp_path: Path) -> None:
    from adaptive_vlm.benchmark.heterogeneous_correctness import (
        evaluate_rows,
        verify_reference_evidence,
    )

    paths = _evidence(tmp_path)
    config = load_experiment_config(CONFIG)
    manifest = load_frozen_manifest(config.dataset.manifest_path)
    run_benchmark(
        config=config,
        manifest=manifest,
        policy=StubInt8(),
        run_manifest=create_run_manifest(
            config, {}, dataset_manifest_sha256=manifest.sha256
        ),
        run_dir=tmp_path / "control-int8",
        warmups=1,
        repetitions=3,
        pipeline="synchronous",
        payload="int8",
    )
    common = {
        "config_path": CONFIG,
        "edge": paths["edge"],
        "cloud": paths["cloud"],
        "control": paths["control"],
        "edge_runtime": paths["edge_runtime"],
        "cloud_runtime": paths["cloud_runtime"],
        "script_path": SCRIPT,
    }
    index, proof = build_bundle(
        **common, output=tmp_path / "bundle", int8_control=tmp_path / "control-int8"
    )
    assert proof["verified"]
    import hashlib

    outputs = verify_reference_evidence(
        index,
        hashlib.sha256(index.read_bytes()).hexdigest(),
        manifest,
        json.loads(index.read_text())["model_contract_sha256"],
    )["_expected_outputs"]
    first = manifest.items[0].request_id
    assert outputs["distributed_int8"][first][1] == "answer8"
    assert outputs["distributed"][first][1] == "answer"

    def rows(text: str) -> list[dict[str, Any]]:
        return [
            {
                "request_id": item.request_id,
                "warmup": False,
                "repetition": repetition,
                "status": "ok",
                "quality_score": 1.0,
                "output_token_ids": list(outputs["distributed"][item.request_id][0]),
                "output_text": text,
            }
            for item in manifest.items
            for repetition in (0, 1)
        ]

    runs = {
        "edge": rows("answer"),
        "cloud": rows("answer"),
        "distributed": rows("answer"),
        "distributed_prompt_first": rows("answer"),
        "distributed_int8": rows("answer8"),
    }
    expected = {
        **outputs,
        "edge": outputs["distributed"],
        "cloud": outputs["distributed"],
    }
    assert (
        evaluate_rows(runs, manifest, expected_outputs=expected)[
            "reference_output_mismatches"
        ]
        == {}
    )
    wrong = evaluate_rows(
        {**runs, "distributed_int8": rows("answer")},
        manifest,
        expected_outputs=expected,
    )
    assert set(wrong["reference_output_mismatches"]) == {"distributed_int8"}

    # A bf16 control cannot stand in for the int8 one.
    with pytest.raises(ValueError, match="control contract differs"):
        build_bundle(
            **common, output=tmp_path / "bundle2", int8_control=paths["control"]
        )
