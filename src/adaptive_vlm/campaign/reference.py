"""Same-device correctness reference for a campaign manifest (G2 and G2 v2).

The heterogeneous-GPU correctness protocol (``physical-heterogeneous-v2``)
compares each placement's outputs with per-GPU references instead of across
GPUs. Its evidence bundle, checked by ``verify_reference_evidence``, needs:

1. on **each** GPU, every manifest request run through the pinned monolithic
   path and the decomposed path, with bit-identical projected tensors, output
   tokens and text (``same_device_proof``);
2. a distributed control run over the full manifest (1 warmup, 3 repetitions)
   whose outputs repeat exactly (``adaptive-vlm benchmark --mode
   distributed``); G2 v2+ adds a second control with ``--payload int8``,
   whose outputs the lossy int8 configuration must reproduce;
3. both hosts' runtime package versions;
4. an index binding all of it, plus the script that produced the proofs, by
   SHA-256 (``build_bundle``).

The proofs bind the current source hashes, so they must be produced from the
same commit the campaign runs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import append_jsonl, write_json_atomic
from adaptive_vlm.benchmark.heterogeneous_correctness import (
    PROTOCOL,
    implementation_hashes,
    inference_hashes,
    semantic_manifest_sha256,
    verify_reference_evidence,
)
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.data.manifest import load_frozen_manifest

COMPARED = ("projected_tensor_equal", "token_ids_equal", "text_equal")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def same_device_proof(
    *,
    config_path: Path,
    output: Path,
    script_path: Path,
    adapter: Any,
    device_name: str,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Compare monolithic and decomposed runs for every manifest request."""

    config = load_experiment_config(config_path)
    manifest = load_frozen_manifest(
        config.dataset.manifest_path, expected_revision=config.dataset.revision
    )
    output.mkdir(parents=True, exist_ok=False)
    total = len(manifest.items)
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "Same-device monolithic/decomposed parity over the full manifest",
        "started_at_utc": _now(),
        "input_semantic_sha256": semantic_manifest_sha256(manifest),
        "manifest_sha256": manifest.sha256,
        "model_contract_sha256": model_contract_fingerprint(config.model),
        "script_sha256": _sha256(script_path),
        "source_files_sha256": implementation_hashes(),
        "device": device_name,
        "requests": total,
        "completed": 0,
        "mismatches": 0,
        "passed": False,
    }
    write_json_atomic(output / "status.json", report)
    for index, item in enumerate(manifest.items, 1):
        log(f"[{index}/{total}] {item.request_id}")
        request = item.to_request()
        reference = adapter.run_monolithic(request)
        candidate = adapter.run_decomposed(request)
        comparison = adapter.compare_runs(reference, candidate)
        append_jsonl(
            output / "requests.jsonl",
            {
                "request_id": item.request_id,
                "image_sha256": item.image_sha256,
                "comparison": comparison,
                "reference_tokens": list(reference.result.output_token_ids),
                "candidate_tokens": list(candidate.result.output_token_ids),
                "reference_text": reference.result.output_text,
                "candidate_text": candidate.result.output_text,
                "reference_metrics": dict(reference.result.metrics),
                "candidate_metrics": dict(candidate.result.metrics),
            },
        )
        report["completed"] = index
        report["mismatches"] += int(not all(comparison.get(k) for k in COMPARED))
        report["updated_at_utc"] = _now()
        write_json_atomic(output / "status.json", report)
    report["passed"] = report["completed"] == total and report["mismatches"] == 0
    report["finished_at_utc"] = _now()
    write_json_atomic(output / "report.json", report)
    write_json_atomic(output / "status.json", report)
    return report


def build_bundle(
    *,
    config_path: Path,
    edge: Path,
    cloud: Path,
    control: Path,
    edge_runtime: Path,
    cloud_runtime: Path,
    script_path: Path,
    output: Path,
    int8_control: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Copy and bind the evidence, then verify it exactly as preflight will."""

    config = load_experiment_config(config_path)
    manifest = load_frozen_manifest(
        config.dataset.manifest_path, expected_revision=config.dataset.revision
    )
    contract = model_contract_fingerprint(config.model)
    output.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}

    def copy(source: Path, name: str) -> str:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = _sha256(target)
        return name

    hosts = {}
    for host, source in (("edge", edge), ("cloud", cloud)):
        report = json.loads((source / "report.json").read_text())
        if not report.get("passed"):
            raise ValueError(f"{host} same-device proof did not pass")
        hosts[host] = {
            "report": copy(source / "report.json", f"{host}/report.json"),
            "journal": copy(source / "requests.jsonl", f"{host}/requests.jsonl"),
        }
    script = copy(script_path, script_path.name)
    distributed = {
        "plan": copy(control / "plan.json", "control/plan.json"),
        "journal": copy(control / "requests.jsonl", "control/requests.jsonl"),
    }
    copy(control / "run_manifest.json", "control/run_manifest.json")
    controls = {}
    if int8_control is not None:
        controls["int8"] = {
            "plan": copy(int8_control / "plan.json", "control-int8/plan.json"),
            "journal": copy(
                int8_control / "requests.jsonl", "control-int8/requests.jsonl"
            ),
        }
        copy(int8_control / "run_manifest.json", "control-int8/run_manifest.json")
    edge_report = json.loads((edge / "report.json").read_text())
    index = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "created_at_utc": _now(),
        "dataset_manifest_sha256": manifest.sha256,
        "model_contract_sha256": contract,
        "input_semantic_sha256": semantic_manifest_sha256(manifest),
        "reference_manifest_sha256": edge_report["manifest_sha256"],
        "inference_source_sha256": inference_hashes(),
        "reference_script": script,
        "hosts": hosts,
        "distributed_control": distributed,
        **({"distributed_controls": controls} if controls else {}),
        "runtime_versions": {
            "edge": json.loads(edge_runtime.read_text()),
            "cloud": json.loads(cloud_runtime.read_text()),
        },
        "artifacts_sha256": hashes,
    }
    index_path = output / "index.json"
    write_json_atomic(index_path, index)
    devices = {
        host: gpu
        for host in ("edge", "cloud")
        if isinstance(gpu := config.to_dict().get(host, {}).get("gpu"), str)
    }
    proof = verify_reference_evidence(
        index_path,
        _sha256(index_path),
        manifest,
        contract,
        expected_devices=devices or None,
    )
    proof.pop("_expected_outputs")
    return index_path, proof
