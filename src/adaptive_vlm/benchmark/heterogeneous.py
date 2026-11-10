"""Exact implementation checks and task-score control for heterogeneous GPUs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adaptive_vlm import timing
from adaptive_vlm.benchmark.runner import make_schedule
from adaptive_vlm.data.manifest import FrozenManifest
from adaptive_vlm.models import moondream

PROTOCOL = "physical-heterogeneous-v2"


def semantic_manifest_sha256(manifest: FrozenManifest) -> str:
    items = [
        {
            "request_id": item.request_id,
            "image_sha256": item.image_sha256,
            "prompt": item.prompt,
            "references": list(item.references),
            "scoring": item.scoring,
            "source": item.source,
            "revision": item.revision,
        }
        for item in manifest.items
    ]
    return hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def implementation_hashes() -> dict[str, str]:
    result = {}
    for name, module in (
        ("src/adaptive_vlm/models/moondream.py", moondream),
        ("src/adaptive_vlm/timing.py", timing),
    ):
        filename = module.__file__
        if filename is None:
            raise ValueError("missing reference implementation filename")
        result[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return result


def inference_hashes() -> dict[str, str]:
    filename = timing.__file__
    if filename is None:
        raise ValueError("missing inference source filename")
    package = Path(filename).resolve().parent
    excluded = {"benchmark/profile.py", "benchmark/heterogeneous.py"}
    return {
        "src/adaptive_vlm/" + str(path.relative_to(package)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(package.rglob("*.py"))
        if str(path.relative_to(package)) not in excluded
    }


def runtime_versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "numpy",
                "pillow",
                "safetensors",
                "tokenizers",
                "accelerate",
                "huggingface-hub",
            )
        },
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected reference evidence object: {path}")
    return value


def _contained(base: Path, name: str) -> Path:
    path = (base / name).resolve(strict=True)
    if Path(name).is_absolute() or not path.is_relative_to(base.resolve()):
        raise ValueError("reference evidence paths must remain inside their bundle")
    return path


def verify_reference_evidence(
    path: Path,
    expected_sha256: str,
    manifest: FrozenManifest,
    model_contract_sha256: str,
    *,
    expected_devices: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify full-manifest same-GPU evidence, including every committed row."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("same-device reference index hash differs")
    index = _json(path)
    if (
        index.get("schema_version") != 1
        or index.get("protocol") != PROTOCOL
        or index.get("dataset_manifest_sha256") != manifest.sha256
        or index.get("input_semantic_sha256") != semantic_manifest_sha256(manifest)
        or index.get("model_contract_sha256") != model_contract_sha256
    ):
        raise ValueError("same-device reference contract differs")
    hashes = index.get("artifacts_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("missing same-device reference artifact hashes")
    for name, expected in hashes.items():
        source = _contained(path.parent, name)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"changed same-device reference artifact: {name}")
    script = index.get("reference_script")
    if not isinstance(script, str) or script not in hashes:
        raise ValueError("missing bound reference script")
    if index.get("inference_source_sha256") != inference_hashes():
        raise ValueError("reference inference implementation differs")
    runtimes = index.get("runtime_versions", {})
    if not isinstance(runtimes, dict) or set(runtimes) != {"edge", "cloud"}:
        raise ValueError("missing reference runtime versions")
    for runtime in runtimes.values():
        if (
            not isinstance(runtime, dict)
            or not isinstance(runtime.get("python"), str)
            or not isinstance(runtime.get("packages"), dict)
            or not runtime["packages"]
        ):
            raise ValueError("incomplete reference runtime versions")
    hosts = index.get("hosts", {})
    if set(hosts) != {"edge", "cloud"}:
        raise ValueError("reference evidence requires both physical GPUs")
    expected_ids = [item.request_id for item in manifest.items]
    images = {item.request_id: item.image_sha256 for item in manifest.items}
    devices: dict[str, str] = {}
    expected_outputs: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {}
    for host, entry in hosts.items():
        for key in ("report", "journal"):
            if entry[key] not in hashes:
                raise ValueError(f"unbound {host} reference {key}")
        report = _json(_contained(path.parent, entry["report"]))
        if (
            report.get("passed") is not True
            or report.get("completed") != len(expected_ids)
            or report.get("requests") != len(expected_ids)
            or report.get("mismatches") != 0
            or report.get("input_semantic_sha256") != index["input_semantic_sha256"]
            or report.get("manifest_sha256") != index.get("reference_manifest_sha256")
            or report.get("model_contract_sha256") != model_contract_sha256
            or report.get("source_files_sha256") != implementation_hashes()
            or report.get("script_sha256") != hashes[script]
        ):
            raise ValueError(f"incomplete or incompatible {host} reference proof")
        device = report.get("device")
        if not isinstance(device, str) or not device:
            raise ValueError(f"missing {host} reference GPU")
        if (
            expected_devices
            and host in expected_devices
            and device != expected_devices[host]
        ):
            raise ValueError(f"{host} reference GPU differs")
        devices[host] = device
        content = _contained(path.parent, entry["journal"]).read_bytes()
        if not content.endswith(b"\n"):
            raise ValueError(f"torn {host} reference journal")
        rows = [json.loads(line) for line in content.splitlines()]
        if [row.get("request_id") for row in rows] != expected_ids:
            raise ValueError(f"{host} reference request coverage/order differs")
        expected_outputs[host] = {}
        for row in rows:
            expected_outputs[host][row["request_id"]] = (
                tuple(row["reference_tokens"]),
                row["reference_text"],
            )
            comparison = row.get("comparison", {})
            if (
                row.get("image_sha256") != images[row["request_id"]]
                or any(
                    comparison.get(key) is not True
                    for key in (
                        "projected_tensor_equal",
                        "token_ids_equal",
                        "text_equal",
                    )
                )
                or row.get("reference_tokens") != row.get("candidate_tokens")
                or row.get("reference_text") != row.get("candidate_text")
            ):
                raise ValueError(f"{host} same-device path equivalence failed")
    control = index.get("distributed_control", {})
    for key in ("plan", "journal"):
        if control.get(key) not in hashes:
            raise ValueError("missing bound mixed-placement control")
    plan = _json(_contained(path.parent, control["plan"]))
    saved_hash = plan.pop("plan_sha256", None)
    actual_hash = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if saved_hash != actual_hash or (
        plan.get("mode") != "distributed"
        or plan.get("model_contract_sha256") != model_contract_sha256
        or plan.get("dataset_manifest_sha256") != manifest.sha256
    ):
        raise ValueError("mixed-placement control contract differs")
    schedule = [
        trial.to_dict()
        for trial in make_schedule(
            manifest, warmups=1, repetitions=3, seed=plan["config"]["runtime"]["seed"]
        )
    ]
    if plan["schedule"] != schedule:
        raise ValueError("mixed-placement control schedule differs")
    content = _contained(path.parent, control["journal"]).read_bytes()
    if not content.endswith(b"\n"):
        raise ValueError("torn mixed-placement control journal")
    controls = [json.loads(line) for line in content.splitlines()]
    if len(controls) != len(schedule):
        raise ValueError("incomplete mixed-placement control")
    mixed: dict[str, set[tuple[tuple[int, ...], str]]] = defaultdict(set)
    for row, trial in zip(controls, schedule, strict=True):
        if (
            any(row.get(key) != value for key, value in trial.items())
            or row.get("plan_sha256") != saved_hash
            or row.get("model_contract_sha256") != model_contract_sha256
            or row.get("dataset_manifest_sha256") != manifest.sha256
            or row.get("config_sha256") != plan["config_sha256"]
            or row.get("mode") != "distributed"
            or row.get("status") != "ok"
            or row.get("image_sha256") != images.get(row.get("request_id"))
        ):
            raise ValueError("mixed-placement control row differs")
        if not row["warmup"]:
            mixed[row["request_id"]].add(
                (tuple(row["output_token_ids"]), row["output_text"])
            )
    if set(mixed) != set(expected_ids) or any(len(v) != 1 for v in mixed.values()):
        raise ValueError("mixed-placement control is not repeatable/complete")
    expected_outputs["distributed"] = {
        key: next(iter(value)) for key, value in mixed.items()
    }
    return {
        "_expected_outputs": expected_outputs,
        "verified": True,
        "requests_per_gpu": len(expected_ids),
        "devices": devices,
        "evidence_sha256": expected_sha256,
    }


def evaluate_rows(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest: FrozenManifest,
    *,
    expected_outputs: Mapping[str, Mapping[str, tuple[tuple[int, ...], str]]]
    | None = None,
) -> dict[str, Any]:
    """Keep failures; require repeated exact local outputs and exact score control."""
    ids = {item.request_id for item in manifest.items}
    if set(runs) != {"edge", "cloud", "distributed"}:
        raise ValueError("heterogeneous correctness requires all three placements")
    quality: dict[str, set[float]] = defaultdict(set)
    outputs: dict[str, set[tuple[tuple[int, ...], str]]] = defaultdict(set)
    insufficient: dict[str, list[str]] = {}
    repeatable = True
    valid_quality = True
    statuses: dict[str, dict[str, int]] = {}
    reference_mismatches: dict[str, list[str]] = defaultdict(list)
    for mode, rows in runs.items():
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        measured = [row for row in rows if not row["warmup"]]
        statuses[mode] = dict(Counter(str(row["status"]) for row in measured))
        for row in measured:
            if row["request_id"] not in ids:
                raise ValueError("heterogeneous result has unknown request")
            if row["status"] != "ok":
                continue
            grouped[row["request_id"]].append(row)
            score = row.get("quality_score")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                valid_quality = False
            else:
                quality[row["request_id"]].add(float(score))
            signature = (tuple(row["output_token_ids"]), str(row["output_text"]))
            outputs[row["request_id"]].add(signature)
            if expected_outputs is not None and (
                signature != expected_outputs.get(mode, {}).get(row["request_id"])
            ):
                reference_mismatches[mode].append(row["request_id"])
        insufficient[mode] = [
            request_id
            for request_id in sorted(ids)
            if len({row["repetition"] for row in grouped[request_id]}) < 2
        ]
        for group in grouped.values():
            if len({tuple(row["output_token_ids"]) for row in group}) != 1 or (
                len({row["output_text"] for row in group}) != 1
            ):
                repeatable = False
    quality_disagreements = [key for key in sorted(ids) if len(quality[key]) != 1]
    coverage = all(not missing for missing in insufficient.values())
    quality_equal = valid_quality and not quality_disagreements
    return {
        "protocol": PROTOCOL,
        "full_repeated_successful_request_coverage": coverage,
        "within_placement_repeatable": repeatable,
        "task_quality_equal": quality_equal,
        "quality_disagreement_request_ids": quality_disagreements,
        "insufficient_repeatability": insufficient,
        "cross_gpu_output_disagreement_request_ids": [
            key for key in sorted(ids) if len(outputs[key]) > 1
        ],
        "measured_status_counts": statuses,
        "reference_output_mismatches": {
            mode: sorted(set(ids_)) for mode, ids_ in reference_mismatches.items()
        },
        "passed": (
            coverage and repeatable and quality_equal and not reference_mismatches
        ),
        "quality_scope": "frozen benchmark scoring function; no general accuracy claim",
    }
