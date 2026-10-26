"""Run the three Phase 1 configurations with one resident edge model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.contracts import InferenceRequest
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.environment import collect_environment
from adaptive_vlm.execution.adaptive import load_phase1_policies
from adaptive_vlm.execution.remote import check_remote_capabilities
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.scheduler.configuration import (
    CONFIGURATIONS,
    CapabilityRegistry,
)
from adaptive_vlm.scheduler.dispatch import ConfigurationDispatcher
from adaptive_vlm.scheduler.state import SystemState, TaskState


def _rss_mib() -> float | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".model-cache"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--baseline-edge-requests", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vision-device", default="cpu")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    config = load_experiment_config(args.config)
    if args.manifest is None and (args.image is None or args.prompt is None):
        parser.error("provide --manifest or both --image and --prompt")
    if args.manifest is not None and (
        args.image is not None or args.prompt is not None
    ):
        parser.error("--manifest cannot be combined with --image or --prompt")
    if config.cloud.status == "pending":
        parser.error("cloud hardware must be declared before the smoke run")
    if args.manifest is not None:
        frozen = load_frozen_manifest(
            args.manifest, expected_revision=config.dataset.revision
        )
        requests = [item.to_request() for item in frozen.items]
        manifest_sha256 = frozen.sha256
    else:
        requests = [InferenceRequest("phase1-smoke", args.image.resolve(), args.prompt)]
        manifest_sha256 = None
    baseline_tokens: dict[str, list[int]] = {}
    if args.baseline_edge_requests is not None:
        baseline_lines = args.baseline_edge_requests.read_text(
            encoding="utf-8"
        ).splitlines()
        for line in baseline_lines:
            row = json.loads(line)
            if not row["warmup"] and row["status"] == "ok":
                baseline_tokens[row["request_id"]] = row["output_token_ids"]
        if set(baseline_tokens) != {request.request_id for request in requests}:
            parser.error("baseline edge rows do not cover the selected requests")
    seed_everything(config.runtime.seed)
    model = ModelMetadata(
        config.model.model_id,
        config.model.revision,
        model_contract_fingerprint(config.model),
    )
    started = perf_counter()
    policies = load_phase1_policies(
        config,
        artifact_manifest=args.artifact_manifest,
        cache_dir=args.cache_dir,
        offline=args.offline,
        vision_device=args.vision_device,
        text_device=args.text_device,
        edge_components="full",
    )
    load_ms = (perf_counter() - started) * 1000
    remote = check_remote_capabilities(config.transport, model)
    resident_rss_mib = _rss_mib()
    registry = CapabilityRegistry(
        policies=policies,
        expected_model_fingerprint=model.fingerprint,
        remote=remote,
        edge_resident_memory_mib=resident_rss_mib,
    )
    if not all(status.available for status in registry.statuses()):
        raise RuntimeError(f"required policy unavailable: {registry.statuses()}")
    state = SystemState(source="manual")
    dispatcher = ConfigurationDispatcher(registry)
    rows: list[dict[str, Any]] = []
    all_equal = True
    all_match_baseline = True
    for request in requests:
        task = TaskState.from_request(request)
        request_rows: list[dict[str, Any]] = []
        for spec in CONFIGURATIONS:
            started = perf_counter()
            dispatched = dispatcher.infer(
                request,
                configuration_id=spec.id,
                system_state=state,
                task_state=task,
            )
            row = {
                "request_id": request.request_id,
                "decision": dispatched.decision.to_dict(),
                "result": dispatched.result.to_dict(),
                "observed_dispatch_ms": (perf_counter() - started) * 1000,
                "selection_overhead_ms": dispatched.selection_overhead_ms,
                "client_rss_mib_after": _rss_mib(),
            }
            request_rows.append(row)
            rows.append(row)
            print(
                f"{request.request_id} {spec.id.value}: "
                f"{len(dispatched.result.output_token_ids)} tokens",
                flush=True,
            )
        token_sequences = [row["result"]["output_token_ids"] for row in request_rows]
        texts = [row["result"]["output_text"] for row in request_rows]
        all_equal &= len(set(map(tuple, token_sequences))) == 1 and len(set(texts)) == 1
        if baseline_tokens:
            all_match_baseline &= (
                token_sequences[0] == baseline_tokens[request.request_id]
            )
    report = {
        "schema_version": 1,
        "environment": collect_environment(),
        "model_contract_fingerprint": model.fingerprint,
        "dataset_manifest_sha256": manifest_sha256,
        "remote_paths": sorted(remote.paths),
        "edge_model_load_ms": load_ms,
        "edge_resident_rss_mib": resident_rss_mib,
        "rows": rows,
        "request_count": len(requests),
        "complete_and_equal": all_equal,
        "matches_baseline_edge": all_match_baseline if baseline_tokens else None,
    }
    write_json_atomic(args.output_dir / "phase1_smoke.json", report)
    if not all_equal or not all_match_baseline:
        raise RuntimeError("Phase 1 outputs differ; inspect phase1_smoke.json")
    print(
        json.dumps(
            {
                "complete_and_equal": True,
                "matches_baseline_edge": report["matches_baseline_edge"],
                "request_count": len(requests),
                "edge_model_load_ms": load_ms,
                "edge_resident_rss_mib": resident_rss_mib,
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
