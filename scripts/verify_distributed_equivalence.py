#!/usr/bin/env python3
"""Compare local decomposed and localhost distributed Moondream outputs."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.config import (
    ExecutionMode,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import InferenceRequest
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.execution.distributed import DistributedExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.moondream import MoondreamAdapter
from adaptive_vlm.progress import ProgressReporter
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer
from adaptive_vlm.transport.tensors import TensorPayload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path(".model-cache"), type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
    }
    exit_code = 0
    try:
        _run(args, result)
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        exit_code = 3
    finally:
        result["completed_at_utc"] = _utc_now()
        write_json_atomic(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


def _run(args: argparse.Namespace, result: dict[str, Any]) -> None:
    import torch

    progress = ProgressReporter(args.verbose)
    progress.detail(f"Loading experiment configuration from {args.config.resolve()}")
    config = load_experiment_config(args.config)
    if config.mode is not ExecutionMode.DISTRIBUTED:
        raise ValueError("distributed equivalence requires a distributed config")
    if config.model.adapter != "moondream2":
        raise ValueError("distributed equivalence requires the Moondream2 adapter")

    seeded = seed_everything(config.runtime.seed)
    progress.detail(
        f"Model {config.model.model_id}@{config.model.revision}; "
        f"vision={args.vision_device}, text={args.text_device}"
    )
    edge_load_started = perf_counter()
    with progress.phase("Loading the full model for the local reference"):
        edge_adapter = MoondreamAdapter.from_pretrained(
            config.model,
            cache_dir=args.cache_dir,
            artifact_manifest=args.artifact_manifest,
            vision_device=args.vision_device,
            text_device=args.text_device,
            offline=args.offline,
        )
    edge_load_ms = (perf_counter() - edge_load_started) * 1000
    request = InferenceRequest(
        request_id="chunk5-equivalence",
        image_path=args.image.resolve(),
        prompt=args.prompt,
    )
    with progress.phase("Running the local decomposed reference"):
        reference_run = edge_adapter.run_decomposed(request)
    reference = reference_run.result
    reference_payload = TensorPayload.from_torch(reference_run.projected.tensor)
    reconstructed = reference_payload.to_torch(torch, device="cpu")
    roundtrip = TensorPayload.from_torch(reconstructed)
    tensor_roundtrip_equal = roundtrip.data == reference_payload.data
    del reconstructed, reference_run
    if not tensor_roundtrip_equal:
        raise RuntimeError("projected tensor changed during local serialization")

    edge_adapter.retain_only("vision")
    cloud_load_started = perf_counter()
    with progress.phase("Loading the text-only cloud stage"):
        cloud_adapter = MoondreamAdapter.from_pretrained(
            config.model,
            cache_dir=args.cache_dir,
            artifact_manifest=args.artifact_manifest,
            vision_device="cpu",
            text_device=args.text_device,
            offline=args.offline,
            components="text",
        )
    cloud_load_ms = (perf_counter() - cloud_load_started) * 1000

    contract_fingerprint = model_contract_fingerprint(config.model)
    processor = CloudRequestProcessor(
        protocol_version=config.transport.protocol_version,
        model_fingerprint=contract_fingerprint,
        tensor_inference_function=cloud_adapter.generate_from_payload,
    )
    server = CloudServer(
        host="127.0.0.1",
        port=0,
        timeout_seconds=config.transport.timeout_seconds,
        processor=processor,
    )
    server.start()
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="chunk5-moondream-server",
        daemon=True,
    )
    server_thread.start()
    transport = replace(config.transport, host="127.0.0.1", port=server.port)
    model = ModelMetadata(
        model_id=config.model.model_id,
        revision=config.model.revision,
        fingerprint=contract_fingerprint,
    )
    sent_payloads: list[TensorPayload] = []

    def capture_payload(tensor: object) -> TensorPayload:
        payload = TensorPayload.from_torch(tensor)
        sent_payloads.append(payload)
        return payload

    policy = DistributedExecutionPolicy(
        encoder=edge_adapter,
        transport=transport,
        model=model,
        tensor_serializer=capture_payload,
    )
    try:
        with progress.phase("Checking the text-stage server contract"):
            health = policy.health()
        with progress.phase("Running synchronous distributed inference over TCP"):
            distributed = policy.infer(request)
    finally:
        server.shutdown()
        server_thread.join(timeout=config.transport.timeout_seconds + 1)
        progress.info("Localhost distributed server stopped")

    if server_thread.is_alive():
        raise RuntimeError("server did not shut down within the timeout")
    if len(sent_payloads) != 1:
        raise RuntimeError("distributed policy did not serialize exactly one tensor")
    edge_tensor_equal = sent_payloads[0].data == reference_payload.data
    token_ids_equal = reference.output_token_ids == distributed.output_token_ids
    text_equal = reference.output_text == distributed.output_text
    if not edge_tensor_equal or not token_ids_equal or not text_equal:
        raise RuntimeError("distributed result differs from the local reference")

    stats = processor.stats
    result.update(
        {
            "status": "ok",
            "seeded_providers": list(seeded),
            "edge_model_load_ms": edge_load_ms,
            "cloud_model_load_ms": cloud_load_ms,
            "edge_component_role": edge_adapter.component_role,
            "cloud_component_role": cloud_adapter.component_role,
            "server_ready": health.ready,
            "model_contract_fingerprint": contract_fingerprint,
            "tensor_roundtrip_bits_equal": tensor_roundtrip_equal,
            "reencoded_edge_tensor_bits_equal": edge_tensor_equal,
            "projected_tensor": reference_payload.metadata(),
            "token_ids_equal": token_ids_equal,
            "text_equal": text_equal,
            "reference": reference.to_dict(),
            "distributed": distributed.to_dict(),
            "server_stats": {
                "completed": stats.completed_requests,
                "rejected": stats.rejected_requests,
                "failed": stats.failed_requests,
            },
        }
    )
    progress.info("Equivalence gate passed: tensor bits, tokens, and text match")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
