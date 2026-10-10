#!/usr/bin/env python3
"""Compare local monolithic and localhost all-cloud Moondream outputs."""

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
from adaptive_vlm.execution.cloud import CloudExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.moondream import MoondreamAdapter
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path(".model-cache"), type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
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
    config = load_experiment_config(args.config)
    if config.mode is not ExecutionMode.CLOUD:
        raise ValueError("cloud equivalence requires a cloud config")
    if config.model.adapter != "moondream2":
        raise ValueError("cloud equivalence requires the Moondream2 adapter")
    seeded = seed_everything(config.runtime.seed)
    load_started = perf_counter()
    adapter = MoondreamAdapter.from_pretrained(
        config.model,
        cache_dir=args.cache_dir,
        artifact_manifest=args.artifact_manifest,
        vision_device=args.device,
        text_device=args.device,
        offline=args.offline,
    )
    model_load_ms = (perf_counter() - load_started) * 1000
    request = InferenceRequest(
        request_id="chunk4-equivalence",
        image_path=args.image.resolve(),
        prompt=args.prompt,
    )
    reference = adapter.run_monolithic(request).result
    contract_fingerprint = model_contract_fingerprint(config.model)

    processor = CloudRequestProcessor(
        protocol_version=config.transport.protocol_version,
        model_fingerprint=contract_fingerprint,
        inference_function=lambda remote: adapter.run_monolithic(remote).result,
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
        name="chunk4-moondream-server",
        daemon=True,
    )
    server_thread.start()
    transport = replace(config.transport, host="127.0.0.1", port=server.port)
    remote_model = ModelMetadata(
        model_id=adapter.metadata.model_id,
        revision=adapter.metadata.revision,
        fingerprint=contract_fingerprint,
    )
    policy = CloudExecutionPolicy(transport=transport, model=remote_model)
    try:
        health = policy.health()
        remote = policy.infer(request)
    finally:
        server.shutdown()
        server_thread.join(timeout=config.transport.timeout_seconds + 1)

    if server_thread.is_alive():
        raise RuntimeError("server did not shut down within the timeout")
    token_ids_equal = reference.output_token_ids == remote.output_token_ids
    text_equal = reference.output_text == remote.output_text
    if not token_ids_equal or not text_equal:
        raise RuntimeError(
            "localhost cloud output differs from local monolithic output"
        )
    result.update(
        {
            "status": "ok",
            "seeded_providers": list(seeded),
            "model_load_ms": model_load_ms,
            "model": {
                "model_id": adapter.metadata.model_id,
                "revision": adapter.metadata.revision,
                "fingerprint": adapter.metadata.fingerprint,
                "contract_fingerprint": contract_fingerprint,
            },
            "server_ready": health.ready,
            "server_model_fingerprint": health.model_fingerprint,
            "token_ids_equal": token_ids_equal,
            "text_equal": text_equal,
            "reference": reference.to_dict(),
            "cloud": remote.to_dict(),
            "server_stats": {
                "completed": processor.stats.completed_requests,
                "rejected": processor.stats.rejected_requests,
                "failed": processor.stats.failed_requests,
            },
        }
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
