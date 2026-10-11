#!/usr/bin/env python3
"""Exercise 100 full-size projected-tensor requests over localhost TCP."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.config import (
    ExecutionMode,
    load_experiment_config,
    model_contract_fingerprint,
)
from adaptive_vlm.contracts import InferenceRequest, InferenceResult
from adaptive_vlm.execution.distributed import DistributedExecutionPolicy
from adaptive_vlm.models.base import ModelMetadata
from adaptive_vlm.models.features import ProjectedVisionFeatures, VisionEncoding
from adaptive_vlm.progress import ProgressReporter
from adaptive_vlm.transport.server import CloudRequestProcessor, CloudServer
from adaptive_vlm.transport.tensors import TensorInferenceRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--requests", default=100, type=int)
    parser.add_argument("--max-rss-growth-mib", default=64.0, type=float)
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

    if args.requests <= 0:
        raise ValueError("requests must be positive")
    if args.max_rss_growth_mib < 0:
        raise ValueError("max-rss-growth-mib must be non-negative")
    config = load_experiment_config(args.config)
    if config.mode is not ExecutionMode.DISTRIBUTED:
        raise ValueError("stability verification requires a distributed config")

    progress = ProgressReporter(args.verbose)
    projected = ProjectedVisionFeatures.from_tensor(
        torch.arange(729 * 2048, dtype=torch.int32)
        .remainder(2048)
        .to(torch.bfloat16)
        .reshape(729, 2048)
    )

    class StaticEncoder:
        component_role = "vision"

        @staticmethod
        def encode_projected(_request: InferenceRequest) -> VisionEncoding:
            return VisionEncoding(projected=projected, metrics={"vision_total_ms": 0.0})

    def tensor_infer(request: TensorInferenceRequest) -> InferenceResult:
        if request.tensor.shape != projected.shape:
            raise ValueError("server received an unexpected tensor shape")
        return InferenceResult(
            request_id=request.request_id,
            output_text="stable",
            output_token_ids=(101, 102),
            metrics={"fake_text_ms": 0.0},
        )

    contract_fingerprint = model_contract_fingerprint(config.model)
    processor = CloudRequestProcessor(
        protocol_version=config.transport.protocol_version,
        model_fingerprint=contract_fingerprint,
        tensor_inference_function=tensor_infer,
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
        name="chunk5-stability-server",
        daemon=True,
    )
    server_thread.start()
    transport = replace(config.transport, host="127.0.0.1", port=server.port)
    model = ModelMetadata(
        model_id=config.model.model_id,
        revision=config.model.revision,
        fingerprint=contract_fingerprint,
    )
    policy = DistributedExecutionPolicy(
        encoder=StaticEncoder(),
        transport=transport,
        model=model,
    )
    image_path = args.image.resolve()
    request_bytes: list[int] = []
    response_bytes: list[int] = []
    try:
        with progress.phase("Checking the tensor server contract"):
            health = policy.health()
        if not health.ready:
            raise RuntimeError("localhost tensor server did not report ready")
        policy.infer(
            InferenceRequest(
                request_id="stability-warmup",
                image_path=image_path,
                prompt="warm up the tensor path",
            )
        )
        rss_samples = [_rss_mib()]
        with progress.phase(f"Sending {args.requests} full-size tensor requests"):
            for index in range(args.requests):
                actual = policy.infer(
                    InferenceRequest(
                        request_id=f"tensor-stability-{index:04d}",
                        image_path=image_path,
                        prompt="exercise the projected tensor path",
                    )
                )
                if actual.output_token_ids != (101, 102):
                    raise RuntimeError("tensor response changed during stability run")
                request_bytes.append(int(actual.metrics["request_wire_bytes"]))
                response_bytes.append(int(actual.metrics["response_wire_bytes"]))
                rss_samples.append(_rss_mib())
                progress.detail(
                    f"Completed request {index + 1}/{args.requests}; "
                    f"RSS={rss_samples[-1]:.3f} MiB"
                )
    finally:
        server.shutdown()
        server_thread.join(timeout=config.transport.timeout_seconds + 1)
        progress.info("Localhost tensor server stopped")

    if server_thread.is_alive():
        raise RuntimeError("server did not shut down within the timeout")
    stats = processor.stats
    if stats.completed_requests != args.requests + 1:
        raise RuntimeError("server did not complete every request")
    peak_growth_mib = max(rss_samples) - rss_samples[0]
    if peak_growth_mib > args.max_rss_growth_mib:
        raise RuntimeError(
            f"RSS grew by {peak_growth_mib:.3f} MiB, above the configured limit"
        )

    result.update(
        {
            "status": "ok",
            "requests": args.requests,
            "warmup_requests": 1,
            "projected_tensor": projected.metadata(),
            "model_contract_fingerprint": contract_fingerprint,
            "server_stats": {
                "completed": stats.completed_requests,
                "rejected": stats.rejected_requests,
                "failed": stats.failed_requests,
            },
            "rss_initial_mib": rss_samples[0],
            "rss_final_mib": rss_samples[-1],
            "rss_peak_mib": max(rss_samples),
            "rss_peak_growth_mib": peak_growth_mib,
            "rss_growth_limit_mib": args.max_rss_growth_mib,
            "request_wire_bytes": sorted(set(request_bytes)),
            "response_wire_bytes_min": min(response_bytes),
            "response_wire_bytes_max": max(response_bytes),
        }
    )
    progress.info(
        f"Stability gate passed: {args.requests} requests, "
        f"peak RSS growth {peak_growth_mib:.3f} MiB"
    )


def _rss_mib() -> float:
    statm = Path("/proc/self/statm").read_text(encoding="ascii").split()
    resident_pages = int(statm[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024**2


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
