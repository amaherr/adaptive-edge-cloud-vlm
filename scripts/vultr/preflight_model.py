"""Single-node CUDA/model feasibility only; never independent-host performance."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.contracts import InferenceRequest
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.environment import collect_environment
from adaptive_vlm.models.moondream import MoondreamAdapter


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    request_data = json.loads(args.request.read_text())
    image = args.request.parent / request_data["image_file"]
    if hashlib.sha256(image.read_bytes()).hexdigest() != request_data["image_sha256"]:
        raise ValueError("preflight training image hash mismatch")
    # Only the frozen model settings are reused from this image's packaged
    # common config. Its AWS hardware/network declarations are not executed or
    # recorded as Vultr provenance. Actual hardware is discovered below.
    model = load_experiment_config("/app/configs/experiments/cloud_vqa_v1.yaml").model
    fingerprint = model_contract_fingerprint(model)
    if fingerprint != request_data["model_contract_sha256"]:
        raise ValueError("preflight model contract differs from the frozen split")
    seed_everything(42)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA/BF16 is unavailable")
    torch.manual_seed(42)
    value = torch.randn(256, 256, device="cuda:0", dtype=torch.bfloat16)
    assert bool(torch.isfinite(value @ value).all())
    del value
    torch.cuda.empty_cache()
    devices = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(i)),
        }
        for i in range(torch.cuda.device_count())
    ]
    report = {
        "kind": "Vultr single-node feasibility; independent-host Gate 2 remains open",
        "declared_image_reference": os.environ["APP_IMAGE"],
        "model_contract_sha256": fingerprint,
        "bf16_matrix_finite": True,
        "devices": devices,
        "environment": collect_environment(),
        "request_id": request_data["request_id"],
        "image_sha256": request_data["image_sha256"],
        "results": {},
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, report)
    request = InferenceRequest(
        request_data["request_id"], image, request_data["prompt"]
    )
    common = dict(
        cache_dir="/models",
        artifact_manifest="/app/configs/model/moondream2_artifact.yaml",
        vision_device="cuda:0",
        text_device="cuda:0",
    )
    adapter = MoondreamAdapter.from_pretrained(
        model, offline=False, components="full", **common
    )
    result = adapter.generate(request)
    full_peak = torch.cuda.max_memory_reserved(0)
    report["results"]["full"] = result.to_dict()
    report["full_peak_reserved_bytes"] = full_peak
    report["full_memory_headroom_fraction"] = 1 - full_peak / devices[0]["memory_bytes"]
    write_json_atomic(args.output, report)
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    vision = MoondreamAdapter.from_pretrained(
        model, offline=True, components="vision", **common
    )
    torch.cuda.empty_cache()
    encoding = vision.encode_projected(request)
    report["results"]["vision"] = dict(encoding.metrics)
    report["projected_nbytes"] = encoding.projected.nbytes
    write_json_atomic(args.output, report)
    del vision
    gc.collect()
    torch.cuda.empty_cache()
    text = MoondreamAdapter.from_pretrained(
        model, offline=True, components="text", **common
    )
    projected_result = text.generate_from_projected(
        request_id=request.request_id,
        prompt=request.prompt,
        projected=encoding.projected,
    )
    report["results"]["text"] = projected_result.to_dict()
    report["same_node_exact_tokens"] = (
        result.output_token_ids == projected_result.output_token_ids
    )
    report["passed"] = (
        report["same_node_exact_tokens"]
        and report["full_memory_headroom_fraction"] >= 0.10
    )
    write_json_atomic(args.output, report)
    if not report["passed"]:
        raise SystemExit("Single-node feasibility failed; inspect archived evidence")
    print(
        "Single-node feasibility passed; independent-host gates remain open"
    )


if __name__ == "__main__":
    main()
