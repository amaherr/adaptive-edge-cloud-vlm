#!/usr/bin/env python3
"""Run the controlled monolithic Moondream feasibility experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="vikhyatk/moondream2")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument(
        "--cache-dir",
        default=Path(".model-cache"),
        type=Path,
        help="Repository-local Hugging Face home directory.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require every model artifact to exist in the local cache.",
    )
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        required=True,
        type=Path,
        help="Correctness fixture; repeat the option to test multiple images.",
    )
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", default=32, type=int)
    parser.add_argument("--warmups", default=1, type=int)
    parser.add_argument("--repetitions", default=2, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = _base_result(args)
    exit_code = 0
    try:
        _run_probe(args, result)
    except Exception as exc:  # The probe must preserve expected OOM evidence.
        result["status"] = _classify_error(exc)
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        exit_code = 3
    finally:
        result["completed_at_utc"] = _utc_now()
        write_json_atomic(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


def _base_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
        "python": platform.python_version(),
        "model_id": args.model_id,
        "revision": args.revision,
        "artifact_sha256": args.artifact_sha256,
        "cache_dir": str(args.cache_dir.resolve()),
        "offline": args.offline,
        "precision": "bfloat16",
        "device": args.device,
        "images": [str(image.resolve()) for image in args.images],
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
    }


def _run_probe(args: argparse.Namespace, result: dict[str, Any]) -> None:
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import psutil
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    uses_cuda = device.type == "cuda"
    if uses_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    missing_images = [image for image in args.images if not image.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Image does not exist: {missing_images[0]}")
    if args.max_new_tokens <= 0 or args.warmups < 0 or args.repetitions <= 0:
        raise ValueError("token count/repetitions are invalid")

    process = psutil.Process()
    result["software"] = {
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
    }
    if uses_cuda:
        properties = torch.cuda.get_device_properties(device)
        result["hardware"] = {
            "gpu": properties.name,
            "gpu_total_memory_mib": round(properties.total_memory / 1024**2, 3),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    else:
        result["hardware"] = {"processor": platform.processor(), "device": "cpu"}
    result["rss_before_load_mib"] = _rss_mib(process)

    if uses_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    load_started = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
        cache_dir=cache_dir / "hub",
        local_files_only=args.offline,
    )
    _synchronize(torch, device)
    result["model_load_ms"] = (perf_counter() - load_started) * 1000
    result["rss_after_load_mib"] = _rss_mib(process)
    if uses_cuda:
        result["load_peak_allocated_mib"] = _mib(
            torch.cuda.max_memory_allocated(device)
        )
        result["load_peak_reserved_mib"] = _mib(
            torch.cuda.max_memory_reserved(device)
        )

    settings = {
        "max_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "variant": None,
    }
    fixture_results = [
        _run_fixture(
            model=model,
            tokenizer=model.model.tokenizer,
            image=Image.open(image_path).convert("RGB"),
            image_path=image_path,
            prompt=args.prompt,
            settings=settings,
            warmups=args.warmups,
            repetitions=args.repetitions,
            torch=torch,
            device=device,
        )
        for image_path in args.images
    ]
    result.update(
        {
            "status": "ok",
            "fixtures": fixture_results,
            "all_outputs_repeatable": all(
                fixture["outputs_identical"] for fixture in fixture_results
            ),
            "rss_after_inference_mib": _rss_mib(process),
        }
    )
    if uses_cuda:
        result["inference_peak_allocated_mib"] = _mib(
            torch.cuda.max_memory_allocated(device)
        )
        result["inference_peak_reserved_mib"] = _mib(
            torch.cuda.max_memory_reserved(device)
        )


def _run_fixture(
    *,
    model: Any,
    tokenizer: Any,
    image: Any,
    image_path: Path,
    prompt: str,
    settings: dict[str, Any],
    warmups: int,
    repetitions: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    for _ in range(warmups):
        model.query(image, prompt, reasoning=False, settings=settings)
    _synchronize(torch, device)

    outputs: list[str] = []
    latencies_ms: list[float] = []
    for _ in range(repetitions):
        started = perf_counter()
        response = model.query(image, prompt, reasoning=False, settings=settings)
        _synchronize(torch, device)
        latencies_ms.append((perf_counter() - started) * 1000)
        outputs.append(response["answer"])

    reencoded_token_ids = tokenizer.encode(outputs[0]).ids
    return {
        "image": str(image_path.resolve()),
        "outputs": outputs,
        "outputs_identical": len(set(outputs)) == 1,
        "reencoded_output_token_ids": reencoded_token_ids,
        "output_tokens": len(reencoded_token_ids),
        "latencies_ms": latencies_ms,
    }


def _classify_error(error: Exception) -> str:
    try:
        import torch

        if isinstance(error, torch.OutOfMemoryError):
            return "oom"
    except ModuleNotFoundError:
        pass
    if "device not ready" in str(error).lower():
        return "device_error"
    return "error"


def _rss_mib(process: Any) -> float:
    return round(process.memory_info().rss / 1024**2, 3)


def _mib(byte_count: int) -> float:
    return round(byte_count / 1024**2, 3)


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
