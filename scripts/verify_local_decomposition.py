#!/usr/bin/env python3
"""Verify monolithic and locally decomposed Moondream inference equivalence."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.config import ExecutionMode, load_experiment_config
from adaptive_vlm.contracts import InferenceRequest
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.models.moondream import MoondreamAdapter, MoondreamRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path(".model-cache"), type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--vision-device", default="cpu")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repetitions", default=2, type=int)
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
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    config = load_experiment_config(args.config)
    if config.mode is not ExecutionMode.EDGE:
        raise ValueError("decomposition verification requires an edge config")
    model_config = config.model
    if args.max_new_tokens is not None:
        if args.max_new_tokens <= 0:
            raise ValueError("max-new-tokens must be positive")
        model_config = replace(model_config, max_new_tokens=args.max_new_tokens)

    seeded = seed_everything(config.runtime.seed)
    load_started = perf_counter()
    adapter = MoondreamAdapter.from_pretrained(
        model_config,
        cache_dir=args.cache_dir,
        artifact_manifest=args.artifact_manifest,
        vision_device=args.vision_device,
        text_device=args.text_device,
        offline=args.offline,
    )
    load_ms = (perf_counter() - load_started) * 1000
    request = InferenceRequest(
        request_id="chunk3-equivalence",
        image_path=args.image.resolve(),
        prompt=args.prompt,
    )

    reference = adapter.run_monolithic(request)
    candidates = [adapter.run_decomposed(request) for _ in range(args.repetitions)]
    comparisons = [adapter.compare_runs(reference, run) for run in candidates]
    repeatable = _outputs_repeatable(candidates)
    equivalent = all(
        comparison["projected_tensor_equal"]
        and comparison["token_ids_equal"]
        and comparison["text_equal"]
        for comparison in comparisons
    )
    if not equivalent or not repeatable:
        raise RuntimeError("monolithic and decomposed outputs are not equivalent")

    rss_values = [
        float(run.result.metrics["rss_after_mib"]) for run in candidates
    ]
    result.update(
        {
            "status": "ok",
            "model": {
                "model_id": adapter.metadata.model_id,
                "revision": adapter.metadata.revision,
                "fingerprint": adapter.metadata.fingerprint,
                "max_new_tokens": model_config.max_new_tokens,
            },
            "devices": {
                "vision": adapter.vision_device,
                "text": adapter.text_device,
            },
            "seeded_providers": list(seeded),
            "model_load_ms": load_ms,
            "reference": _run_payload(reference),
            "decomposed_runs": [_run_payload(run) for run in candidates],
            "comparisons": comparisons,
            "decomposed_outputs_repeatable": repeatable,
            "rss_after_range_mib": max(rss_values) - min(rss_values),
        }
    )


def _outputs_repeatable(runs: list[MoondreamRun]) -> bool:
    token_outputs = {run.result.output_token_ids for run in runs}
    text_outputs = {run.result.output_text for run in runs}
    return len(token_outputs) == 1 and len(text_outputs) == 1


def _run_payload(run: MoondreamRun) -> dict[str, Any]:
    return {
        "result": run.result.to_dict(),
        "projected": run.projected.metadata(),
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
