"""Run a resident cloud CPU reference against archived physical Phase-1 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.determinism import seed_everything
from adaptive_vlm.environment import collect_environment
from adaptive_vlm.models.moondream import MoondreamAdapter
from adaptive_vlm.timing import perf_counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.repetitions < 1 or (args.limit is not None and args.limit < 1):
        parser.error("repetitions and limit must be positive")

    config = load_experiment_config(args.config)
    frozen = load_frozen_manifest(
        args.manifest, expected_revision=config.dataset.revision
    )
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline["dataset_manifest_sha256"] != frozen.sha256:
        parser.error("baseline manifest SHA differs")
    if baseline["model_contract_fingerprint"] != model_contract_fingerprint(
        config.model
    ):
        parser.error("baseline model fingerprint differs")
    baseline_rows = {
        (row["request_id"], row["decision"]["configuration_id"]): row["result"]
        for row in baseline["rows"]
    }
    seeded = seed_everything(config.runtime.seed)
    environment = collect_environment()
    environment["seeded_providers"] = list(seeded)
    started = perf_counter()
    model = MoondreamAdapter.from_pretrained(
        config.model,
        cache_dir=args.cache_dir,
        artifact_manifest=args.artifact_manifest,
        vision_device="cpu",
        text_device="cpu",
        offline=True,
    )
    load_ms = (perf_counter() - started) * 1000
    rows = []
    for item in frozen.items[: args.limit]:
        request = item.to_request()
        for repetition in range(args.repetitions):
            started = perf_counter()
            result = model.generate(request)
            ids = list(result.output_token_ids)
            reference = {
                name: baseline_rows[(request.request_id, name)]["output_token_ids"]
                for name in (
                    "edge/full",
                    "cloud/raw_image",
                    "distributed/projected/synchronous",
                )
            }
            row = {
                "request_id": request.request_id,
                "repetition": repetition,
                "cloud_cpu": result.to_dict(),
                "elapsed_ms": (perf_counter() - started) * 1000,
                "matches": {name: ids == tokens for name, tokens in reference.items()},
            }
            rows.append(row)
            print(request.request_id, repetition, row["matches"], flush=True)
    write_json_atomic(
        args.output,
        {
            "schema_version": 1,
            "environment": environment,
            "config": str(args.config.resolve()),
            "model_artifact_sha256": config.model.artifact_sha256,
            "model_contract_fingerprint": model_contract_fingerprint(config.model),
            "dataset_manifest_sha256": frozen.sha256,
            "seed": config.runtime.seed,
            "placement": {"vision": "cpu", "text": "cpu"},
            "model_load_ms": load_ms,
            "rows": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
