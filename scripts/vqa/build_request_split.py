"""Write the crop-stratified G2 v2 request split (protocol section 4.3)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.data.vqa_selection import SPLIT_NAMES, build_request_split

NETWORK_CONDITIONS = {
    "training": [f"T{index:02d}" for index in range(1, 13)],
    "validation": [f"V{index:02d}" for index in range(1, 6)],
    "evaluation": [f"E{index:02d}" for index in range(1, 6)],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--counts", type=int, nargs=3, required=True)
    parser.add_argument("--two-crop-counts", type=int, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    config = load_experiment_config(args.config)
    manifest = load_frozen_manifest(
        config.dataset.manifest_path, expected_revision=config.dataset.revision
    )
    counts = dict(zip(SPLIT_NAMES, args.counts, strict=True))
    split = build_request_split(
        manifest.path,
        counts=counts,
        two_crop_counts=dict(zip(SPLIT_NAMES, args.two_crop_counts, strict=True)),
        seed=args.seed,
    )
    output_dir = args.output.resolve().parent
    payload = {
        "schema_version": 2,
        "campaign_id": args.campaign_id,
        "dataset_revision": manifest.revision,
        "dataset_manifest_path": Path(
            os.path.relpath(manifest.path, output_dir)
        ).as_posix(),
        "dataset_manifest_sha256": manifest.sha256,
        "model_contract_sha256": model_contract_fingerprint(config.model),
        "protocol_path": Path(
            os.path.relpath(args.protocol.resolve(), output_dir)
        ).as_posix(),
        "request_split_seed": args.seed,
        "request_split_counts": counts,
        "two_crop_request_counts": dict(
            zip(SPLIT_NAMES, args.two_crop_counts, strict=True)
        ),
        **{f"{name}_request_ids": split[name] for name in SPLIT_NAMES},
        **{
            f"{name}_network_condition_ids": NETWORK_CONDITIONS[name]
            for name in SPLIT_NAMES
        },
        "max_new_tokens": config.model.max_new_tokens,
        "sla_ms": 60000,
        "bootstrap_seed": 1707,
        "bootstrap_resamples": 2000,
        "confidence_level": 0.95,
        "task_type": "vqa_short_answer",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output} ({digest})")


if __name__ == "__main__":
    main()
