"""Live S3 round-trip acceptance using CPU fake-adapter data, never research results."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from adaptive_vlm.artifacts import create_run_manifest, write_json_atomic
from adaptive_vlm.benchmark.runner import run_benchmark
from adaptive_vlm.config import load_experiment_config, model_contract_fingerprint
from adaptive_vlm.data.manifest import load_frozen_manifest
from adaptive_vlm.execution.local import LocalExecutionPolicy
from adaptive_vlm.models.fake import FakeModelAdapter
from adaptive_vlm.ops.checkpoints import CheckpointSession, restore, sha
from adaptive_vlm.ops.store import S3Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = load_experiment_config("configs/experiments/chunk6_fake_benchmark.yaml")
    manifest = load_frozen_manifest(config.dataset.manifest_path)
    # This explicitly synthetic reference tests contract comparisons. No container
    # or GPU execution, image publication or real-model result is claimed.
    image = "diagnostic/synthetic-reference@sha256:" + "a" * 64
    contract = {
        "campaign_id": "s3-api-acceptance",
        "block_id": "diagnostic",
        "policy": "edge",
        "host_pair_id": "single-local-cpu",
        "provider": "diagnostic-only",
        "image_digest": image,
        "model_sha256": model_contract_fingerprint(config.model),
        "dataset_sha256": manifest.sha256,
        "split_sha256": "b" * 64,
        "grid_sha256": "c" * 64,
        "hardware_contract_sha256": "d" * 64,
    }
    prefix = "research/adaptive-vlm/vultr-v1/acceptance/" + uuid.uuid4().hex
    store = S3Store(args.bucket, profile=args.profile)
    kwargs = dict(
        config=config,
        manifest=manifest,
        policy=LocalExecutionPolicy(FakeModelAdapter()),
        run_manifest=create_run_manifest(
            config,
            {
                "container": {
                    "declared_image_reference": image,
                },
                "acceptance_only": True,
            },
            dataset_manifest_sha256=manifest.sha256,
        ),
        warmups=1,
        repetitions=1,
        pipeline="synchronous",
    )
    source, target = args.output_dir / "source", args.output_dir / "restored"
    expected = run_benchmark(
        **kwargs,
        run_dir=source,
        checkpoint=CheckpointSession(store, prefix, contract, every_trials=5),
    )
    restore(store, prefix, contract, target)
    for name in ("plan.json", "run_manifest.json", "requests.jsonl", "summary.json"):
        assert (source / name).read_bytes() == (target / name).read_bytes()
    result = run_benchmark(
        **kwargs,
        run_dir=target,
        resume=True,
        checkpoint=CheckpointSession(store, prefix, contract),
    )
    assert result == expected
    write_json_atomic(
        args.output_dir / "report.json",
        {
            "kind": (
                "CPU fake-adapter S3 acceptance; synthetic image reference; "
                "no GPU/model claim"
            ),
            "bucket": args.bucket,
            "prefix": prefix,
            "byte_matching_restore": True,
            "existing_resume_unchanged": True,
            "journal_sha256": sha((target / "requests.jsonl").read_bytes()),
            "measured_diagnostic_rows": result["measured_attempts"],
        },
    )
    print(json.dumps({"report": str(args.output_dir / "report.json")}))


if __name__ == "__main__":
    main()
