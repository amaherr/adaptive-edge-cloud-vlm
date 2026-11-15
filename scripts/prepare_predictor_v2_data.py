"""Create exact pre-execution tasks and isolated physical Predictor V2 slices."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path

from adaptive_vlm.artifacts import write_json_atomic
from adaptive_vlm.benchmark.predictor_v2_data import (
    digest,
    prepare_partitions,
    strict_json,
)
from adaptive_vlm.data.manifest import load_frozen_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path, default=Path("configs/predictor/physical_vqa_v2.json")
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("results/physical-g2-20261110/evidence.tar.gz"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("results/physical-g2-20261110/archive_inventory.json"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/evaluation/physical_vqa_split_v1.json"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/physical_vqa_v1.jsonl")
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            ".model-cache/hub/models--moondream--starmie-v1/snapshots/35192e10a54e36eabe0a7cc57a2c1aab371cafc5/tokenizer.json"
        ),
    )
    parser.add_argument(
        "--template-source",
        type=Path,
        default=Path(".model-cache/audit/moondream2-9a7d4024/config.py"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output directory")
    contract = strict_json(args.contract.read_bytes())
    if (
        digest(args.tokenizer) != contract["tokenizer_sha256"]
        or digest(args.template_source) != contract["template_source_sha256"]
    ):
        raise ValueError("pinned tokenizer/template SHA mismatch")
    template = None
    for node in ast.walk(ast.parse(args.template_source.read_text())):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "query":
                    template = ast.literal_eval(value)
    if template != {"prefix": [1, 15381, 2], "suffix": [3]}:
        raise ValueError("unsupported pinned query template")
    tokenizer = importlib.import_module("tokenizers").Tokenizer.from_file(
        str(args.tokenizer)
    )
    image_module = importlib.import_module("PIL.Image")
    manifest = load_frozen_manifest(args.manifest)
    tasks = {}
    for item in manifest.items:
        raw = item.image_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item.image_sha256:
            raise ValueError("frozen image bytes changed")
        with image_module.open(item.image_path) as image:
            width, height = image.size
        # The pinned non-reasoning query appends the suffix a second time.
        prompt_tokens = (
            len(template["prefix"])
            + len(tokenizer.encode(item.prompt).ids)
            + 2 * len(template["suffix"])
        )
        tasks[item.request_id] = {
            "task_bucket": contract["supported_task_bucket"],
            "image_bytes": len(raw),
            "image_width": width,
            "image_height": height,
            "prompt_tokens": prompt_tokens,
            "requested_max_new_tokens": 32,
        }
    feature_path = args.output.parent / (args.output.name + "-task-features.json")
    if feature_path.exists():
        raise FileExistsError("pre-execution feature artifact exists")
    write_json_atomic(
        feature_path,
        {
            "schema_version": 2,
            "model_contract_sha256": contract["model_contract_sha256"],
            "dataset_manifest_sha256": manifest.sha256,
            "tokenizer_sha256": digest(args.tokenizer),
            "template_source_sha256": digest(args.template_source),
            "feature_source": (
                "Frozen input bytes, Pillow headers and exact pinned "
                "tokenizer/template; no measured outcomes"
            ),
            "tasks": tasks,
        },
    )
    report = prepare_partitions(
        archive=args.archive,
        inventory_path=args.inventory,
        split_path=args.split,
        manifest_path=args.manifest,
        feature_path=feature_path,
        contract_path=args.contract,
        output=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
