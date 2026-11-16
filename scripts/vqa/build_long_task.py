"""Build the G2 v2 long-answer task from the frozen short-answer dataset.

Protocol section 3.5: the same images and questions as the short-answer
manifest, with the selected long-answer template; request IDs get ``-long``;
scoring is ``none`` (timed only). Each split keeps a crop-stratified subset of
its short-answer IDs, drawn with seed 3907: 30/15/15 IDs, 6/3/3 of them
two-crop images.

Writes the manifest, the split and the experiment config; refuses to
overwrite them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHORT_MANIFEST = ROOT / "data/manifests/physical_vqa_v2/physical_vqa_v2.jsonl"
SELECTION = ROOT / "data/manifests/physical_vqa_v2/physical_vqa_v2.selection.json"
SHORT_SPLIT = ROOT / "configs/evaluation/physical_vqa_split_v2.json"
SHORT_CONFIG = ROOT / "configs/experiments/physical_vqa_v2.yaml"
SHORT_SUFFIX = " Answer with a single word or phrase."
COUNTS = {"training": (30, 6), "validation": (15, 3), "evaluation": (15, 3)}
SEED = 3907
LONG_MANIFEST_FROM_SPLIT = (
    "../../data/manifests/physical_vqa_v2/physical_vqa_v2_long.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-id", required=True)
    parser.add_argument("--template", required=True, help="May contain {question}")
    parser.add_argument(
        "--conditions",
        type=Path,
        required=True,
        help="JSON {split: [condition IDs]} for the long cells",
    )
    args = parser.parse_args()
    # A template may ignore the question (E1 is a detailed-description task).
    manifest = ROOT / "data/manifests/physical_vqa_v2/physical_vqa_v2_long.jsonl"
    split_path = ROOT / "configs/evaluation/physical_vqa_split_v2_long.json"
    config_path = ROOT / "configs/experiments/physical_vqa_v2_long.yaml"
    for path in (manifest, split_path, config_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    short = json.loads(SHORT_SPLIT.read_text())
    crops = json.loads(SELECTION.read_text())["crop_counts"]
    rng = random.Random(SEED)
    chosen: dict[str, list[str]] = {}
    for name, (total, two_crop) in COUNTS.items():
        ids = sorted(short[f"{name}_request_ids"])
        two = [i for i in ids if crops[i] == 2]
        rest = [i for i in ids if crops[i] != 2]
        picked = rng.sample(two, two_crop) + rng.sample(rest, total - two_crop)
        chosen[name] = sorted(picked)
    wanted = {rid for ids in chosen.values() for rid in ids}

    rows = []
    for line in SHORT_MANIFEST.read_text().splitlines():
        row = json.loads(line)
        if row["request_id"] not in wanted:
            continue
        if not row["prompt"].endswith(SHORT_SUFFIX):
            raise ValueError(f"unexpected short prompt: {row['request_id']}")
        question = row["prompt"][: -len(SHORT_SUFFIX)]
        row.update(
            request_id=row["request_id"] + "-long",
            prompt=args.template.format(question=question),
            scoring="none",
            references=[],
            revision="vqa-v2-val2014-physical-v2-long",
        )
        rows.append(row)
    if len(rows) != len(wanted):
        raise ValueError("some chosen IDs are missing from the short manifest")
    manifest.write_text(
        "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows
        ),
        newline="\n",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    conditions = json.loads(args.conditions.read_text())
    split = {
        "schema_version": 2,
        "campaign_id": "physical-vqa-v2",
        "task": "long",
        "task_bucket": "vqa-long-answer-32-token-budget",
        "template_id": args.template_id,
        "template": args.template,
        "dataset_revision": "vqa-v2-val2014-physical-v2-long",
        "dataset_manifest_path": LONG_MANIFEST_FROM_SPLIT,
        "dataset_manifest_sha256": manifest_sha,
        "model_contract_sha256": short["model_contract_sha256"],
        "selection_seed": SEED,
        "description": (
            "G2 v2 long-answer task (protocol section 3.5): crop-stratified "
            "subset of each split's short-answer IDs with -long appended."
        ),
    }
    for name in COUNTS:
        split[f"{name}_request_ids"] = [rid + "-long" for rid in chosen[name]]
        split[f"{name}_two_crop_count"] = sum(crops[r] == 2 for r in chosen[name])
        split[f"{name}_network_condition_ids"] = sorted(conditions.get(name, []))
    split_path.write_text(json.dumps(split, indent=2) + "\n", newline="\n")

    text = SHORT_CONFIG.read_text()
    old_manifest = "../../data/manifests/physical_vqa_v2/physical_vqa_v2.jsonl"
    if old_manifest not in text:
        raise ValueError("unexpected short experiment config")
    text = text.replace(
        old_manifest, "../../data/manifests/physical_vqa_v2/physical_vqa_v2_long.jsonl"
    )
    text = text.replace(
        "revision: vqa-v2-val2014-physical-v2\n",
        "revision: vqa-v2-val2014-physical-v2-long\n",
    )
    text = text.replace("name: physical-vqa-v2", "name: physical-vqa-v2-long", 1)
    config_path.write_text(text, newline="\n")
    print(
        json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "counts": {k: len(v) for k, v in chosen.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
