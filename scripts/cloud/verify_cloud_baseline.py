"""Verify frozen G2 v2+ inputs in the image: build inputs, manifests, images, splits."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "short": (
        "data/manifests/physical_vqa_v2/physical_vqa_v2.jsonl",
        "configs/evaluation/physical_vqa_split_v2.json",
    ),
    "long": (
        "data/manifests/physical_vqa_v2/physical_vqa_v2_long.jsonl",
        "configs/evaluation/physical_vqa_split_v2_long.json",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> None:
    provenance = json.loads((ROOT / "configs/cloud/image_provenance.json").read_text())
    for name, expected in provenance["files"].items():
        assert _sha256(ROOT / name) == expected, name
    for task, (manifest_name, split_name) in TASKS.items():
        manifest = ROOT / manifest_name
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        ids = [row["request_id"] for row in rows]
        assert len(ids) == len(set(ids)), f"{task}: duplicate request IDs"
        for row in rows:
            image = manifest.parent / row["image_path"]
            assert _sha256(image) == row["image_sha256"], str(image)
        split = json.loads((ROOT / split_name).read_text())
        assert split["dataset_manifest_sha256"] == _sha256(manifest), task
        groups = [
            set(split[f"{name}_request_ids"])
            for name in ("training", "validation", "evaluation")
        ]
        assert all(groups), f"{task}: empty split"
        assert sum(map(len, groups)) == len(set.union(*groups)), f"{task}: overlap"
        assert set.union(*groups) <= set(ids), f"{task}: split IDs not in manifest"
        sizes = [len(group) for group in groups]
        print(f"{task}: {len(rows)} requests, images verified, split {sizes}")
    print("Verified build inputs, both G2 v2 manifests, every image and both splits")


if __name__ == "__main__":
    verify()
