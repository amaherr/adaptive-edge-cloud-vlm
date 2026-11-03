"""Verify frozen cloud campaign inputs, including every original image."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    provenance = json.loads(
        (ROOT / "configs/experiments/cloud_provenance_v1.json").read_text()
    )
    for name, expected in provenance["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    manifest = ROOT / "data/manifests/physical_vqa_v1.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == len({row["request_id"] for row in rows}) == 100
    for row in rows:
        path = manifest.parent / row["image_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["image_sha256"], (
            str(path)
        )
    split = json.loads(
        (ROOT / "configs/evaluation/cloud_vqa_split_v1.json").read_text()
    )
    groups = [
        set(split[f"{name}_request_ids"])
        for name in ("training", "validation", "evaluation")
    ]
    assert list(map(len, groups)) == [60, 20, 20]
    assert len(set.union(*groups)) == 100
    assert set.union(*groups) == {row["request_id"] for row in rows}
    print("Verified frozen contracts, 100 requests/images, and disjoint 60/20/20 split")


if __name__ == "__main__":
    verify()
