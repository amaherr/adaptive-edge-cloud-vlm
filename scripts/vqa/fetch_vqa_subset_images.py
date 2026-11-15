"""Fetch the seeded VQA v2 val2014 image subset from the official COCO bucket."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def selected_image_ids(questions_path: Path, count: int, seed: int) -> list[int]:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    by_id = {row["question_id"]: row["image_id"] for row in questions}
    if len(by_id) != len(questions):
        raise ValueError("duplicate question IDs")
    if count < 1 or count > len(by_id):
        raise ValueError(f"count must be in 1..{len(by_id)}")
    chosen = random.Random(seed).sample(sorted(by_id), count)
    return sorted({by_id[question_id] for question_id in chosen})


def fetch_one(image_id: int, images_dir: Path) -> str:
    name = f"COCO_val2014_{image_id:012d}.jpg"
    target = images_dir / name
    if target.exists():
        return "existing"
    url = f"https://s3.amazonaws.com/images.cocodataset.org/val2014/{name}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            if len(data) < 100 or not data.startswith(b"\xff\xd8"):
                raise ValueError(f"invalid JPEG response for {name}")
            with tempfile.NamedTemporaryFile(dir=images_dir, delete=False) as output:
                temporary = Path(output.name)
                output.write(data)
            os.replace(temporary, target)
            return "downloaded"
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    ids = selected_image_ids(args.questions, args.count, args.seed)
    args.images_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(fetch_one, image_id, args.images_dir) for image_id in ids
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            downloaded += future.result() == "downloaded"
            if index % 10 == 0 or index == len(ids):
                print(f"images {index}/{len(ids)}; downloaded {downloaded}", flush=True)
    print(f"selected questions {args.count}; unique images {len(ids)}")


if __name__ == "__main__":
    main()
