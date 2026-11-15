"""Freeze a seeded, exclusion-aware VQA v2 candidate order (G2 v2 protocol 4.1).

Each candidate is a question whose question and image are absent from every
exclusion manifest, with one question per image and exactly ten annotations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_vlm.data.vqa_selection import write_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest", type=Path, action="append", default=[], dest="exclude"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = write_candidates(
        args.output,
        questions_path=args.questions,
        annotations_path=args.annotations,
        exclusion_manifests=args.exclude,
        seed=args.seed,
        count=args.count,
    )
    print(
        f"froze {len(record['candidates'])} candidates; excluded "
        f"{record['excluded_question_count']} questions and "
        f"{record['excluded_image_count']} images"
    )


if __name__ == "__main__":
    main()
