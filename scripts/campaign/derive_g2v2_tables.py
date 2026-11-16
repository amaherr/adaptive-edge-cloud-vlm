"""Derive Predictor V2 partitions and measured-oracle tables from a G2 v2 campaign.

Steps (or ``all`` for features, profiles, partitions and the training oracle):

  features    pre-execution task features for the split's request IDs
  profiles    build_profile for every valid training/validation sub-block
  partitions  training rows, validation inputs and sealed validation targets
  oracle      measured-oracle attempt and cell tables (training by default)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_vlm.evaluation.measured_oracle import build_measured_oracle
from adaptive_vlm.prediction.v2.dataset.campaign_v2 import (
    prepare_campaign_partitions,
    profile_campaign,
    write_feature_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", choices=("features", "profiles", "partitions", "oracle", "all")
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Derived root")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        choices=("training", "validation", "evaluation"),
        help="profiles: default training validation; oracle: default training",
    )
    parser.add_argument("--unseal-evaluation", action="store_true")
    parser.add_argument("--sealed-root", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    if args.splits and "evaluation" in args.splits and not args.unseal_evaluation:
        parser.error("evaluation cells are sealed; add --unseal-evaluation")
    root = args.output
    features = root / "task_features"
    profiles = root / "profiles"
    steps = (
        ("features", "profiles", "partitions", "oracle")
        if args.step == "all"
        else (args.step,)
    )
    for step in steps:
        if step == "features":
            write_feature_bundle(args.campaign, args.contract, features)
        elif step == "profiles":
            profile_campaign(
                campaign=args.campaign,
                contract_path=args.contract,
                output=profiles,
                splits=tuple(args.splits or ("training", "validation")),
            )
        elif step == "partitions":
            receipt = prepare_campaign_partitions(
                campaign=args.campaign,
                profiles=profiles,
                contract_path=args.contract,
                feature_dir=features,
                output=root / "partitions",
                sealed_root=args.sealed_root,
            )
            print(
                json.dumps(
                    {
                        k: receipt[k]
                        for k in (
                            "classification",
                            "training_attempts",
                            "validation_attempts",
                            "training_load_levels",
                            "sealed_validation_path",
                        )
                    },
                    indent=2,
                )
            )
        else:
            splits = tuple(args.splits or ("training",))
            summary = build_measured_oracle(
                campaign=args.campaign,
                profiles=profiles,
                output=root / f"oracle-{'-'.join(splits)}",
                splits=splits,
                unseal_evaluation=args.unseal_evaluation,
            )
            for cell in summary["cells"]:
                medians = {
                    cid.split("/")[0]: round(s["median_ms"], 1)
                    for cid, s in cell["placements"].items()
                    if s["median_ms"] is not None
                }
                print(
                    cell["cell_id"],
                    medians,
                    "winner",
                    cell["winner"],
                    "tie" if cell["tie"] else "",
                )


if __name__ == "__main__":
    main()
