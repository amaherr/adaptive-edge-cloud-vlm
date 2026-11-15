"""Render descriptive figures from saved training-only diagnostic residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    report = json.loads((root / "diagnosis.json").read_text())["reports"][
        "training_oof.jsonl"
    ]
    rows = [
        json.loads(line)
        for line in args.residuals.read_text().splitlines()
        if json.loads(line)["source"] == "training_oof.jsonl"
    ]
    configs = ["cloud/raw_image", "distributed/projected/synchronous", "edge/full"]
    labels = ["Cloud-PC", "Distributed", "Edge-only (diagnostic)"]
    components = [
        ("cloud_nondecode_including_prefill_ms", "Cloud nondecode", "#3579b5"),
        ("edge_vision_and_projector_ms", "Edge vision/projector", "#519b67"),
        ("edge_nondecode_ms", "Edge nondecode", "#519b67"),
        ("decode_ms", "Decode", "#db9a31"),
        ("transport_ms", "Transport residual", "#a778b4"),
        ("client_residual_ms", "Client bookkeeping", "#8c8c8c"),
        ("client_image_read_ms", "Image read", "#bcbcbc"),
        ("tensor_serialization_ms", "Serialization", "#d4d4d4"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for i, cid in enumerate(configs):
        bottom = 0.0
        for component, label, color in components:
            value = (
                report[cid]["components"]
                .get(component, {})
                .get("share_of_absolute_component_error", 0)
                * 100
            )
            if value:
                ax.bar(
                    i,
                    value,
                    bottom=bottom,
                    color=color,
                    label=label
                    if i == 0
                    or component
                    in {
                        "edge_vision_and_projector_ms",
                        "edge_nondecode_ms",
                        "tensor_serialization_ms",
                    }
                    else None,
                )
                bottom += value
    ax.set(
        xticks=range(3),
        xticklabels=labels,
        ylim=(0, 100),
        ylabel="Share of summed absolute component error (%)",
        title="Where V2-A training OOF errors originate",
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle(
        "60 independent request IDs; five training blocks; three repeats\n"
        "Component shares do not equal additive contributions to total MAE",
        fontsize=10,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(root / f"stage_error_attribution.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = {2: "#dc7732", 9: "#3579b5", 10: "#519b67"}
    for ax, cfg, stage, title in [
        (
            axes[0],
            "cloud/raw_image",
            "cloud_nondecode_including_prefill_ms",
            "Cloud-PC nondecode",
        ),
        (axes[1], "edge/full", "edge_nondecode_ms", "Edge nondecode (diagnostic only)"),
    ]:
        subset = [r for r in rows if r["configuration_id"] == cfg]
        for crop, color in colors.items():
            ids = sorted(
                {
                    r["request_id"]
                    for r in subset
                    if r["features"]["geometry_crop_count_diagnostic"] == crop
                }
            )
            points = [[r for r in subset if r["request_id"] == rid] for rid in ids]
            ax.scatter(
                [p[0]["features"]["image_pixels"] / 1e6 for p in points],
                [
                    median(r["observed_components_ms"][stage] for r in p) / 1000
                    for p in points
                ],
                label=f"{crop} crops, {len(ids)} IDs",
                color=color,
                s=30,
            )
        ax.set(
            xlabel="Input image megapixels",
            ylabel="Observed nondecode median (s)",
            title=title,
        )
        ax.legend(fontsize=8)
    fig.suptitle(
        "Discrete crop regimes are hidden by a smooth pixel-count feature\n"
        "One point per training request; median across training blocks/repeats",
        fontsize=11,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(root / f"crop_compute_regimes.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
