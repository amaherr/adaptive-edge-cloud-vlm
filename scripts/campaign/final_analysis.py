"""Phase 7 final analysis: G5 breakdowns and the thesis figures.

Steps:

  g5       extended G5 analysis (paired CIs by load state, trace and
           repetition; communication; overhead; stability; state changes)
  figures  G3 predicted against observed (validation, r2_component) and the
           G5 per-trace timelines, as static SVG (light and dark)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from adaptive_vlm.prediction.r2.g5 import analyse, phase7, requests
from adaptive_vlm.prediction.r2.units import CONFIGURATIONS, scored_units

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "baseline_report", ROOT / "scripts/campaign/report_g2v2plus_baseline.py"
)
assert _spec and _spec.loader
_baseline: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)
_text, _open_svg = _baseline._text, _baseline._open_svg
SHORT = {
    "edge/full": "edge/full",
    "cloud/raw_image": "cloud/raw_image",
    "distributed/projected/synchronous": "split, bf16",
    "distributed/projected/prompt_first": "split, prompt_first",
    "distributed/projected/int8": "split, int8",
}


def g5_step(args: argparse.Namespace) -> None:
    features = json.loads(args.features.read_text())["tasks"]
    records = requests(args.campaign)
    summary = analyse(
        records,
        calibration=json.loads(args.calibration.read_text()),
        features=features,
        best_static="cloud/raw_image",
        sla_ms=1500.0,
    )
    result = {"summary": summary, "phase7": phase7(records, features)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "g5_final.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(result["phase7"]["overhead"], indent=1))


def _log_axis(
    value: float, low: float, high: float, origin: float, size: float
) -> float:
    span = math.log(high) - math.log(low)
    return origin + size * (math.log(value) - math.log(low)) / span


def g3_scatter_svg(points: dict[str, list[tuple[float, float]]]) -> str:
    """Small multiples: predicted against observed per configuration."""

    width, height = 1080, 330
    panel, gap, left, top = 180, 26, 70, 92
    low, high = 200.0, 7000.0
    parts = _open_svg(
        width,
        height,
        "G3 validation: predicted against observed",
        "One panel per configuration: each dot is a validation request-cell, "
        "predicted latency against observed latency on log axes, with the "
        "identity line and a plus or minus 30 percent band.",
    )
    parts += [
        _text(
            24,
            32,
            "G3 validation (V01-V05): r2_component predicted against observed latency",
            "ink",
            font_size=16,
            font_weight=600,
        ),
        _text(
            24,
            52,
            "Each dot is one request-cell (630 per configuration). "
            "Diagonal: perfect prediction; dashed: ±30%.",
            "ink2",
            font_size=12,
        ),
    ]
    for index, configuration in enumerate(CONFIGURATIONS):
        ox = left + index * (panel + gap)
        oy = top
        parts.append(
            _text(
                ox, oy - 12, SHORT[configuration], "ink", font_size=12, font_weight=600
            )
        )
        for tick in (250, 500, 1000, 2000, 5000):
            x = _log_axis(tick, low, high, ox, panel)
            y = oy + panel - _log_axis(tick, low, high, 0, panel)
            parts.append(
                f'<line class="grid" x1="{ox}" x2="{ox + panel}" '
                f'y1="{y:.1f}" y2="{y:.1f}"/>'
            )
            if tick >= 500:  # 0.25 s would sit on the panel border
                parts.append(
                    _text(
                        x,
                        oy + panel + 15,
                        f"{tick / 1000:g}",
                        "muted",
                        font_size=10,
                        text_anchor="middle",
                    )
                )
            if index == 0:
                parts.append(
                    _text(
                        ox - 6,
                        y + 3,
                        f"{tick / 1000:g}",
                        "muted",
                        font_size=10,
                        text_anchor="end",
                    )
                )

        def xy(
            p: float, o: float, ox: float = ox, oy: float = oy
        ) -> tuple[float, float]:
            return (
                _log_axis(o, low, high, ox, panel),
                oy + panel - _log_axis(p, low, high, 0, panel),
            )

        for factor, dash in (
            (1.0, ""),
            (1.3, ' stroke-dasharray="4 3"'),
            (1 / 1.3, ' stroke-dasharray="4 3"'),
        ):
            x1, y1 = xy(low * factor, low)
            x2, y2 = xy(high * factor, high)
            parts.append(
                f'<line class="axis" x1="{x1:.1f}" y1="{y1:.1f}" '
                f'x2="{x2:.1f}" y2="{y2:.1f}"{dash}/>'
            )
        parts.append(
            f'<rect x="{ox}" y="{oy}" width="{panel}" height="{panel}" '
            'fill="none" class="axis"/>'
        )
        for predicted, observed in points[configuration]:
            if low <= predicted <= high and low <= observed <= high:
                x, y = xy(predicted, observed)
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" '
                    'class="f-cloud" fill-opacity="0.35"/>'
                )
    parts.append(
        _text(
            left + 2.5 * (panel + gap) - gap / 2,
            top + panel + 34,
            "Observed latency (s, log)",
            "muted",
            font_size=11,
            text_anchor="middle",
        )
    )
    parts.append(
        _text(
            20,
            top + panel / 2,
            "Predicted (s, log)",
            "muted",
            font_size=11,
            text_anchor="middle",
            transform=f"rotate(-90 20 {top + panel / 2})",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def g5_traces_svg(summary: dict[str, Any]) -> str:
    """Per trace: mean latency per segment for four policies."""

    segments: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for subblock, row in summary["by_segment"].items():
        trace, rest = subblock.split("-s", 1)
        segments[trace][int(rest.split("-", 1)[0])].append(row)
    series = (
        ("cloud", "always cloud", "best_static_mean_ms"),
        ("edge", "always edge", "edge_mean_ms"),
        ("synchronous", "adaptive", "adaptive_mean_ms"),
        ("int8", "oracle", "oracle_mean_ms"),
    )
    width, height = 1080, 430
    panel_w, panel_h, gap, left, top = 215, 250, 50, 70, 96
    y_max = 1500.0
    parts = _open_svg(
        width,
        height,
        "G5 traces: latency per segment",
        "For each of the four traces, the mean latency per segment of the "
        "adaptive policy, always-cloud, always-edge and the per-request oracle, "
        "averaged over four repetitions.",
    )
    parts += [
        _text(
            24,
            32,
            "G5 on AWS: mean latency per trace segment (four repetitions)",
            "ink",
            font_size=16,
            font_weight=600,
        ),
        _text(
            24,
            52,
            "Adaptive includes its sensing and selection time; the oracle "
            "is the fastest fixed configuration per request, in hindsight.",
            "ink2",
            font_size=12,
        ),
    ]
    order = ("cloud", "edge", "mixed", "net")
    titles = {
        "cloud": "Cloud-load episodes (E01)",
        "edge": "Edge-load episodes under C2 (E05)",
        "mixed": "Both loads (E03)",
        "net": "Network changes under C2",
    }
    for index, trace in enumerate(order):
        ox = left + index * (panel_w + gap)
        oy = top
        parts.append(
            _text(ox, oy - 14, titles[trace], "ink", font_size=12, font_weight=600)
        )
        for ms in range(0, 1600, 250):
            y = oy + panel_h - panel_h * ms / y_max
            parts.append(
                f'<line class="{"axis" if ms == 0 else "grid"}" x1="{ox}" '
                f'x2="{ox + panel_w}" y1="{y:.1f}" y2="{y:.1f}"/>'
            )
            if index == 0:
                parts.append(
                    _text(
                        ox - 6,
                        y + 4,
                        f"{ms / 1000:g}",
                        "muted",
                        font_size=10,
                        text_anchor="end",
                    )
                )
        rows = segments[trace]
        xs = {s: ox + 15 + (panel_w - 30) * s / 4 for s in rows}
        for s, members in sorted(rows.items()):
            state = members[0]["state"].split(" ", 1)[1]
            label = (members[0]["state"] if trace == "net" else state).replace(
                " L", "\nL"
            )
            for line_no, line in enumerate(label.split("\n")):
                parts.append(
                    _text(
                        xs[s],
                        oy + panel_h + 15 + 12 * line_no,
                        line,
                        "muted",
                        font_size=9,
                        text_anchor="middle",
                    )
                )
        for name, label, field in series:
            points = []
            for s, members in sorted(rows.items()):
                if field == "edge_mean_ms":
                    value = mean(m.get("edge_mean_ms", float("nan")) for m in members)
                else:
                    value = mean(m[field] for m in members)
                points.append(
                    (xs[s], oy + panel_h - panel_h * min(value, y_max) / y_max, value)
                )
            if any(math.isnan(v) for *_, v in points):
                continue
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            width_px = 3 if label == "adaptive" else 2
            parts.append(
                f'<polyline class="k-{name}" fill="none" '
                f'stroke-width="{width_px}" points="{path}"/>'
            )
            for x, y, value in points:
                parts.append(
                    f"<g><title>{label}: {value:.0f} ms</title><circle "
                    f'cx="{x:.1f}" cy="{y:.1f}" r="3.5" class="f-{name} ring" '
                    'stroke-width="1.5"/></g>'
                )
    legend_y = height - 16
    lx = left
    for name, label, _ in series:
        parts.append(
            f'<line class="k-{name}" x1="{lx}" x2="{lx + 22}" '
            f'y1="{legend_y - 4}" y2="{legend_y - 4}" stroke-width="2.5"/>'
        )
        parts.append(_text(lx + 28, legend_y, label, "ink2", font_size=12))
        lx += 60 + 7 * len(label)
    parts.append(
        _text(
            24,
            top + panel_h / 2,
            "Mean latency (s)",
            "muted",
            font_size=11,
            text_anchor="middle",
            transform=f"rotate(-90 24 {top + panel_h / 2})",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def figures_step(args: argparse.Namespace) -> None:
    audit = json.loads((args.partitions / "partition_audit.json").read_text())
    units = scored_units(
        args.partitions / "validation_inputs.jsonl",
        Path(audit["sealed_validation_path"]).read_bytes().splitlines(),
    )
    observed = {
        (u.inputs.request_id, u.inputs.cell_id, u.configuration): u.targets["e2e"]
        for u in units
        if u.successes
    }
    points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for line in args.predictions.read_text().splitlines():
        record = json.loads(line)
        if record["model"] != "r2_component" or record["prediction"]["total"] is None:
            continue
        key = (record["request_id"], record["cell_id"], record["configuration"])
        points[record["configuration"]].append(
            (record["prediction"]["total"], observed[key])
        )
    final = json.loads((args.output / "g5_final.json").read_text())
    summary = final["summary"]
    # The per-segment table lacks always-edge; add it from the request records.
    records = requests(args.campaign)
    edge: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["fixed_ms"].get("edge/full") is not None:
            edge[r["subblock_id"]].append(r["fixed_ms"]["edge/full"])
    for subblock, row in summary["by_segment"].items():
        row["edge_mean_ms"] = mean(edge[subblock])
    (args.output / "g3_predicted_vs_observed.svg").write_text(g3_scatter_svg(points))
    (args.output / "g5_traces.svg").write_text(g5_traces_svg(summary))
    print("figures written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("g5", "figures"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "configs/g5/calibration_r2_component.json",
    )
    parser.add_argument(
        "--features", type=Path, default=ROOT / "configs/g5/task_features-short.json"
    )
    parser.add_argument("--partitions", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (g5_step if args.step == "g5" else figures_step)(args)


if __name__ == "__main__":
    main()
