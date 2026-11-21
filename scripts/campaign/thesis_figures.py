"""Six supplementary thesis figures, as static SVG (light and dark).

  load_effects.svg          G2 v2+ training cells: cloud load on the cloud,
                            edge load on the edge, per network condition
  static_regret.svg         G2 v2+ training cells: mean regret of each static
                            configuration against the per-cell oracle
  g3_ape_cdf.svg            G3 validation: absolute percentage error CDF per
                            configuration (r2_component)
  g4_regret.svg             G4 evaluation cells: mean regret by load state for
                            the selector and the static policies
  bytes_vs_latency.svg      G5: bytes per request and latency per policy
  network_verification.svg  G2 v2+: achieved rate and RTT of every probe
                            against the shaped target

Inputs are committed records, plus the git-ignored campaign mirror (network
probes) and the predictor partitions (G3 targets); see docs/reproducibility.md.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from adaptive_vlm.prediction.r2.units import CONFIGURATIONS, scored_units

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "baseline_report", ROOT / "scripts/campaign/report_g2v2plus_baseline.py"
)
assert _spec and _spec.loader
_baseline: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)
_text = _baseline._text

CAMPAIGN = ROOT / "results/aws-campaign-20261119"
FINAL = ROOT / "results/final-analysis-20261121"
# The reference categorical palette (validated, light / dark), in slot order.
COLORS = {
    **_baseline.COLORS,
    "prompt_first": ("#e87ba4", "#d55181"),
    "adaptive": ("#6250d6", "#9085e9"),
}
KEY = {
    "edge/full": "edge",
    "cloud/raw_image": "cloud",
    "distributed/projected/synchronous": "synchronous",
    "distributed/projected/prompt_first": "prompt_first",
    "distributed/projected/int8": "int8",
}
LABEL = {
    "edge/full": "edge/full",
    "cloud/raw_image": "cloud/raw_image",
    "distributed/projected/synchronous": "split, bf16",
    "distributed/projected/prompt_first": "split, prompt-first",
    "distributed/projected/int8": "split, int8",
}
DIR_TO_CONFIGURATION = {
    "edge": "edge/full",
    "cloud": "cloud/raw_image",
    "distributed": "distributed/projected/synchronous",
    "distributed_prompt_first": "distributed/projected/prompt_first",
    "distributed_int8": "distributed/projected/int8",
}


def _style() -> str:
    def rules(index: int, indent: str) -> str:
        return "\n".join(
            f"{indent}.f-{n} {{ fill: {c[index]}; }} .k-{n} {{ stroke: {c[index]}; }}"
            for n, c in COLORS.items()
        )

    style: str = _baseline.STYLE % {
        "series_light": rules(0, "  ") + "\n  .f-neutral { fill: #898781; } "
        ".k-neutral { stroke: #898781; }",
        "series_dark": rules(1, "    ") + "\n    .f-neutral { fill: #898781; } "
        ".k-neutral { stroke: #898781; }",
    }
    return style


def _open(width: int, height: int, title: str, desc: str, sub: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{title}</title>',
        f'<desc id="desc">{desc}</desc>',
        _style(),
        f'<rect class="surface" width="{width}" height="{height}"/>',
        _text(24, 32, title, "ink", font_size=16, font_weight=600),
        _text(24, 52, sub, "ink2", font_size=12),
    ]


def _line(x1: float, y1: float, x2: float, y2: float, cls: str, extra: str = "") -> str:
    return (
        f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
        f'y2="{y2:.1f}"{extra}/>'
    )


def _swatch(x: float, y: float, key: str, label: str) -> list[str]:
    return [
        f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" '
        f'class="f-{key}"/>',
        _text(x + 15, y, label, "ink2", font_size=11),
    ]


def _linear(value: float, low: float, high: float, origin: float, size: float) -> float:
    return origin + size * (value - low) / (high - low)


def _log(value: float, low: float, high: float, origin: float, size: float) -> float:
    return origin + size * (math.log(value) - math.log(low)) / (
        math.log(high) - math.log(low)
    )


def _bar(x: float, y: float, w: float, h: float, cls: str) -> str:
    """A horizontal bar with a 4 px rounded data end, anchored at x."""

    if w <= 0:
        return ""
    r = min(4.0, h / 2, w)
    return (
        f'<path class="{cls}" d="M{x:.1f},{y:.1f} h{w - r:.1f} '
        f"a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} a{r},{r} 0 0 1 {-r},{r} "
        f'h{-(w - r):.1f} z"/>'
    )


def _end(parts: list[str]) -> str:
    return "\n".join([*parts, "</svg>"]) + "\n"


# --------------------------------------------------------------------------
# 1. Load effects


def _training_medians() -> dict[tuple[str, str, str, str], float]:
    """(condition, edge load, cloud load, configuration) -> median over reps."""

    values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    with (CAMPAIGN / "subblock_medians.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "training" or row["task"] != "short":
                continue
            key = (
                row["condition"],
                row["edge_load"],
                row["cloud_load"],
                DIR_TO_CONFIGURATION[row["configuration_dir"]],
            )
            values[key].append(float(row["median_ms"]))
    return {k: median(v) for k, v in values.items()}


def load_effects_svg() -> str:
    data = _training_medians()
    conditions = sorted({k[0] for k in data})
    width, row_h, left, top = 1000, 24, 150, 112
    panel_w, gap = 330, 140
    height = top + row_h * len(conditions) + 70
    rates = {}
    with (CAMPAIGN / "subblock_medians.csv").open() as handle:
        for row in csv.DictReader(handle):
            rates[row["condition"]] = (row["rate_mbps"], row["added_rtt_ms"])
    cloud = [
        (
            data[(c, "L0", "C0", "cloud/raw_image")],
            data[(c, "L0", "C2", "cloud/raw_image")],
        )
        for c in conditions
    ]
    edge = [
        (data[(c, "L0", "C0", "edge/full")], data[(c, "L1", "C0", "edge/full")])
        for c in conditions
    ]
    # The headline effects are the baseline report's paired medians.
    effects = json.loads((CAMPAIGN / "baseline/baseline_summary.json").read_text())[
        "load_effects"
    ]
    added = effects["cloud"]["c2_minus_c0_at_L0_ms"]["median"]
    ratio = effects["edge"]["l1_over_l0_at_C0"]["median"]
    parts = _open(
        width,
        height,
        "Each load acts on its own side, by a fixed amount",
        "Two dumbbell panels over the twelve training network conditions. Left: "
        "cloud/raw_image median latency without and with cloud GPU load. Right: "
        "edge/full median latency without and with edge GPU load.",
        "G2 v2+ training cells, short task; median latency per condition "
        "(two repetitions). Hollow: unloaded; filled: loaded.",
    )
    panels = (
        (
            "cloud",
            f"Cloud load (C0 → C2) on cloud/raw_image: +{added:.0f} ms",
            cloud,
            0,
            2200,
        ),
        ("edge", f"Edge load (L0 → L1) on edge/full: ×{ratio:.3f}", edge, 0, 2200),  # noqa: RUF001
    )
    for index, (key, title, pairs, low, high) in enumerate(panels):
        ox = left + index * (panel_w + gap)
        parts.append(_text(ox, top - 30, title, "ink", font_size=12, font_weight=600))
        for tick in range(0, 2201, 500):
            x = _linear(tick, low, high, ox, panel_w)
            parts.append(_line(x, top - 12, x, top + row_h * len(conditions), "grid"))
            parts.append(
                _text(
                    x,
                    top + row_h * len(conditions) + 16,
                    f"{tick / 1000:g}",
                    "muted",
                    font_size=10,
                    text_anchor="middle",
                )
            )
        for position, (condition, (a, b)) in enumerate(
            zip(conditions, pairs, strict=True)
        ):
            y = top + position * row_h
            if index == 0:
                rate, rtt = rates[condition]
                parts.append(
                    _text(
                        left - 12,
                        y + 4,
                        f"{condition}  {float(rate):g} Mb/s, +{float(rtt):g} ms",
                        "ink2",
                        font_size=11,
                        text_anchor="end",
                    )
                )
            xa, xb = (_linear(v, low, high, ox, panel_w) for v in (a, b))
            parts.append(_line(xa, y, xb, y, "axis", ' stroke-width="2"'))
            parts.append(
                f'<circle cx="{xa:.1f}" cy="{y:.1f}" r="4.5" class="surface k-{key}" '
                'stroke-width="2"/>'
            )
            parts.append(
                f'<circle cx="{xb:.1f}" cy="{y:.1f}" r="5" class="f-{key} ring" '
                'stroke-width="2"/>'
            )
        parts.append(
            _text(
                ox + panel_w / 2,
                top + row_h * len(conditions) + 34,
                "Median latency (s)",
                "muted",
                font_size=11,
                text_anchor="middle",
            )
        )
    return _end(parts)


# --------------------------------------------------------------------------
# 2. Static-policy regret


def static_regret_svg() -> str:
    summary = json.loads((CAMPAIGN / "baseline/baseline_summary.json").read_text())
    per_cell = summary["static_policies"]["per_cell"]
    names = {v: k for k, v in KEY.items()}
    rows = sorted(per_cell.items(), key=lambda item: item[1]["mean_regret_ms"])
    width, left, top, bar_h, step = 900, 190, 96, 22, 38
    plot_w = 520
    height = top + step * len(rows) + 56
    high = 1400.0
    parts = _open(
        width,
        height,
        "Always-cloud is the best static policy, but still loses where loads bite",
        "Horizontal bars: mean regret against the per-cell measured oracle for "
        "each static configuration over the 54 G2 v2+ training cells.",
        "G2 v2+ training cells (54); regret = the static choice's median latency "
        "minus the fastest configuration's, per cell.",
    )
    for tick in range(0, 1401, 200):
        x = _linear(tick, 0, high, left, plot_w)
        parts.append(_line(x, top - 8, x, top + step * len(rows) - 10, "grid"))
        parts.append(
            _text(
                x,
                top + step * len(rows) + 6,
                f"{tick:g}",
                "muted",
                font_size=10,
                text_anchor="middle",
            )
        )
    for index, (key, stats) in enumerate(rows):
        y = top + index * step
        w = _linear(stats["mean_regret_ms"], 0, high, 0, plot_w)
        parts.append(
            _text(
                left - 12,
                y + bar_h / 2 + 4,
                LABEL[names[key]],
                "ink2",
                font_size=12,
                text_anchor="end",
            )
        )
        parts.append(_bar(left, y, w, bar_h, f"f-{key}"))
        parts.append(
            _text(
                left + w + 8,
                y + bar_h / 2 + 4,
                f"{stats['mean_regret_ms']:.0f} ms "
                f"({stats['mean_relative_regret_percent']:.0f}%), "
                f"not best in {stats['cells_not_best']} of 54 cells",
                "ink",
                font_size=11,
            )
        )
    parts.append(
        _text(
            left + plot_w / 2,
            top + step * len(rows) + 26,
            "Mean regret against the per-cell oracle (ms)",
            "muted",
            font_size=11,
            text_anchor="middle",
        )
    )
    return _end(parts)


# --------------------------------------------------------------------------
# 3. G3 APE CDF


def _ape(partitions: Path, predictions: Path) -> dict[str, list[float]]:
    audit = json.loads((partitions / "partition_audit.json").read_text())
    units = scored_units(
        partitions / "validation_inputs.jsonl",
        Path(audit["sealed_validation_path"]).read_bytes().splitlines(),
    )
    observed = {
        (u.inputs.request_id, u.inputs.cell_id, u.configuration): u.targets["e2e"]
        for u in units
        if u.successes
    }
    apes: dict[str, list[float]] = defaultdict(list)
    for line in predictions.read_text().splitlines():
        record = json.loads(line)
        if record["model"] != "r2_component" or record["prediction"]["total"] is None:
            continue
        key = (record["request_id"], record["cell_id"], record["configuration"])
        obs = observed[key]
        apes[record["configuration"]].append(
            100 * abs(record["prediction"]["total"] - obs) / obs
        )
    return apes


def g3_ape_svg(apes: dict[str, list[float]]) -> str:
    width, height, left, top, plot_w, plot_h = 900, 420, 80, 110, 560, 240
    high = 15.0
    parts = _open(
        width,
        height,
        "Latency is predictable before running: 90% of predictions within 6%",
        "Cumulative distribution of absolute percentage error of the frozen "
        "r2_component predictor on the validation cells, one line per configuration.",
        "G3 validation cells (V01–V05), r2_component; share of request-cells with "  # noqa: RUF001
        "absolute percentage error at most x. Targets: median ≤ 15%, P90 ≤ 30%.",
    )
    for tick in range(0, 16, 3):
        x = _linear(tick, 0, high, left, plot_w)
        parts.append(_line(x, top, x, top + plot_h, "grid"))
        parts.append(
            _text(
                x,
                top + plot_h + 16,
                f"{tick}%",
                "muted",
                font_size=10,
                text_anchor="middle",
            )
        )
    for share in (0, 0.25, 0.5, 0.75, 0.9, 1.0):
        y = top + plot_h * (1 - share)
        dash = ' stroke-dasharray="4 3"' if share == 0.9 else ""
        parts.append(
            _line(left, y, left + plot_w, y, "axis" if share == 0.9 else "grid", dash)
        )
        parts.append(
            _text(
                left - 8,
                y + 3,
                f"{share:.0%}" if share != 0.9 else "90%",
                "muted",
                font_size=10,
                text_anchor="end",
            )
        )
    legend_y = top - 22
    lx = left
    for configuration in CONFIGURATIONS:
        values = sorted(apes[configuration])
        n = len(values)
        points = []
        for i, v in enumerate(values):
            x = _linear(min(v, high), 0, high, left, plot_w)
            points.append(f"{x:.1f},{top + plot_h * (1 - (i + 1) / n):.1f}")
        start = f"{left:.1f},{top + plot_h:.1f}"
        parts.append(
            f'<polyline fill="none" stroke-width="2" class="k-{KEY[configuration]}" '
            f'points="{start} {" ".join(points)}"/>'
        )
        p90 = values[math.ceil(0.9 * n) - 1]
        parts += _swatch(
            lx, legend_y, KEY[configuration], f"{LABEL[configuration]} (P90 {p90:.1f}%)"
        )
        lx += 160
    parts.append(
        _text(
            left + plot_w / 2,
            top + plot_h + 34,
            "Absolute percentage error (axis clipped at 15%)",
            "muted",
            font_size=11,
            text_anchor="middle",
        )
    )
    parts.append(
        _text(
            22,
            top + plot_h / 2,
            "Share of request-cells",
            "muted",
            font_size=11,
            text_anchor="middle",
            transform=f"rotate(-90 22 {top + plot_h / 2})",
        )
    )
    return _end(parts)


# --------------------------------------------------------------------------
# 4. G4 regret by load state


def g4_regret_svg() -> str:
    rows = [
        json.loads(line)
        for line in (CAMPAIGN / "g4/evaluation/decisions.jsonl")
        .read_text()
        .splitlines()
    ]
    states = ("L0/C0", "L1/C0", "L0/C2", "L1/C2")
    policies = (
        ("selector", "adaptive", "selector (r2 + select)"),
        ("always_cloud", "cloud", "always cloud"),
        ("always_edge", "edge", "always edge"),
    )
    means = {
        (s, p): mean(
            r["regret"][p]["regret_ms"]
            for r in rows
            if r["complete"] and r["load_state"] == s
        )
        for s in states
        for p, _, _ in policies
    }
    counts = {
        s: sum(1 for r in rows if r["complete"] and r["load_state"] == s)
        for s in states
    }
    width, left, top, plot_w = 960, 150, 110, 620
    bar_h, inner, group = 16, 2, 72
    height = top + group * len(states) + 50
    high = 1400.0
    parts = _open(
        width,
        height,
        "Selecting per request removes the regret of any fixed choice",
        "Grouped horizontal bars: mean regret against the per-request oracle on "
        "the G4 evaluation cells, by load state, for the selector, always-cloud "
        "and always-edge.",
        "G4 evaluation cells (E01–E05, unseen networks), 630 request-cells; mean "  # noqa: RUF001
        "regret against the measured oracle of the same request-cell.",
    )
    lx = left
    for _, key, label in policies:
        parts += _swatch(lx, top - 24, key, label)
        lx += 190
    for tick in range(0, 1401, 200):
        x = _linear(tick, 0, high, left, plot_w)
        parts.append(_line(x, top - 6, x, top + group * len(states) - 16, "grid"))
        parts.append(
            _text(
                x,
                top + group * len(states) - 2,
                f"{tick}",
                "muted",
                font_size=10,
                text_anchor="middle",
            )
        )
    for g, state in enumerate(states):
        y0 = top + g * group
        parts.append(
            _text(
                left - 12,
                y0 + 1.5 * bar_h + 6,
                f"{state}  (n = {counts[state]})",
                "ink2",
                font_size=12,
                text_anchor="end",
            )
        )
        for i, (policy, key, _) in enumerate(policies):
            value = means[(state, policy)]
            y = y0 + i * (bar_h + inner)
            w = _linear(min(value, high), 0, high, 0, plot_w)
            parts.append(_bar(left, y, max(w, 0.0), bar_h, f"f-{key}"))
            parts.append(
                _text(
                    left + max(w, 0) + 6,
                    y + bar_h - 4,
                    f"{value:.0f} ms",
                    "ink",
                    font_size=10,
                )
            )
    parts.append(
        _text(
            left + plot_w / 2,
            top + group * len(states) + 16,
            "Mean regret against the oracle (ms)",
            "muted",
            font_size=11,
            text_anchor="middle",
        )
    )
    return _end(parts)


# --------------------------------------------------------------------------
# 5. Bytes and latency (G5)


def bytes_latency_svg() -> str:
    final = json.loads((FINAL / "g5_final.json").read_text())
    nbytes = final["phase7"]["bytes_per_request_mean"]
    latency = {k: v["mean_ms"] for k, v in final["summary"]["latency"].items()}
    order = ["adaptive", "oracle", *CONFIGURATIONS]
    order.sort(key=lambda k: latency[k])
    label = {**LABEL, "adaptive": "adaptive (G5)", "oracle": "oracle (hindsight)"}
    color = {**KEY, "adaptive": "adaptive", "oracle": "neutral"}
    width, left, top, step, bar_h = 1000, 170, 96, 34, 20
    panel_w, gap = 320, 90
    height = top + step * len(order) + 56
    parts = _open(
        width,
        height,
        "The adaptive policy sends less than always-cloud and is faster",
        "Two aligned bar panels per policy in G5: mean application bytes per "
        "request, and mean end-to-end latency.",
        "G5 on AWS, 2,400 measured requests per policy; means per request. "
        "Adaptive latency includes sensing and selection.",
    )
    values: tuple[dict[str, float], dict[str, float]] = (
        {k: v / 1000 for k, v in nbytes.items()},
        latency,
    )
    panels = (
        ("Bytes sent per request (KB)", values[0], 3000.0, 500),
        ("Mean latency (ms)", values[1], 2000.0, 500),
    )
    for p, (title, value, high, step_tick) in enumerate(panels):
        ox = left + p * (panel_w + gap)
        parts.append(_text(ox, top - 14, title, "ink", font_size=12, font_weight=600))
        tick = 0.0
        while tick <= high:
            x = _linear(tick, 0, high, ox, panel_w)
            parts.append(_line(x, top - 4, x, top + step * len(order) - 10, "grid"))
            parts.append(
                _text(
                    x,
                    top + step * len(order) + 4,
                    f"{tick:g}",
                    "muted",
                    font_size=10,
                    text_anchor="middle",
                )
            )
            tick += step_tick
        for i, key in enumerate(order):
            y = top + i * step
            if p == 0:
                parts.append(
                    _text(
                        left - 12,
                        y + bar_h / 2 + 4,
                        label[key],
                        "ink2",
                        font_size=12,
                        text_anchor="end",
                    )
                )
            v = value[key]
            w = _linear(v, 0, high, 0, panel_w)
            parts.append(_bar(ox, y, w, bar_h, f"f-{color[key]}"))
            parts.append(
                _text(ox + w + 6, y + bar_h / 2 + 4, f"{v:,.0f}", "ink", font_size=10)
            )
    return _end(parts)


# --------------------------------------------------------------------------
# 6. Network verification


def _probes(campaign: Path) -> list[dict[str, Any]]:
    grid = json.loads(
        (ROOT / "configs/profiles/aws_network_grid_v2plus.json").read_text()
    )
    conditions = grid["conditions"]
    baseline = json.loads((campaign / "network_baseline.json").read_text())
    base_rtt = float(baseline["echo_median_rtt_ms"])
    out: list[dict[str, Any]] = []
    for directory in sorted((campaign / "subblocks").iterdir()):
        condition = conditions[directory.name.split("-", 1)[0]]
        for name in ("network_pre.json", "network_post.json"):
            probe = json.loads((directory / name).read_text())
            target_bps = condition["rate_mbps"] * 1e6
            target_rtt = base_rtt + condition["added_rtt_ms"]
            for direction in (
                "edge_to_cloud_receiver_bps",
                "cloud_to_edge_receiver_bps",
            ):
                out.append(
                    {
                        "rate": condition["rate_mbps"],
                        "ratio": probe[direction] / target_bps,
                        "added": condition["added_rtt_ms"],
                        "rtt_error": probe["echo_median_rtt_ms"] - target_rtt,
                        "direction": direction,
                    }
                )
    return out


def network_svg(probes: list[dict[str, Any]]) -> str:
    width, height, top, plot_w, plot_h = 1000, 400, 110, 360, 210
    lefts = (80, 580)
    n = len(probes) // 2
    parts = _open(
        width,
        height,
        "Every shaped condition was verified, before and after each sub-block",
        "Two scatter panels over all G2 v2+ network probes. Left: achieved receiver "
        "rate as a fraction of the target rate. Right: echo RTT minus the target RTT. "
        "Tolerance bands shaded.",
        f"G2 v2+, 196 sub-blocks × pre and post probes ({n} probes; rate in both "  # noqa: RUF001
        "directions). Shaded: the protocol's tolerance; none outside it.",
    )
    # Left: rate ratio against target rate (log x).
    ox = lefts[0]
    lo, hi, ylo, yhi = 4.0, 250.0, 0.75, 1.05
    yband = top + plot_h * (yhi - 0.8) / (yhi - ylo)
    parts.append(
        f'<rect x="{ox}" y="{top}" width="{plot_w}" height="{yband - top:.1f}" '
        'class="f-neutral" fill-opacity="0.12"/>'
    )
    parts.append(
        _text(
            ox,
            top - 14,
            "Achieved rate ÷ target rate",
            "ink",
            font_size=12,
            font_weight=600,
        )
    )
    for t in (5, 10, 20, 50, 100, 200):
        x = _log(t, lo, hi, ox, plot_w)
        parts.append(_line(x, top, x, top + plot_h, "grid"))
        parts.append(
            _text(
                x,
                top + plot_h + 15,
                f"{t}",
                "muted",
                font_size=10,
                text_anchor="middle",
            )
        )
    for v in (0.8, 0.9, 1.0):
        y = top + plot_h * (yhi - v) / (yhi - ylo)
        parts.append(_line(ox, y, ox + plot_w, y, "axis" if v == 1.0 else "grid"))
        parts.append(
            _text(ox - 6, y + 3, f"{v:.1f}", "muted", font_size=10, text_anchor="end")
        )
    for p in probes:
        x = _log(p["rate"], lo, hi, ox, plot_w)
        y = top + plot_h * (yhi - p["ratio"]) / (yhi - ylo)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" class="f-cloud" '
            'fill-opacity="0.4"/>'
        )
    parts.append(
        _text(
            ox + plot_w / 2,
            top + plot_h + 34,
            "Target rate (Mb/s, log)",
            "muted",
            font_size=11,
            text_anchor="middle",
        )
    )
    # Right: RTT error against added RTT.
    ox = lefts[1]
    xlo, xhi, elo, ehi = 0.0, 160.0, -6.0, 6.0
    parts.append(
        _text(
            ox,
            top - 14,
            "Echo RTT − target RTT (ms)",  # noqa: RUF001
            "ink",
            font_size=12,
            font_weight=600,
        )
    )
    band = [(a, max(5.0, 0.2 * a)) for a in range(0, 161, 5)]
    upper = " ".join(
        f"{_linear(a, xlo, xhi, ox, plot_w):.1f},"
        f"{top + plot_h * (ehi - min(t, ehi)) / (ehi - elo):.1f}"
        for a, t in band
    )
    lower = " ".join(
        f"{_linear(a, xlo, xhi, ox, plot_w):.1f},"
        f"{top + plot_h * (ehi + min(t, -elo)) / (ehi - elo):.1f}"
        for a, t in reversed(band)
    )
    parts.append(
        f'<polygon points="{upper} {lower}" class="f-neutral" fill-opacity="0.12"/>'
    )
    for t in (5, 20, 50, 100, 150):
        x = _linear(t, xlo, xhi, ox, plot_w)
        parts.append(_line(x, top, x, top + plot_h, "grid"))
        parts.append(
            _text(
                x,
                top + plot_h + 15,
                f"{t}",
                "muted",
                font_size=10,
                text_anchor="middle",
            )
        )
    for v in (-5, 0, 5):
        y = top + plot_h * (ehi - v) / (ehi - elo)
        parts.append(_line(ox, y, ox + plot_w, y, "axis" if v == 0 else "grid"))
        parts.append(
            _text(
                ox - 6,
                y + 3,
                f"{v:+g}" if v else "0",
                "muted",
                font_size=10,
                text_anchor="end",
            )
        )
    seen = set()
    for p in probes:
        if p["direction"] != "edge_to_cloud_receiver_bps":
            continue  # one RTT per probe
        key = (p["added"], round(p["rtt_error"], 3))
        if key in seen:
            continue
        seen.add(key)
        x = _linear(p["added"], xlo, xhi, ox, plot_w)
        y = top + plot_h * (ehi - p["rtt_error"]) / (ehi - elo)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" class="f-cloud" '
            'fill-opacity="0.4"/>'
        )
    parts.append(
        _text(
            ox + plot_w / 2,
            top + plot_h + 34,
            "Added RTT (ms)",
            "muted",
            font_size=11,
            text_anchor="middle",
        )
    )
    return _end(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        required=True,
        help="G2 v2+ campaign folder (mirror of the edge data volume)",
    )
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=CAMPAIGN / "predictor_r2/candidate/validation_predictions.jsonl",
    )
    parser.add_argument("--output", type=Path, default=FINAL)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = {
        "load_effects.svg": load_effects_svg(),
        "static_regret.svg": static_regret_svg(),
        "g3_ape_cdf.svg": g3_ape_svg(_ape(args.partitions, args.predictions)),
        "g4_regret.svg": g4_regret_svg(),
        "bytes_vs_latency.svg": bytes_latency_svg(),
        "network_verification.svg": network_svg(_probes(args.campaign)),
    }
    for name, svg in figures.items():
        (args.output / name).write_text(svg)
    print("written:", ", ".join(figures))


if __name__ == "__main__":
    main()
