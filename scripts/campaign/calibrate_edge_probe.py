"""Pilot step (amendment A4): calibrate the G5 edge timing probe.

Runs the probe on the edge GPU without load (L0) and under the calibrated
edge load (L1), then applies the predeclared rule:

    threshold = sqrt(median(L0) * median(L1))
    go        iff p95(L0) < threshold < p05(L1)

On a go it writes the sensing configuration with the threshold filled in.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from adaptive_vlm.benchmark.edge_load import load_levels, make_generator
from adaptive_vlm.prediction.v1.calibration import percentile
from adaptive_vlm.scheduler.sensing import torch_edge_probe


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p05_ms": percentile(values, 0.05),
        "median_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-load-config", type=Path, required=True)
    parser.add_argument("--level", default="L1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--gap-s", type=float, default=0.25)
    parser.add_argument("--sensing-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensing-output", type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(args.sensing_template.read_text())
    probe = torch_edge_probe(
        args.device,
        int(template.get("edge_probe_matrix", 2048)),
        int(template.get("edge_probe_repetitions", 5)),
    )

    def sample() -> list[float]:
        values = []
        for _ in range(args.samples):
            values.append(probe())
            time.sleep(args.gap_s)
        return values

    unloaded = sample()
    level = load_levels(args.edge_load_config)[args.level]
    generator = make_generator(
        level, directory=args.output.parent / ".edge-probe-load", device=args.device
    )
    generator.start()
    try:
        time.sleep(2.0)
        generator.check()
        loaded = sample()
        generator.check()
    finally:
        generator.stop()
    low, high = _summary(unloaded), _summary(loaded)
    threshold = math.sqrt(low["median_ms"] * high["median_ms"])
    go = low["p95_ms"] < threshold < high["p05_ms"]
    result: dict[str, Any] = {
        "rule": "threshold = sqrt(median L0 * median L1); go iff p95 L0 < "
        "threshold < p05 L1",
        "L0": low,
        args.level: high,
        "threshold_ms": threshold,
        "go": go,
        "samples_ms": {"L0": unloaded, args.level: loaded},
    }
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if go:
        args.sensing_output.write_text(
            json.dumps(template | {"edge_probe_threshold_ms": threshold}, indent=1)
            + "\n"
        )
    print(json.dumps({k: result[k] for k in ("L0", args.level, "threshold_ms", "go")}))


if __name__ == "__main__":
    main()
