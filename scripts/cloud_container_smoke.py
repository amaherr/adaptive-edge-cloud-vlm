"""Run 12 training requests through each actual placement; retain failure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    image = os.environ.get("APP_IMAGE", "")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise RuntimeError(
            "Run inside the final digest-deployed image with APP_IMAGE set"
        )
    # The frozen source manifest remains untouched. Use only training IDs for
    # diagnostic acceptance so final-evaluation requests stay unexamined.
    split_path = ROOT / "configs/evaluation/cloud_vqa_split_v1.json"
    split = json.loads(split_path.read_text())
    original = ROOT / "data/manifests/physical_vqa_v1.jsonl"
    source_rows = [json.loads(line) for line in original.read_text().splitlines()]
    selected_ids = split["training_request_ids"][:12]
    lookup = {row["request_id"]: row for row in source_rows}
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = args.output_root / "smoke-manifest.jsonl"
    selected = []
    for request_id in selected_ids:
        row = dict(lookup[request_id])
        path = (original.parent / row["image_path"]).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise RuntimeError(f"Corrupt input image: {request_id}")
        row["image_path"] = str(path)
        selected.append(row)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in selected))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; assert torch.cuda.is_available(); "
            "a=torch.randn(256,256,device='cuda',dtype=torch.bfloat16); "
            "assert torch.isfinite(a@a).all(); print(torch.cuda.get_device_name(0))",
        ],
        check=True,
    )
    results = {}
    for mode in ("edge", "cloud", "distributed"):
        run_dir = args.output_root / mode
        command = [
            "adaptive-vlm",
            "benchmark",
            "--config",
            str(ROOT / "configs/experiments/cloud_vqa_v1.yaml"),
            "--mode",
            mode,
            "--manifest",
            str(manifest),
            "--run-dir",
            str(run_dir),
            "--artifact-manifest",
            str(ROOT / "configs/model/moondream2_artifact.yaml"),
            "--cache-dir",
            "/models",
            "--vision-device",
            "cuda:0",
            "--text-device",
            "cuda:0",
            "--offline",
            "--warmups",
            "0",
            "--repetitions",
            "1",
        ]
        with (args.output_root / f"{mode}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        raw = run_dir / "requests.jsonl"
        results[mode] = {
            "returncode": result.returncode,
            "rows": [json.loads(line) for line in raw.read_text().splitlines()]
            if raw.exists()
            else [],
        }
    accepted = True
    outputs = {}
    for mode, result in results.items():
        rows = result["rows"]
        if (
            result["returncode"]
            or len(rows) != 12
            or {row["request_id"] for row in rows} != set(selected_ids)
            or len({row["trial_id"] for row in rows}) != 12
            or any(row["status"] != "ok" for row in rows)
        ):
            accepted = False
        outputs[mode] = {
            row["request_id"]: (row.get("output_text"), row.get("output_token_ids"))
            for row in rows
        }
    # Predeclared output criterion is exact decoded text and token-ID equivalence.
    equivalent = outputs["edge"] == outputs["cloud"] == outputs["distributed"]
    report = {
        "schema_version": 1,
        "accepted": accepted and equivalent,
        "diagnostic_only": True,
        "declared_image_reference": image,
        "request_ids": selected_ids,
        "criterion": "exact text and token IDs",
        "outputs_equivalent": equivalent,
        "results": results,
        "limitations": [
            "Not a host/container timing comparison or overhead measurement",
            "Does not replace network-path or restart acceptance",
            "Cloud GPU BF16 must also be tested locally on the cloud node",
        ],
    }
    (args.output_root / "smoke-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if not report["accepted"]:
        raise SystemExit(
            "Container correctness smoke failed; inspect preserved evidence"
        )
    print("All 36 diagnostic attempts passed with equivalent placement outputs")


if __name__ == "__main__":
    main()
