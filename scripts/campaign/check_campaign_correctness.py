"""Row-level correctness of every sub-block of a G2 v2 campaign (outputs only).

Compares every measured row with the same-device reference bundles, per
sub-block (``evaluate_rows`` with ``scope="measured"``). It reads only outputs,
statuses and scores, never latency, so it opens no sealed split (amendment A3).

The reference bundles are verified against the current source tree, which must
match the campaign's frozen commit apart from the two hash-exempt analysis
modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from adaptive_vlm.benchmark.heterogeneous_correctness import (
    evaluate_rows,
    verify_reference_evidence,
)
from adaptive_vlm.data.manifest import load_frozen_manifest

KEEP = (
    "passed",
    "scoped_request_count",
    "full_repeated_successful_request_coverage",
    "within_placement_repeatable",
    "task_quality_equal",
    "quality_disagreement_request_ids",
    "reference_output_mismatches",
    "cross_gpu_output_disagreement_request_ids",
    "measured_status_counts",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Local copy of the edge /artifacts volume (reference bundles)",
    )
    parser.add_argument("--short-manifest", type=Path, required=True)
    parser.add_argument("--long-manifest", type=Path, required=True)
    parser.add_argument("--edge-gpu", required=True)
    parser.add_argument("--cloud-gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    manifests = {
        "short": load_frozen_manifest(args.short_manifest),
        "long": load_frozen_manifest(args.long_manifest),
    }
    expected = {}
    for task, path in contract["same_device_reference_evidence"].items():
        local = args.artifact_root / Path(path).relative_to("/artifacts")
        proof = verify_reference_evidence(
            local,
            hashlib.sha256(local.read_bytes()).hexdigest(),
            manifests[task],
            contract["model_contract_sha256"],
            expected_devices={"edge": args.edge_gpu, "cloud": args.cloud_gpu},
        )
        expected[task] = proof["_expected_outputs"]

    report, tally = {}, Counter()
    for subblock in sorted((args.campaign / "subblocks").iterdir()):
        task = json.loads((subblock / "plan.json").read_text())["task"]
        runs = {}
        for run in sorted(p for p in (subblock / "run").iterdir() if p.is_dir()):
            placement = json.loads((run / "plan.json").read_text())["placement"]
            runs[placement] = [
                json.loads(line)
                for line in (run / "requests.jsonl").read_text().splitlines()
            ]
        result = evaluate_rows(
            runs, manifests[task], expected_outputs=expected[task], scope="measured"
        )
        report[subblock.name] = {key: result[key] for key in KEEP}
        tally["passed" if result["passed"] else "failed"] += 1
        tally["reference_mismatch_rows"] += sum(
            len(v) for v in result["reference_output_mismatches"].values()
        )
        tally["cross_gpu_disagreement_ids"] += len(
            result["cross_gpu_output_disagreement_request_ids"]
        )
        tally["quality_disagreement_ids"] += len(
            result["quality_disagreement_request_ids"]
        )
        tally["scoped_ids"] += result["scoped_request_count"]
    payload = {
        "schema_version": 1,
        "scope": "measured",
        "subblocks": len(report),
        "tally": dict(tally),
        "per_subblock": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(tally)))
    print("failed:", [k for k, v in report.items() if not v["passed"]])


if __name__ == "__main__":
    main()
