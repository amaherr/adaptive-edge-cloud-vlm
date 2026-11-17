"""Verify model, tokenizer and executable cache artifacts before startup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def verify(artifact: Path, hub: Path) -> dict:
    manifest = yaml.safe_load(artifact.read_text())
    model = (
        hub
        / ("models--" + manifest["repository"].replace("/", "--"))
        / "snapshots"
        / manifest["revision"]
    )
    tokenizer = manifest["tokenizer_dependency"]
    token_root = (
        hub
        / ("models--" + tokenizer["repository"].replace("/", "--"))
        / "snapshots"
        / tokenizer["revision"]
    )
    targets = [
        (model / manifest["weights"]["filename"], manifest["weights"]["sha256"]),
        (token_root / tokenizer["filename"], tokenizer["sha256"]),
    ]
    targets.extend(
        (model / name, expected)
        for name, expected in manifest["remote_code_sha256"].items()
    )
    for path, expected in targets:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {path.name}")
    return {
        "schema_version": 1,
        "verified_files": len(targets),
        "weights_sha256": manifest["weights"]["sha256"],
        "tokenizer_sha256": tokenizer["sha256"],
        "model_revision": manifest["revision"],
        "tokenizer_revision": tokenizer["revision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/app/configs/model/moondream2_artifact.yaml"),
    )
    parser.add_argument("--hub", type=Path, default=Path("/models/hub"))
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact, args.hub), indent=2))


if __name__ == "__main__":
    main()
