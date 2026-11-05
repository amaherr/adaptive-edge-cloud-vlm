"""Build a clean committed application with frozen provenance; never deploy it."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", help="Local image tag; defaults to adaptive-vlm:<commit>"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/infra/image-build.json"
    )
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise RuntimeError("Commit changes before building an experimental image")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    created = subprocess.check_output(
        ["git", "show", "-s", "--format=%cI", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tag = args.tag or f"adaptive-vlm:{commit}"
    hashes = {
        name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for name, path in {
            "MODEL_CONTRACT_SHA": "configs/model/moondream2.yaml",
            "LOCK_SHA": "requirements/model-cu128.lock",
        }.items()
    }
    command = [
        "docker",
        "build",
        "--platform",
        "linux/amd64",
        "-f",
        "container/Dockerfile",
        "--tag",
        tag,
        "--build-arg",
        f"GIT_SHA={commit}",
        "--build-arg",
        f"BUILD_CREATED={created}",
    ]
    for name, value in hashes.items():
        command.extend(["--build-arg", f"{name}={value}"])
    subprocess.run([*command, "."], cwd=ROOT, check=True)
    inspection = json.loads(
        subprocess.check_output(["docker", "image", "inspect", tag], text=True)
    )[0]
    report = {
        "git_commit": commit,
        "tag": tag,
        "local_image_id": inspection["Id"],
        "local_descriptor": inspection.get("Descriptor"),
        "labels": inspection["Config"]["Labels"],
        "note": (
            "Docker image-ID semantics vary by image store; "
            "deploy by the verified registry manifest reference"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Built {tag}; provenance saved to {args.output}")


if __name__ == "__main__":
    main()
