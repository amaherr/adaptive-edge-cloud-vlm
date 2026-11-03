"""Validate the immutable image and persistent host mounts before Compose startup."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def validate_environment(environment: dict[str, str]) -> dict[str, str]:
    image = environment.get("APP_IMAGE", "")
    if not re.fullmatch(
        r"\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[^\s@]+@sha256:[0-9a-f]{64}",
        image,
    ):
        raise ValueError("APP_IMAGE must be an immutable ECR manifest reference")
    for name in ("MODEL_DIR", "DATA_DIR", "ARTIFACT_DIR"):
        path = Path(environment.get(name, ""))
        if not path.is_absolute() or not path.is_dir():
            raise ValueError(f"{name} must be an existing absolute directory")
        if not path.resolve().is_relative_to(Path("/srv/adaptive-vlm")):
            raise ValueError(f"{name} must be on the managed persistent data volume")
    return {
        name: environment[name]
        for name in ("APP_IMAGE", "MODEL_DIR", "DATA_DIR", "ARTIFACT_DIR")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("edge", "cloud"), required=True)
    args = parser.parse_args()
    checked = validate_environment(dict(os.environ))
    subprocess.run(["mountpoint", "-q", "/srv/adaptive-vlm"], check=True)
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            f"container/compose.{args.role}.yaml",
            "config",
            "--quiet",
        ],
        check=True,
    )
    image = checked["APP_IMAGE"]
    metadata = json.loads(
        subprocess.check_output(["docker", "image", "inspect", image], text=True)
    )[0]
    if image not in metadata.get("RepoDigests", []):
        raise RuntimeError(
            "Pulled image does not report the requested repository digest"
        )
    labels = metadata["Config"]["Labels"]
    for name, length in (
        ("org.opencontainers.image.revision", 40),
        ("research.model-contract-sha256", 64),
        ("research.lock-sha256", 64),
    ):
        if not re.fullmatch(r"[0-9a-f]{" + str(length) + "}", labels.get(name, "")):
            raise RuntimeError(f"Missing image provenance label: {name}")
    print(
        json.dumps(
            {
                "validated": checked,
                "image_config_digest": metadata["Id"],
                "labels": labels,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
