"""Push a verified build using temporary ECR authentication, never desktop helpers."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--profile", default="cloud-infrastructure")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--build-report", type=Path, default=Path("artifacts/infra/image-build.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/infra/image-deployment.json")
    )
    args = parser.parse_args()
    if not re.fullmatch(
        r"\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9/_-]+", args.repository
    ):
        raise ValueError("Expected a private ECR repository URL")
    build = json.loads(args.build_report.read_text())
    aws = ["aws", "--profile", args.profile, "--region", args.region]
    repository_name = args.repository.split("/", 1)[1]
    subprocess.run(
        [*aws, "ecr", "describe-repositories", "--repository-names", repository_name],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    current = json.loads(
        subprocess.check_output(["docker", "image", "inspect", build["tag"]], text=True)
    )[0]
    # Earlier reports used a misleading config-digest key for Docker's opaque ID.
    expected_id = build.get("local_image_id") or build.get("local_image_config_digest")
    if current["Id"] != expected_id:
        raise RuntimeError("Local image differs from the verified build report")
    tag = args.repository + ":" + build["git_commit"]
    # Bypass broken desktop credential helpers without changing user settings.
    # TemporaryDirectory is mode 0700 and removes the short-lived token on exit.
    with tempfile.TemporaryDirectory(prefix="cloud-ecr-auth-") as directory:
        environment = {**os.environ, "DOCKER_CONFIG": directory}
        password = subprocess.check_output(
            [*aws, "ecr", "get-login-password"], text=True
        )
        subprocess.run(
            [
                "docker",
                "login",
                "--username",
                "AWS",
                "--password-stdin",
                args.repository.split("/")[0],
            ],
            input=password,
            text=True,
            env=environment,
            check=True,
            capture_output=True,
        )
        del password
        subprocess.run(
            ["docker", "tag", build["tag"], tag], env=environment, check=True
        )
        subprocess.run(["docker", "push", tag], env=environment, check=True)
    response = json.loads(
        subprocess.check_output(
            [
                *aws,
                "ecr",
                "describe-images",
                "--repository-name",
                repository_name,
                "--image-ids",
                "imageTag=" + build["git_commit"],
                "--output",
                "json",
            ],
            text=True,
        )
    )["imageDetails"][0]
    report = {
        "image_reference": args.repository + "@" + response["imageDigest"],
        "image_manifest_digest": response["imageDigest"],
        "git_commit": build["git_commit"],
        "registry_size_bytes": response["imageSizeInBytes"],
        "build": build,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Published {report['image_reference']}; metadata saved to {args.output}")


if __name__ == "__main__":
    main()
