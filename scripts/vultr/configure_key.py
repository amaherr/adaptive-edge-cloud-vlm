"""Store the exported Vultr key encrypted in SSM without logging it or using argv."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="cloud-infrastructure")
    args = parser.parse_args()
    key = os.environ.get("VULTR_API_KEY", "").strip()
    if not key:
        parser.error("VULTR_API_KEY must be exported in this terminal")
    with tempfile.TemporaryDirectory(prefix="vlm-secret-") as directory:
        path = Path(directory) / "input.json"
        path.touch(mode=0o600)
        path.write_text(
            json.dumps(
                {
                    "Name": "/adaptive-vlm/vultr/api-key",
                    "Type": "SecureString",
                    "Value": key,
                    "Overwrite": True,
                }
            )
        )
        result = subprocess.run(
            [
                "aws",
                "--profile",
                args.profile,
                "--region",
                "us-east-1",
                "ssm",
                "put-parameter",
                "--cli-input-json",
                f"file://{path}",
                "--no-cli-pager",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise SystemExit("Secret setup failed; check AWS login and SSM permissions")
    print("Encrypted Vultr key stored at /adaptive-vlm/vultr/api-key; value not logged")


if __name__ == "__main__":
    main()
