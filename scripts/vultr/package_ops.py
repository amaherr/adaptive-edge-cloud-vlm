"""Build a small control-plane ZIP from an explicit nonsecret source allowlist."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    files = {
        "watchdog.py": "infra/runtime/vultr/watchdog.py",
        "teardown.py": "infra/runtime/vultr/teardown.py",
        "budget.py": "src/adaptive_vlm/ops/budget.py",
    }
    terraform = "infra/terraform/vultr/environments/gpu"
    for path in (root / terraform).glob("*.tf"):
        files[f"{terraform}/{path.name}"] = str(path.relative_to(root))
    files[f"{terraform}/.terraform.lock.hcl"] = f"{terraform}/.terraform.lock.hcl"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            archive.writestr(name, (root / source).read_bytes())
    print(f"Nonsecret watchdog/teardown package: {args.output}")


if __name__ == "__main__":
    main()
