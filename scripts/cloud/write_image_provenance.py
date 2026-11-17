"""Write configs/cloud/image_provenance.json from the build inputs (at each freeze)."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "configs/model/moondream2.yaml",
    "configs/model/moondream2_artifact.yaml",
    "requirements/model-cu128.lock",
)


def main() -> None:
    payload = {
        "schema_version": 2,
        "description": (
            "Inputs the container build verifies (container/verify_image.py). "
            "Regenerate at each freeze with scripts/cloud/write_image_provenance.py."
        ),
        "files": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in FILES
        },
    }
    target = ROOT / "configs/cloud/image_provenance.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", newline="\n")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
