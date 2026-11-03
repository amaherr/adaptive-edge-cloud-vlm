"""Fail build on provenance or installed dependency mismatch."""

import hashlib
import importlib.metadata
import re
import sys
from pathlib import Path

if sys.argv[1] == "build":
    base, git, model, lock = sys.argv[2:]
    assert re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", base), (
        "Digest-pinned base required"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", git), "Commit SHA required"
    for path, expected in [
        ("configs/model/moondream2.yaml", model),
        ("requirements/model-cu128.lock", lock),
    ]:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
else:
    for line in Path("requirements/model-cu128.lock").read_text().splitlines():
        if "==" in line:
            name, version = line.split("==")
            assert importlib.metadata.version(name) == version, name
