"""Detect cache corruption before executing remote model code."""

import hashlib
import runpy
from pathlib import Path

import pytest
import yaml

VERIFY = runpy.run_path(str(Path(__file__).parents[1] / "container/verify_cache.py"))[
    "verify"
]


@pytest.mark.parametrize(
    "corrupt", [None, "model.safetensors", "model.py", "tokenizer.json"]
)
def test_frozen_cache_checks_every_kind_of_artifact(tmp_path, corrupt):
    hub = tmp_path / "hub"
    model = hub / "models--test--model/snapshots/revision"
    tokenizer = hub / "models--test--tokenizer/snapshots/token-revision"
    model.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    paths = {
        "model.safetensors": model / "model.safetensors",
        "model.py": model / "model.py",
        "tokenizer.json": tokenizer / "tokenizer.json",
    }
    hashes = {}
    for name, path in paths.items():
        path.write_bytes(name.encode())
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "repository": "test/model",
        "revision": "revision",
        "weights": {
            "filename": "model.safetensors",
            "sha256": hashes["model.safetensors"],
        },
        "tokenizer_dependency": {
            "repository": "test/tokenizer",
            "revision": "token-revision",
            "filename": "tokenizer.json",
            "sha256": hashes["tokenizer.json"],
        },
        "remote_code_sha256": {"model.py": hashes["model.py"]},
    }
    artifact = tmp_path / "artifact.yaml"
    artifact.write_text(yaml.safe_dump(manifest))
    if corrupt:
        paths[corrupt].write_bytes(b"changed")
        with pytest.raises(ValueError, match="Frozen artifact hash mismatch"):
            VERIFY(artifact, hub)
    else:
        assert VERIFY(artifact, hub)["verified_files"] == 3
