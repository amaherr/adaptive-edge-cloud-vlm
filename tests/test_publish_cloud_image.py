"""Protect image identity and isolate temporary ECR credentials."""

import json
import runpy
import stat
import subprocess
import sys
from pathlib import Path

import pytest

PUBLISHER = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/publish_cloud_image.py")
)


def test_private_temporary_auth_is_removed(tmp_path, monkeypatch):
    build = {
        "tag": "local:test",
        "git_commit": "a" * 40,
        "local_image_config_digest": "sha256:config",
    }
    report = tmp_path / "build.json"
    report.write_text(json.dumps(build))
    output = tmp_path / "deployment.json"
    repository = "123456789012.dkr.ecr.us-east-1.amazonaws.com/project/test"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish",
            "--repository",
            repository,
            "--build-report",
            str(report),
            "--output",
            str(output),
        ],
    )
    directories = []

    def check_output(argv, **kwargs):
        if argv[:3] == ["docker", "image", "inspect"]:
            return json.dumps([{"Id": build["local_image_config_digest"]}])
        if "get-login-password" in argv:
            return "fixture-token"
        assert "describe-images" in argv
        return json.dumps(
            {
                "imageDetails": [
                    {"imageDigest": "sha256:" + "b" * 64, "imageSizeInBytes": 42}
                ]
            }
        )

    def run(argv, **kwargs):
        assert "fixture-token" not in argv
        if argv[0] == "docker":
            directory = Path(kwargs["env"]["DOCKER_CONFIG"])
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700
            directories.append(directory)
        if "login" in argv:
            assert kwargs["input"] == "fixture-token"
            assert "--password-stdin" in argv
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(subprocess, "run", run)
    PUBLISHER["main"]()
    assert directories and all(not path.exists() for path in directories)
    result = json.loads(output.read_text())
    assert result["image_reference"] == repository + "@sha256:" + "b" * 64
    assert "fixture-token" not in output.read_text()


def test_changed_local_image_is_rejected(tmp_path, monkeypatch):
    report = tmp_path / "build.json"
    report.write_text(
        json.dumps({"tag": "local:test", "local_image_config_digest": "expected"})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish",
            "--repository",
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/test",
            "--build-report",
            str(report),
        ],
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: json.dumps([{"Id": "changed"}]),
    )
    with pytest.raises(RuntimeError, match="differs from the verified build"):
        PUBLISHER["main"]()
