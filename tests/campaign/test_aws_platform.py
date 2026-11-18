"""AWS platform pieces: checkpoints, heartbeats, image identity, settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adaptive_vlm.campaign.checkpoint import (
    RECEIPT,
    Heartbeats,
    S3Checkpoint,
    write_receipt,
)
from adaptive_vlm.campaign.remote import RemoteCommandError
from adaptive_vlm.campaign.supervisor import PLATFORMS, load_settings

ROOT = Path(__file__).resolve().parents[2]
DIGEST = "1234.dkr.ecr.us-east-1.amazonaws.com/adaptive-vlm@sha256:" + "a" * 64


class Recorder:
    def __init__(self, fail: bool = False) -> None:
        self.commands: list[list[str]] = []
        self.fail = fail

    def __call__(self, command: Any) -> str:
        self.commands.append(list(command))
        if self.fail:
            raise RemoteCommandError("s3 unreachable")
        return ""


def _subblock(tmp_path: Path) -> Path:
    directory = tmp_path / "campaign" / "subblocks" / "T01-L0-r0"
    (directory / "run" / "edge").mkdir(parents=True)
    (directory / "validation_report.json").write_text('{"valid": true}')
    (directory / "run" / "edge" / "requests.jsonl").write_text("{}\n")
    (directory / ".run.writer.lock").write_text("")
    (tmp_path / "campaign" / "status.json").write_text("{}")
    return tmp_path / "campaign"


def test_receipt_hashes_every_file_but_locks_and_itself(tmp_path: Path) -> None:
    output = _subblock(tmp_path)
    receipt = write_receipt(output / "subblocks" / "T01-L0-r0")
    assert sorted(receipt["files"]) == [
        "run/edge/requests.jsonl",
        "validation_report.json",
    ]
    again = write_receipt(output / "subblocks" / "T01-L0-r0")
    assert RECEIPT not in again["files"]


def test_checkpoint_syncs_subblock_then_campaign_files(tmp_path: Path) -> None:
    output = _subblock(tmp_path)
    runner = Recorder()
    checkpoint = S3Checkpoint(
        "s3://bucket/research/g2v2plus/", region="us-east-1", runner=runner
    )
    assert checkpoint.push_subblock(output, "T01-L0-r0") is None
    first, second = runner.commands
    assert first[:5] == [
        "aws",
        "s3",
        "sync",
        str(output / "subblocks" / "T01-L0-r0"),
        "s3://bucket/research/g2v2plus/subblocks/T01-L0-r0",
    ]
    assert "--region" in first and "*.lock" in first
    assert second[4] == "s3://bucket/research/g2v2plus/"
    assert "subblocks/*" in second and "preflights/*" in second
    assert (output / "subblocks" / "T01-L0-r0" / RECEIPT).exists()
    with pytest.raises(ValueError, match="s3://"):
        S3Checkpoint("bucket/path")


def test_checkpoint_failure_is_reported_not_raised(tmp_path: Path) -> None:
    output = _subblock(tmp_path)
    error = S3Checkpoint("s3://b/p", runner=Recorder(fail=True)).push_subblock(
        output, "T01-L0-r0"
    )
    assert error is not None and "s3 unreachable" in error


def test_heartbeats_touch_edge_every_time_and_cloud_on_interval(
    tmp_path: Path,
) -> None:
    now = [0.0]
    cloud = Recorder()
    beats = Heartbeats(
        edge_path=str(tmp_path / "hb" / "job.heartbeat"),
        cloud_path="/srv/adaptive-vlm/artifacts/job.heartbeat",
        cloud_runner=cloud,
        cloud_every_s=300,
        clock=lambda: now[0],
    )
    assert beats.beat() is None
    assert (tmp_path / "hb" / "job.heartbeat").exists()
    now[0] = 100
    beats.beat()
    now[0] = 301
    beats.beat()
    assert cloud.commands == [
        ["touch", "/srv/adaptive-vlm/artifacts/job.heartbeat"],
        ["touch", "/srv/adaptive-vlm/artifacts/job.heartbeat"],
    ]
    failing = Heartbeats(
        edge_path=None, cloud_path="/x", cloud_runner=Recorder(fail=True)
    )
    assert "cloud heartbeat failed" in str(failing.beat())


def test_image_build_record_replaces_git_inside_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import adaptive_vlm.environment as environment

    record = tmp_path / "adaptive-vlm-build.json"
    record.write_text(
        json.dumps({"git_commit": "b" * 40, "base_image": "python@sha256:x"})
    )
    monkeypatch.setattr(environment, "BUILD_RECORD", record)
    monkeypatch.setenv("APP_IMAGE", DIGEST)
    workdir = tmp_path / "app"
    workdir.mkdir()
    state = environment._git_state(workdir)
    assert state == {
        "available": True,
        "commit": "b" * 40,
        "dirty": False,
        "source": "image",
        "image": DIGEST,
    }
    (workdir / ".git").mkdir()  # A real checkout keeps using git.
    assert environment._git_state(workdir).get("source") != "image"


def test_supervisor_image_identity_requires_a_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import adaptive_vlm.campaign.supervisor as supervisor
    import adaptive_vlm.environment as environment

    record = tmp_path / "build.json"
    record.write_text(json.dumps({"git_commit": "c" * 40}))
    monkeypatch.setattr(environment, "BUILD_RECORD", record)
    fake = type("Fake", (), {"settings": {"platform": "aws-linux"}})()
    monkeypatch.setenv("APP_IMAGE", DIGEST)
    assert supervisor.CampaignSupervisor._source_identity(fake) == ("c" * 40, "")
    monkeypatch.setenv("APP_IMAGE", "adaptive-vlm:latest")
    with pytest.raises(RuntimeError, match="digest"):
        supervisor.CampaignSupervisor._source_identity(fake)
    monkeypatch.setattr(environment, "BUILD_RECORD", tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="campaign image"):
        supervisor.CampaignSupervisor._source_identity(fake)


def test_aws_settings_template_validation(tmp_path: Path) -> None:
    template = json.loads(
        (ROOT / "configs/campaigns/g2v2_aws.example.json").read_text()
    )
    assert PLATFORMS[template["platform"]] == ("edge_gpu_memory_used_mib", "image")
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(template))
    with pytest.raises(ValueError, match=r"cloud_access"):
        load_settings(path, ROOT)
    pilot = json.loads(
        (ROOT / "configs/campaigns/g2v2_aws_pilot.example.json").read_text()
    )
    path.write_text(
        json.dumps(
            {
                **pilot,
                "cloud_access": {**pilot["cloud_access"], "instance_id": "i-0abc"},
            }
        )
    )
    assert load_settings(path, ROOT)["allow_unfrozen_protocol"] is True
    path.write_text(json.dumps(template))
    filled = {
        **template,
        "cloud_interface": "ens5",
        "edge_residency": "swap",
        "edge_dedicated_limit_mib": 5600,
        "cloud_access": {"kind": "ssm", "instance_id": "i-0abc", "region": "us-east-1"},
    }
    path.write_text(json.dumps(filled))
    settings = load_settings(path, ROOT)
    assert settings["edge_dedicated_limit_mib"] == 5600.0
    assert settings["tc_privilege"] == [] and settings["shaping"] == "classified"
    path.write_text(json.dumps({**filled, "platform": "mac-os"}))
    with pytest.raises(ValueError, match="platform"):
        load_settings(path, ROOT)
    path.write_text(json.dumps({**filled, "cloud_container": ""}))
    with pytest.raises(ValueError, match="cloud_container"):
        load_settings(path, ROOT)


def test_acceptance_drill_needs_no_references_and_is_never_data(
    tmp_path: Path,
) -> None:
    from adaptive_vlm.prediction.v2.dataset.campaign_v2 import _campaign_grid

    template = json.loads(
        (ROOT / "configs/campaigns/aws_drill.example.json").read_text()
    )
    template["cloud_access"]["instance_id"] = "i-0abc"
    path = tmp_path / "drill.json"
    path.write_text(json.dumps(template))
    settings = load_settings(path, ROOT)
    assert settings["acceptance_drill"] is True
    assert "same_device_reference_evidence" not in template
    path.write_text(json.dumps({**template, "acceptance_drill": False}))
    with pytest.raises(ValueError, match="same_device_reference_evidence"):
        load_settings(path, ROOT)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign_plan.json").write_text(
        json.dumps({"acceptance_drill": True, "settings": settings})
    )
    with pytest.raises(ValueError, match="not campaign data"):
        _campaign_grid(campaign)
