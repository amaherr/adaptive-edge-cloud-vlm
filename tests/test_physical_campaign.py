"""Recovery, progress and supervision gates without university hardware."""

from __future__ import annotations

import json
import runpy
import subprocess
import venv
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_physical_campaign.py")
)
progress = MODULE["progress"]
validate_block = MODULE["validate_block"]
Supervisor = MODULE["Supervisor"]


def test_progress_counts_durable_failures_and_ignores_torn_tail(tmp_path: Path) -> None:
    raw = tmp_path / "N1" / "edge" / "requests.jsonl"
    raw.parent.mkdir(parents=True)
    rows = [
        {"warmup": True, "status": "ok"},
        {"warmup": False, "status": "ok"},
        {"warmup": False, "status": "oom"},
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"warmup":')
    result = progress(tmp_path)
    assert result["measured_attempts"] == 2
    assert result["warmup_attempts"] == 1
    assert result["status_counts"] == {"ok": 1, "oom": 1}
    assert result["last_journal_write_unix"] is not None


def test_completed_block_requires_gate_and_unchanged_raw_inputs(tmp_path: Path) -> None:
    summary = {
        "row_count": 903,
        "warmup_rows": 3,
        "controlled_network_gate_passed": True,
    }
    hashes = {}
    for name in (
        "network_pre.json",
        "network_post.json",
        "profile/profile.jsonl",
        "edge_gpu_samples.csv",
        "edge/requests.jsonl",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
        hashes[name] = MODULE["sha"](path)
    (tmp_path / "profile/summary.json").write_text(json.dumps(summary))
    report = {
        "profile_summary": summary,
        "network_pre_sha256": hashes["network_pre.json"],
        "network_post_sha256": hashes["network_post.json"],
        "profile_sha256": hashes["profile/profile.jsonl"],
        "edge_gpu_samples_sha256": hashes["edge_gpu_samples.csv"],
        "artifact_hashes": {"edge/requests.jsonl": hashes["edge/requests.jsonl"]},
    }
    (tmp_path / "validation_report.json").write_text(json.dumps(report))
    assert validate_block(tmp_path) == report
    (tmp_path / "edge/requests.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="changed completed block input"):
        validate_block(tmp_path)


def test_failure_during_second_peer_shaping_cleans_both_peers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = Supervisor({"output_root": str(tmp_path), "cloud_interface": "eth0"})
    supervisor.block = "N1"
    (tmp_path / "N1").mkdir()
    globals_ = Supervisor.shape.__globals__
    calls = []

    def remote(settings: dict, command: list[str]) -> str:
        calls.append(("remote", command))
        if "replace" in command:
            return ""
        return "qdisc mq 0: root"

    def local(command: list[str]) -> str:
        calls.append(("local", command))
        if "replace" in command:
            raise subprocess.CalledProcessError(1, command)
        return "qdisc mq 0: root"

    monkeypatch.setitem(globals_, "ssh", remote)
    monkeypatch.setitem(globals_, "checked", local)
    with pytest.raises(subprocess.CalledProcessError):
        supervisor.shape(
            {"rate_mbps": 2, "egress_delay_ms": 10}, {"edge_interface": "eth0"}
        )
    assert supervisor.shaped
    assert supervisor.cleanup({"edge_interface": "eth0"}) == []
    assert sum("show" in cmd for _, cmd in calls) >= 6
    assert not (tmp_path / "shaping_active.json").exists()


def test_supervisor_skips_only_validated_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"blocks": {}, "edge_interface": "eth0"}))
    supervisor = Supervisor({"output_root": str(tmp_path), "grid": str(grid)})
    for i in range(1, 10):
        path = tmp_path / f"N{i}" / "validation_report.json"
        path.parent.mkdir()
        path.write_text("{}")
    seen = []
    globals_ = Supervisor.execute.__globals__
    monkeypatch.setattr(supervisor, "preflight", lambda: {})
    monkeypatch.setattr(supervisor, "smoke", lambda: None)
    monkeypatch.setitem(globals_, "validate_block", lambda path: seen.append(path.name))
    monkeypatch.setitem(
        globals_,
        "progress",
        lambda path: {
            "measured_attempts": 8100,
            "percent": 100,
            "status_counts": {"ok": 8100},
        },
    )
    supervisor.execute(preflight_only=False)
    assert seen == [f"N{i}" for i in range(1, 10)]
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "completed"


def test_edge_smoke_runs_without_cloud_and_resumes_saved_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("edge smoke must not use cloud preflight or shaping")

    monkeypatch.setattr(Supervisor, "preflight", forbidden)
    monkeypatch.setattr(Supervisor, "shape", forbidden)
    smoke = MODULE["edge_smoke"]
    config = Path(__file__).resolve().parents[1] / (
        "configs/experiments/chunk6_fake_benchmark.yaml"
    )
    smoke(config, tmp_path, 3)
    raw = tmp_path / "run/requests.jsonl"
    before = raw.read_bytes()
    rows = [json.loads(line) for line in before.splitlines()]
    assert len(rows) == 4
    assert sum(not row["warmup"] for row in rows) == 3
    assert all(row["mode"] == "edge" and row["status"] == "ok" for row in rows)
    assert "[4/4] edge" in (tmp_path / "smoke.log").read_text()
    smoke(config, tmp_path, 3)
    assert raw.read_bytes() == before
    with pytest.raises(ValueError, match="subset changed"):
        smoke(config, tmp_path, 2)
    assert raw.read_bytes() == before
    with pytest.raises(ValueError, match="positive"):
        smoke(config, tmp_path, 0)


def test_settings_preserve_virtual_environment_interpreters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / ".venv"
    venv.EnvBuilder(symlinks=True).create(environment)
    load_settings = MODULE["load_settings"]
    monkeypatch.setitem(load_settings.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(load_settings.__globals__, "checked", lambda command: "")
    settings = {
        "cloud_ssh": "cloud",
        "cloud_repo": "/cloud/repo",
        "cloud_python": "/cloud/repo/.venv/bin/python",
        "cloud_interface": "eth0",
        "stall_seconds": 900,
        **{key: "unused" for key in (
            "grid", "split", "config", "smoke_config", "output_root",
            "artifact_manifest", "cache_dir",
        )},
        "python": ".venv/bin/python",
        "test_python": ".venv/bin/python",
        "adaptive_vlm": ".venv/bin/adaptive-vlm",
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    loaded = load_settings(path)
    for key in ("python", "test_python"):
        executable = loaded[key]
        assert executable == str(environment / "bin/python")
        prefix = subprocess.check_output(
            [executable, "-c", "import sys; print(sys.prefix)"], text=True
        ).strip()
        assert prefix == str(environment)
