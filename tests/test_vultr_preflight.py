"""Offline tests for catalog completeness, safety and conservative gate reporting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

SPEC = importlib.util.spec_from_file_location(
    "vultr_preflight", Path(__file__).parents[1] / "scripts/vultr/check_plans.py"
)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_pages_preserves_filters_and_follows_cursor(monkeypatch):
    client = preflight.Client("secret")
    calls = []

    def get(path, params):
        calls.append((path, dict(params)))
        if "cursor" not in params:
            return {
                "plans": [{"id": "first"}],
                "meta": {
                    "links": {"next": "https://api.vultr.com/v2/plans?cursor=second"}
                },
            }
        return {"plans": [{"id": "last"}]}

    monkeypatch.setattr(client, "get", get)
    assert client.pages("/plans", "plans", {"type": "vcg"}) == [
        {"id": "first"},
        {"id": "last"},
    ]
    assert calls[1][1] == {"per_page": 500, "type": "vcg", "cursor": "second"}


@pytest.mark.parametrize(
    "link",
    [
        "https://evil.example/v2/plans?cursor=secret",
        "https://api.vultr.com/v2/account?cursor=a",
        "?cursor=a",  # Same cursor is returned repeatedly.
        "?page=2",
    ],
)
def test_unsafe_or_looping_pagination_fails(monkeypatch, link):
    client = preflight.Client("secret")
    monkeypatch.setattr(
        client, "get", lambda *_: {"plans": [], "meta": {"links": {"next": link}}}
    )
    with pytest.raises(preflight.PreflightError):
        client.pages("/plans", "plans")


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_http_failure_does_not_expose_credentials(monkeypatch, status):
    client = preflight.Client("sensitive-token")

    def fail(*args, **kwargs):
        raise HTTPError(
            "https://api.vultr.com", status, "sensitive-token provider body", {}, None
        )

    monkeypatch.setattr(client.opener, "open", fail)
    with pytest.raises(preflight.PreflightError) as error:
        client.get("/account")
    assert f"HTTP {status}" in str(error.value)
    assert "sensitive-token" not in str(error.value)


def test_redirect_never_forwards_token():
    with pytest.raises(preflight.PreflightError, match="redirect refused"):
        preflight.NoRedirect().redirect_request(
            None, None, 302, "", {}, "https://evil.example"
        )


def test_join_keeps_unknown_prices_and_no_account_details():
    class Fake:
        def get(self, path):
            assert path == "/account"
            return {"account": {"email": "private@example.com"}}

        def pages(self, path, key, params=None):
            return {
                "/plans": [{"id": "vcg-test", "type": "vcg", "monthly_cost": 100}],
                "/regions": [{"id": "test"}, {"id": "empty"}],
                "/regions/test/availability": ["vcg-test"],
                "/regions/empty/availability": [],
            }[path]

    report = preflight.collect(Fake(), [])
    assert report["gpu_plans"][0]["available_regions"] == ["test"]
    assert report["gpu_plans"][0]["hourly_cost"] is None
    assert report["gpu_plans"][0]["gpu_vram"] is None
    assert "private@example.com" not in json.dumps(report)
    with pytest.raises(preflight.PreflightError, match="absent"):
        preflight.collect(Fake(), ["unknown"])


def test_missing_token_writes_blocked_report_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("VULTR_API_KEY", raising=False)
    output = tmp_path / "report.json"
    assert preflight.main(["--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["gate"] == "BLOCKED"
    assert report["launch_authorized"] is False
    with pytest.raises(FileExistsError):
        preflight.main(["--output", str(output)])


def test_successful_catalog_still_does_not_approve_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("VULTR_API_KEY", "secret")
    monkeypatch.setattr(preflight, "collect", lambda *_: {"gpu_plans": []})
    output = tmp_path / "report.json"
    assert preflight.main(["--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["discovery_status"] == "COLLECTED"
    assert report["gate"] == "BLOCKED"
    assert report["launch_authorized"] is False
