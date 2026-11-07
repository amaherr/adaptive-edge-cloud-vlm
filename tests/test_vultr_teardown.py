"""The teardown worker must never apply create/update or unrelated deletions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def teardown(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "watchdog", SimpleNamespace(instances=lambda _: [])
    )
    spec = importlib.util.spec_from_file_location(
        "vultr_teardown", Path(__file__).parents[1] / "infra/runtime/vultr/teardown.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session():
    return {
        "nodes": [{"id": "approved-node"}],
        "campaign_tag": "adaptive-vlm-vultr-test",
    }


def change(actions=None, identity="approved-node", kind="vultr_instance", tags=None):
    return {
        "type": kind,
        "name": "gpu",
        "change": {
            "actions": actions or ["delete"],
            "before": {
                "id": identity,
                "tags": ["adaptive-vlm-vultr-test"] if tags is None else tags,
            },
        },
    }


def test_only_exact_registered_deletion_is_accepted(teardown):
    teardown.validate_plan({"resource_changes": [change()]}, session())
    for resource in [
        change(["create"]),
        change(["update"]),
        change(["delete", "create"]),
        change(identity="unrelated"),
        change(kind="aws_s3_bucket"),
        change(tags=["other-campaign"]),
        change() | {"module_address": "module.aws"},
    ]:
        with pytest.raises(ValueError):
            teardown.validate_plan({"resource_changes": [resource]}, session())


def test_empty_or_read_only_plan_cannot_hide_registered_live_node(teardown):
    for resources in [[], [change(["no-op"])], [change(["read"])]]:
        with pytest.raises(ValueError, match="every registered"):
            teardown.validate_plan({"resource_changes": resources}, session())
