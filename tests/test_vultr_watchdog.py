"""Off-VM orchestration: alerts cannot disable STOP and bounded teardown."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from adaptive_vlm.ops.budget import evaluate, register_instances

NOW = datetime(2026, 11, 7, 14, tzinfo=UTC)
PREFIX = "research/adaptive-vlm/vultr-v1"


class ClientError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class S3:
    def __init__(self):
        self.objects = {}

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError("NoSuchKey")
        data, when, etag = self.objects[Key]
        return {"Body": io.BytesIO(data), "LastModified": when, "ETag": etag}

    def put_object(
        self, *, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **kwargs
    ):
        old = self.objects.get(Key)
        if (IfNoneMatch and old) or (IfMatch and (not old or old[2] != IfMatch)):
            raise ClientError("PreconditionFailed")
        self.objects[Key] = (Body, NOW, str(len(self.objects)))
        return {"ETag": str(len(self.objects))}


def make_session():
    return {
        "session_id": "preflight-1",
        "approved": True,
        "campaign_tag": "adaptive-vlm-vultr-test",
        "region": "atl",
        "checkpoint_prefixes": [PREFIX + "/runs/preflight"],
        "expires_at": "2026-11-07T13:30:00Z",
        "idle_timeout_seconds": 600,
        "session_budget_usd": 5,
        "spent_before_usd": 65,
        "ancillary_reserve_usd": 7,
        "total_budget_usd": 80,
        "teardown_at_usd": 72,
        "nodes": [
            {
                "id": "node-1",
                "plan_id": "verified-plan",
                "hourly_usd": "0.471",
                "created_at": "2026-11-07T13:20:00Z",
                "activity_id": "job",
            }
        ],
    }


@pytest.fixture
def runtime(monkeypatch):
    store, alerts, builds = S3(), [], []

    def publish(**kwargs):
        alerts.append(kwargs)
        return {"MessageId": "message-1"}

    def start_build(**kwargs):
        builds.append(kwargs)
        return {"build": {"id": "build-1"}}

    sns = SimpleNamespace(publish=publish)
    services = {
        "s3": store,
        "sns": sns,
        "ssm": SimpleNamespace(
            get_parameter=lambda **_: {"Parameter": {"Value": "secret"}}
        ),
        "codebuild": SimpleNamespace(
            start_build=start_build,
            batch_get_builds=lambda **_: {"builds": [{"buildStatus": "IN_PROGRESS"}]},
        ),
    }
    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(client=services.__getitem__)
    )
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "botocore.exceptions", SimpleNamespace(ClientError=ClientError)
    )
    monkeypatch.setitem(
        sys.modules,
        "budget",
        SimpleNamespace(evaluate=evaluate, register_instances=register_instances),
    )
    spec = importlib.util.spec_from_file_location(
        "vultr_watchdog", Path(__file__).parents[1] / "infra/runtime/vultr/watchdog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(
        module,
        "instances",
        lambda _: [
            {
                "id": "node-1",
                "plan": "verified-plan",
                "region": "atl",
                "tags": ["adaptive-vlm-vultr-test"],
                "power_status": "running",
            }
        ],
    )
    for key, value in {
        "BUCKET": "bucket",
        "PREFIX": PREFIX,
        "TOPIC_ARN": "topic",
        "VULTR_KEY_PARAMETER": "/test/key",
        "TEARDOWN_PROJECT": "project",
    }.items():
        monkeypatch.setenv(key, value)
    return module, store, sns, alerts, builds


def register(store, session=None):
    store.put_object(
        Bucket="bucket",
        Key=PREFIX + "/control/session.json",
        Body=json.dumps(session or make_session()).encode(),
    )


def test_no_registered_session_never_starts_compute_or_teardown(runtime):
    module, _, _, alerts, builds = runtime
    assert module.handler({}, None)["state"] == "NO_REGISTERED_SESSION"
    assert not alerts and not builds


def test_stop_is_durable_before_grace_then_only_teardown_job_starts(runtime):
    module, store, _, alerts, builds = runtime
    register(store)
    assert module.handler({}, None)["state"] == "CHECKPOINT_GRACE"
    assert PREFIX + "/runs/preflight/STOP" in store.objects
    assert len(alerts) == 4  # $40/$60/$72 plus the stop alert.
    assert not builds
    key = PREFIX + "/control/stops/preflight-1.json"
    data, _, etag = store.objects[key]
    store.objects[key] = (data, NOW - timedelta(seconds=181), etag)
    store.objects.pop(PREFIX + "/control/watchdog-lease.json", None)
    assert module.handler({}, None)["state"] == "TERRAFORM_TEARDOWN_STARTED"
    assert len(builds) == 1
    store.objects.pop(PREFIX + "/control/watchdog-lease.json", None)
    module.handler({}, None)
    assert len(builds) == 1  # The same in-progress worker must not be duplicated.
    assert len(alerts) == 4  # Successful threshold deliveries are not repeated.


def test_failed_alert_delivery_does_not_disable_shutdown(runtime):
    module, store, sns, _, builds = runtime
    register(store)

    def broken(**kwargs):
        raise RuntimeError("SNS unavailable")

    sns.publish = broken
    assert module.handler({}, None)["state"] == "CHECKPOINT_GRACE"
    assert PREFIX + "/runs/preflight/STOP" in store.objects
    key = PREFIX + "/control/stops/preflight-1.json"
    data, _, etag = store.objects[key]
    store.objects[key] = (data, NOW - timedelta(seconds=181), etag)
    store.objects.pop(PREFIX + "/control/watchdog-lease.json", None)
    assert module.handler({}, None)["state"] == "TERRAFORM_TEARDOWN_STARTED"
    assert len(builds) == 1


def test_provider_failure_still_requests_bounded_teardown(runtime, monkeypatch):
    module, store, _, _, _ = runtime
    register(store)

    def broken(_):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(module, "instances", broken)
    report = module.handler({}, None)
    assert report["state"] == "CHECKPOINT_GRACE"
    assert report["destroy_required"]


def test_checkpoint_escape_is_rejected(runtime):
    module, store, _, _, builds = runtime
    register(store, make_session() | {"checkpoint_prefixes": ["aws/other-project"]})
    with pytest.raises(ValueError, match="unapproved"):
        module.handler({}, None)
    assert not builds


def test_execution_lease_rejects_overlapping_invocation(runtime):
    module, store, _, _, builds = runtime
    register(store)
    assert module.handler({}, None)["state"] == "CHECKPOINT_GRACE"
    assert module.handler({}, None)["state"] == "WATCHDOG_ALREADY_RUNNING"
    assert not builds
