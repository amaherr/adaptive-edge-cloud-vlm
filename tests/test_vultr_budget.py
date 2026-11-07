"""Budget decisions count all registered GPU lifetimes and fail closed."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from adaptive_vlm.ops.budget import evaluate

NOW = datetime(2026, 11, 7, 14, tzinfo=UTC)


def session():
    return {
        "approved": True,
        "campaign_tag": "adaptive-vlm-vultr-test",
        "region": "atl",
        "expires_at": "2026-11-07T15:00:00Z",
        "idle_timeout_seconds": 600,
        "session_budget_usd": 5,
        "spent_before_usd": 0,
        "ancillary_reserve_usd": 7,
        "total_budget_usd": 80,
        "teardown_at_usd": 72,
        "nodes": [
            {
                "id": "edge-id",
                "plan_id": "verified-plan",
                "hourly_usd": "0.471",
                "created_at": "2026-11-07T13:59:00Z",
            }
        ],
    }


def node():
    return {
        "id": "edge-id",
        "plan": "verified-plan",
        "region": "atl",
        "tags": ["adaptive-vlm-vultr-test"],
        "power_status": "running",
    }


def check(config=None, inventory=None, heartbeats=None):
    return evaluate(
        config or session(),
        [node()] if inventory is None else inventory,
        {"edge-id": NOW.isoformat()} if heartbeats is None else heartbeats,
        now=NOW,
    )


def test_minimum_hour_stopped_idle_and_expiry():
    assert check()["estimated_compute_usd"] == "0.471"
    assert not check()["destroy_required"]
    stopped = node() | {"power_status": "stopped"}
    assert check(inventory=[stopped])["destroy_required"]
    assert check(heartbeats={"edge-id": "2026-11-07T13:40:00Z"})["destroy_required"]
    expired = session() | {"expires_at": NOW.isoformat()}
    assert check(expired)["destroy_required"]


def test_destroyed_nodes_still_cost_and_absence_does_not_stop_accrual():
    config = session()
    config["nodes"][0].update(
        created_at="2026-11-07T10:00:00Z", destroyed_at="2026-11-07T11:01:00Z"
    )
    assert check(config, inventory=[])["estimated_compute_usd"] == "0.942"
    del config["nodes"][0]["destroyed_at"]
    assert check(config, inventory=[])["estimated_compute_usd"] == "1.884"


def test_campaign_and_phase_ceilings_trigger_shutdown():
    config = session() | {"spent_before_usd": "64.6"}
    result = check(config)
    assert result["destroy_required"]
    assert result["alert_thresholds_usd"] == [40, 60, 72]
    assert check(session() | {"session_budget_usd": "0.4"})["destroy_required"]
    assert check(session() | {"spent_before_usd": 80})["alert_thresholds_usd"] == [
        40,
        60,
        72,
        80,
    ]


def test_unrelated_nodes_are_ignored_and_unknown_managed_nodes_are_refused():
    unrelated = node() | {"id": "other", "tags": ["aws-work"]}
    assert not check(inventory=[node(), unrelated])["destroy_required"]
    with pytest.raises(ValueError, match="unregistered"):
        check(inventory=[node() | {"id": "unregistered"}])


def test_hardware_or_region_drift_stops_approved_session():
    assert check(inventory=[node() | {"plan": "expensive-other-plan"}])[
        "destroy_required"
    ]
    assert check(inventory=[node() | {"region": "other"}])["destroy_required"]
    assert check(heartbeats={"edge-id": "2026-11-07T14:05:00Z"})["destroy_required"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("approved", False),
        ("campaign_tag", "unrelated"),
        ("expires_at", "2026-11-07T15:00:00"),
        ("session_budget_usd", -1),
        ("spent_before_usd", -1),
        ("ancillary_reserve_usd", "NaN"),
        ("total_budget_usd", 81),
        ("teardown_at_usd", 100),
        ("idle_timeout_seconds", 1800),
    ],
)
def test_invalid_controls_fail_closed(field, value):
    with pytest.raises(ValueError):
        check(session() | {field: value})


def test_invalid_rates_lifetimes_and_duplicate_ids():
    for update in [
        {"hourly_usd": 0},
        {"hourly_usd": "NaN"},
        {"created_at": "2026-11-08T12:00:00Z"},
        {"destroyed_at": "2026-11-06T12:00:00Z"},
    ]:
        config = deepcopy(session())
        config["nodes"][0].update(update)
        with pytest.raises(ValueError):
            check(config)
    config = session()
    config["nodes"] *= 2
    with pytest.raises(ValueError, match="unique"):
        check(config)
    with pytest.raises(ValueError, match="offset"):
        evaluate(session(), [node()], {}, now=NOW.replace(tzinfo=None))


def test_preapproved_launch_registration_survives_operator_disconnect():
    from adaptive_vlm.ops.budget import register_instances

    pending = session() | {
        "registration_pending": True,
        "nodes": [],
        "approved_at": "2026-11-07T13:00:00Z",
        "approved_quotes": {
            "edge": {"plan_id": "verified-plan", "hourly_usd": "0.471"}
        },
    }
    assert not check(pending, inventory=[])["destroy_required"]
    created = node() | {
        "tags": ["adaptive-vlm-vultr-test", "role-edge"],
        "date_created": "2026-11-07T13:59:00Z",
    }
    registered = register_instances(pending, [created])
    assert registered["nodes"][0]["id"] == "edge-id"
    assert not registered["registration_pending"]
    assert not check(registered, inventory=[created])["destroy_required"]
    assert pending["nodes"] == []  # Caller state is not mutated before its CAS.
    for changed in [
        {"plan": "wrong-plan"},
        {"region": "other"},
        {"date_created": "2026-11-06T13:59:00Z"},
        {"tags": ["adaptive-vlm-vultr-test"]},
    ]:
        with pytest.raises(ValueError):
            register_instances(pending, [created | changed])
