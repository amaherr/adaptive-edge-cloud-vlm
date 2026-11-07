"""Conservative GPU cost and idle/expiry decisions shared with the off-VM guard."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("control timestamps must include a UTC offset")
    return result.astimezone(UTC)


def evaluate(
    session: dict[str, Any],
    instances: list[dict[str, Any]],
    heartbeat_times: dict[str, str],
    *,
    now: datetime,
) -> dict[str, Any]:
    """Include stopped nodes, round each session up, and retain destroyed-node cost."""
    if now.tzinfo is None:
        raise ValueError("now must include a UTC offset")
    if session.get("approved") is not True:
        raise ValueError("session lacks owner approval")
    tag = session["campaign_tag"]
    if not isinstance(tag, str) or not tag.startswith("adaptive-vlm-vultr-"):
        raise ValueError("invalid dedicated campaign tag")
    nodes = session["nodes"]
    ids = [node["id"] for node in nodes]
    if (not nodes and not session.get("registration_pending")) or len(set(ids)) != len(
        ids
    ):
        raise ValueError("session needs unique registered instance IDs")
    live = [node for node in instances if tag in node.get("tags", [])]
    unknown = [node["id"] for node in live if node["id"] not in ids]
    if unknown:
        raise ValueError(
            "unregistered campaign nodes; manual inventory review required"
        )
    reasons = []
    if len(live) > 2:
        reasons.append("more than two campaign GPU instances")
    current = {node["id"]: node for node in live}
    compute = Decimal("0")
    for node in nodes:
        rate = Decimal(str(node["hourly_usd"]))
        if not rate.is_finite() or rate <= 0:
            raise ValueError("every registered node needs a positive hourly quote")
        start = timestamp(node["created_at"])
        end = timestamp(node["destroyed_at"]) if node.get("destroyed_at") else now
        if start > now or end < start:
            raise ValueError("invalid instance lifetime")
        # Absence from one API response alone does not stop accrual. Record a
        # verified destroy timestamp in the durable ledger after teardown.
        compute += rate * max(1, math.ceil((end - start).total_seconds() / 3600))
        if node["id"] not in current:
            continue
        actual = current[node["id"]]
        if actual.get("plan") != node["plan_id"]:
            reasons.append("instance plan differs from the approved quote")
        if actual.get("region") != session["region"]:
            reasons.append("instance region differs from approval")
        # Powering off is still billable. Never wait for an idle heartbeat then.
        if actual.get("power_status") == "stopped":
            reasons.append("powered-off campaign instance is still billable")
        activity = heartbeat_times.get(node["id"], node["created_at"])
        age = (now - timestamp(activity)).total_seconds()
        if age < -60:
            reasons.append("heartbeat timestamp is in the future")
        if age >= session["idle_timeout_seconds"]:
            reasons.append("campaign instance is idle or its heartbeat is missing")
    phase_ceiling = Decimal(str(session["session_budget_usd"]))
    previous = Decimal(str(session["spent_before_usd"]))
    reserve = Decimal(str(session.get("ancillary_reserve_usd", 7)))
    ceiling = Decimal(str(session.get("total_budget_usd", 80)))
    cutoff = Decimal(str(session.get("teardown_at_usd", 72)))
    if (
        not all(
            x.is_finite() for x in (phase_ceiling, previous, reserve, ceiling, cutoff)
        )
        or phase_ceiling <= 0
        or previous < 0
        or reserve < 0
        or not 0 < cutoff <= ceiling <= 80
    ):
        raise ValueError("invalid budget controls")
    if not 60 <= session["idle_timeout_seconds"] <= 900:
        raise ValueError("idle timeout must be 60-900 seconds")
    expires = timestamp(session["expires_at"])
    if compute >= phase_ceiling:
        reasons.append("approved session spend ceiling reached")
    total = previous + reserve + compute
    if total >= cutoff:
        reasons.append("campaign checkpoint/teardown threshold reached")
    if now >= expires:
        reasons.append("approved session expired")
    return {
        "estimated_compute_usd": str(compute),
        "estimated_total_usd": str(total),
        "alert_thresholds_usd": [t for t in (40, 60, 72, 80) if total >= t],
        "stop_reasons": sorted(set(reasons)),
        "destroy_required": bool(reasons and live),
        "active_instance_ids": sorted(current),
        "checked_at_utc": now.astimezone(UTC).isoformat(),
    }


def register_instances(
    session: dict[str, Any],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Adopt only quoted role/plan allocations during a preapproved launch.

    The launch ledger is stored before Terraform apply. This closes the gap
    where the operator disconnects after creation but before recording VM IDs.
    """
    import copy

    result = copy.deepcopy(session)
    if not result.get("registration_pending"):
        return result
    quotes = result["approved_quotes"]
    approved_at = timestamp(result["approved_at"])
    expires = timestamp(result["expires_at"])
    known = {node["id"] for node in result["nodes"]}
    for actual in inventory:
        if (
            result["campaign_tag"] not in actual.get("tags", [])
            or actual["id"] in known
        ):
            continue
        roles = [role for role in quotes if f"role-{role}" in actual.get("tags", [])]
        if len(roles) != 1:
            raise ValueError("unregistered instance lacks a unique approved role")
        role = roles[0]
        quote = quotes[role]
        if actual["plan"] != quote["plan_id"] or actual["region"] != result["region"]:
            raise ValueError("new instance differs from approved role/plan/region")
        if any(node.get("role") == role for node in result["nodes"]):
            raise ValueError("duplicate instance for an approved role")
        created = timestamp(actual["date_created"])
        if not approved_at <= created <= expires:
            raise ValueError("instance was not created within the approved session")
        result["nodes"].append(
            {
                "id": actual["id"],
                "role": role,
                "plan_id": actual["plan"],
                "hourly_usd": quote["hourly_usd"],
                "created_at": created.isoformat(),
                "activity_id": "job",
            }
        )
        known.add(actual["id"])
    if len(result["nodes"]) == len(quotes):
        result["registration_pending"] = False
    return result
