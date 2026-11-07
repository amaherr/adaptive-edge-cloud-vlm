"""AWS Lambda control plane: durable alerts/STOP, then Terraform-only teardown."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import boto3
from botocore.exceptions import ClientError
from budget import evaluate, register_instances


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Vultr redirect refused")


def instances(token):
    rows, seen, cursor = [], set(), None
    for _ in range(100):
        query = {"per_page": 500}
        if cursor:
            query["cursor"] = cursor
        request = Request(
            "https://api.vultr.com/v2/instances?" + urlencode(query),
            headers={"Authorization": f"Bearer {token}"},
        )
        with build_opener(NoRedirect()).open(request, timeout=15) as response:
            data = json.load(response)
        rows.extend(data["instances"])
        link = data.get("meta", {}).get("links", {}).get("next")
        if not link:
            return rows
        parts = urlsplit(link)
        if (
            parts.netloc not in ("", "api.vultr.com")
            or parts.path not in ("", "/v2/instances")
            or parts.scheme not in ("", "https")
        ):
            raise RuntimeError("Unexpected Vultr pagination destination")
        cursor = parse_qs(parts.query).get("cursor", [None])[0]
        if not cursor or cursor in seen:
            raise RuntimeError("Invalid Vultr pagination cursor")
        seen.add(cursor)
    raise RuntimeError("Vultr pagination limit exceeded")


def handler(event, context):
    s3, sns = boto3.client("s3"), boto3.client("sns")
    bucket, prefix = os.environ["BUCKET"], os.environ["PREFIX"]
    now = datetime.now(UTC)

    def get(relative):
        try:
            response = s3.get_object(Bucket=bucket, Key=f"{prefix}/{relative}")
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return None
            raise
        return response

    def put(relative, value, **conditions):
        return s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}/{relative}",
            Body=json.dumps(value, sort_keys=True).encode(),
            ServerSideEncryption="AES256",
            **conditions,
        )

    def alert(identity, message):
        key = (
            "control/alerts/" + hashlib.sha256(identity.encode()).hexdigest() + ".json"
        )
        if get(key) is not None:
            return
        try:
            result = sns.publish(
                TopicArn=os.environ["TOPIC_ARN"],
                Subject="Adaptive VLM Vultr budget/control alert",
                Message=json.dumps(message, indent=2),
            )
        except Exception:
            # A broken email channel cannot disable cost enforcement. Do not
            # record delivery success; the next scheduled invocation retries.
            print("SNS publish failed; checkpoint/teardown enforcement continues")
            return
        put(
            key,
            {"sns_message_id": result["MessageId"], "at": now.isoformat()},
            IfNoneMatch="*",
        )

    # A 120-second lease outlasts this function's 90-second hard timeout.
    # This works even in small AWS accounts that cannot reserve concurrency.
    lease = get("control/watchdog-lease.json")
    if lease is not None:
        active = json.loads(lease["Body"].read())
        if datetime.fromisoformat(active["expires_at"]) > now:
            return {"state": "WATCHDOG_ALREADY_RUNNING"}
    condition = {"IfMatch": lease["ETag"]} if lease else {"IfNoneMatch": "*"}
    try:
        put(
            "control/watchdog-lease.json",
            {
                "expires_at": (now + timedelta(seconds=120)).isoformat(),
            },
            **condition,
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"PreconditionFailed", "412"}:
            return {"state": "WATCHDOG_ALREADY_RUNNING"}
        raise

    document = get("control/session.json")
    if document is None:
        return {"state": "NO_REGISTERED_SESSION", "gpu_launch_authorized": False}
    session = json.loads(document["Body"].read())
    session_id = session["session_id"]
    # Never accept artifact prefixes outside the dedicated Vultr campaign.
    for item in session["checkpoint_prefixes"]:
        if not item.startswith(prefix + "/runs/") or ".." in item.split("/"):
            raise ValueError("unapproved checkpoint prefix")
    heartbeats = {}
    for node in session["nodes"]:
        activity_id = node.get("activity_id", node["id"])
        if "/" in activity_id or activity_id in {"", ".", ".."}:
            raise ValueError("invalid heartbeat identity")
        beat = get("control/heartbeats/" + activity_id)
        if beat:
            heartbeats[node["id"]] = beat["LastModified"].isoformat()
    try:
        token = boto3.client("ssm").get_parameter(
            Name=os.environ["VULTR_KEY_PARAMETER"], WithDecryption=True
        )["Parameter"]["Value"]
        inventory = instances(token)
        registered = register_instances(session, inventory)
        if registered != session:
            put("control/session.json", registered, IfMatch=document["ETag"])
            session = registered
        report = evaluate(session, inventory, heartbeats, now=now)
    except Exception:
        # Fail closed: provider/secret failures must not leave approved compute
        # alive indefinitely. Teardown revalidates Terraform ownership itself.
        report = {
            "stop_reasons": ["provider/control verification failed"],
            "destroy_required": any(not n.get("destroyed_at") for n in session["nodes"])
            or bool(session.get("registration_pending")),
            "alert_thresholds_usd": [],
            "checked_at_utc": now.isoformat(),
        }
    for threshold in report["alert_thresholds_usd"]:
        alert(f"{session_id}:budget:{threshold}", report)
    if report["stop_reasons"]:
        alert(f"{session_id}:stop", report)
        for item in session["checkpoint_prefixes"]:
            try:
                s3.put_object(
                    Bucket=bucket,
                    Key=item + "/STOP",
                    Body=json.dumps(report).encode(),
                    IfNoneMatch="*",
                    ServerSideEncryption="AES256",
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
                    raise
    stop_key = f"control/stops/{session_id}.json"
    stopped = get(stop_key)
    if report["destroy_required"]:
        if stopped is None:
            put(stop_key, report, IfNoneMatch="*")
            return {"state": "CHECKPOINT_GRACE", **report}
        grace = (now - stopped["LastModified"]).total_seconds()
        if grace >= 180:
            # Idempotent across polls; a failed build is retried after 10 minutes.
            build_key = f"control/builds/{session_id}.json"
            build = get(build_key)
            previous = json.loads(build["Body"].read()) if build else None
            if previous:
                result = boto3.client("codebuild").batch_get_builds(
                    ids=[previous["build_id"]]
                )["builds"]
                if result and result[0]["buildStatus"] == "IN_PROGRESS":
                    return {"state": "TEARDOWN_RUNNING", **report}
                if (now - build["LastModified"]).total_seconds() < 600:
                    return {"state": "TEARDOWN_RETRY_DELAY", **report}
            result = boto3.client("codebuild").start_build(
                projectName=os.environ["TEARDOWN_PROJECT"],
                environmentVariablesOverride=[
                    {"name": "SESSION_ID", "value": session_id, "type": "PLAINTEXT"}
                ],
            )
            condition = {"IfMatch": build["ETag"]} if build else {"IfNoneMatch": "*"}
            put(build_key, {"build_id": result["build"]["id"]}, **condition)
            report["state"] = "TERRAFORM_TEARDOWN_STARTED"
    put(
        "control/reports/" + now.strftime("%Y%m%dT%H%M%S%fZ") + ".json",
        report,
        IfNoneMatch="*",
    )
    return report
