"""CodeBuild: review a deletion-only GPU plan, apply it and verify the API."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import boto3
from watchdog import instances


def validate_plan(plan, session):
    approved = {node["id"] for node in session["nodes"] if not node.get("destroyed_at")}
    deleted = set()
    for resource in plan.get("resource_changes", []):
        change = resource["change"]
        actions = change["actions"]
        if actions in (["no-op"], ["read"]):
            continue
        if (
            actions != ["delete"]
            or resource["type"] != "vultr_instance"
            or resource["name"] != "gpu"
            or resource.get("module_address")
            or change["before"]["id"] not in approved
            or session["campaign_tag"] not in change["before"].get("tags", [])
        ):
            raise ValueError("refusing a plan outside approved campaign GPU deletions")
        deleted.add(change["before"]["id"])
    # Do not silently accept an empty/wrong state when registered VMs exist.
    if deleted != approved:
        raise ValueError("plan does not account for every registered live GPU node")


def main():
    s3 = boto3.client("s3")
    bucket, prefix = os.environ["BUCKET"], os.environ["PREFIX"]
    response = s3.get_object(Bucket=bucket, Key=prefix + "/control/session.json")
    session = json.loads(response["Body"].read())
    if session["session_id"] != os.environ["SESSION_ID"] or not session["approved"]:
        raise ValueError("teardown session changed or was not approved")
    variables = dict(session["terraform_variables"])
    if (
        variables["campaign_tag"] != session["campaign_tag"]
        or variables["region"] != session["region"]
    ):
        raise ValueError("Terraform scope differs from the approved session")
    variables["deploy_gpu_instances"] = False
    work = Path("infra/terraform/vultr/environments/gpu")
    path = work / "teardown.auto.tfvars.json"
    path.write_text(json.dumps(variables))

    def terraform(*args, capture=False):
        # Terraform errors can contain sensitive provider diagnostics. CodeBuild
        # log access is restricted; never print the key or an environment dump.
        result = subprocess.run(
            ["terraform", f"-chdir={work}", *args],
            check=True,
            capture_output=capture,
            text=True,
            timeout=600,
        )
        return result.stdout

    terraform(
        "init",
        "-input=false",
        "-lockfile=readonly",
        "-backend-config=bucket=" + os.environ["STATE_BUCKET"],
        "-backend-config=key=vultr-v1/gpu/terraform.tfstate",
        "-backend-config=region=us-east-1",
        "-backend-config=use_lockfile=true",
    )
    terraform("plan", "-input=false", "-out=teardown.tfplan")
    plan = json.loads(terraform("show", "-json", "teardown.tfplan", capture=True))
    validate_plan(plan, session)
    evidence = prefix + "/control/teardowns/" + session["session_id"]
    s3.put_object(
        Bucket=bucket,
        Key=evidence + "/plan.json",
        Body=json.dumps(plan).encode(),
        ServerSideEncryption="AES256",
    )
    terraform("apply", "-input=false", "teardown.tfplan")
    remaining = [
        node["id"]
        for node in instances(os.environ["VULTR_API_KEY"])
        if session["campaign_tag"] in node.get("tags", [])
    ]
    if remaining:
        raise RuntimeError("campaign instances remain after Terraform apply")
    destroyed = datetime.now(UTC).isoformat()
    for node in session["nodes"]:
        if not node.get("destroyed_at"):
            node["destroyed_at"] = destroyed
    # A concurrent new session must never be overwritten with an old destroy log.
    s3.put_object(
        Bucket=bucket,
        Key=prefix + "/control/session.json",
        Body=json.dumps(session).encode(),
        IfMatch=response["ETag"],
        ServerSideEncryption="AES256",
    )
    s3.put_object(
        Bucket=bucket,
        Key=evidence + "/verified.json",
        Body=json.dumps(
            {"remaining_campaign_gpu_ids": remaining, "verified_at": destroyed}
        ).encode(),
        ServerSideEncryption="AES256",
    )


if __name__ == "__main__":
    main()
