"""Collect read-only AWS identity, paired-AZ offerings, quota and price evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def collect(aws: str, profile: str, region: str) -> dict:
    def call(service: str, operation: str, *arguments: str, api_region: str = region):
        result = subprocess.run(
            [
                aws,
                "--profile",
                profile,
                "--region",
                api_region,
                "--output",
                "json",
                "--no-cli-pager",
                service,
                operation,
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return json.loads(result.stdout)

    identity = call("sts", "get-caller-identity")
    types = ("g6f.2xlarge", "g6.2xlarge")
    offerings = {}
    prices = {}
    for instance_type in types:
        response = call(
            "ec2",
            "describe-instance-type-offerings",
            "--location-type",
            "availability-zone",
            "--filters",
            f"Name=instance-type,Values={instance_type}",
        )
        offerings[instance_type] = sorted(
            row["Location"] for row in response["InstanceTypeOfferings"]
        )
        filters = [
            {"Type": "TERM_MATCH", "Field": field, "Value": value}
            for field, value in {
                "instanceType": instance_type,
                "regionCode": region,
                "operatingSystem": "Linux",
                "tenancy": "Shared",
                "preInstalledSw": "NA",
                "capacitystatus": "Used",
            }.items()
        ]
        price = call(
            "pricing",
            "get-products",
            "--service-code",
            "AmazonEC2",
            "--filters",
            json.dumps(filters),
            api_region="us-east-1",
        )
        prices[instance_type] = [json.loads(row) for row in price["PriceList"]]
        if not prices[instance_type]:
            raise RuntimeError(f"No On-Demand pricing product for {instance_type}")
    shared = sorted(set(offerings[types[0]]) & set(offerings[types[1]]))
    if not shared:
        raise RuntimeError("No common AZ offering for the frozen GPU pair")
    return {
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "identity": identity,
        "region": region,
        "offerings": offerings,
        "common_azs": shared,
        "availability_zones": call("ec2", "describe-availability-zones"),
        "instance_types": call(
            "ec2", "describe-instance-types", "--instance-types", *types
        ),
        "ec2_quotas": call(
            "service-quotas", "list-service-quotas", "--service-code", "ec2"
        ),
        "pricing_products": prices,
        "limitations": [
            "Offerings are not a capacity reservation",
            "AMIs, regional storage prices and permissions require review",
            "Quota consumption by existing instances must be checked",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument("--profile", default="cloud-infrastructure")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.aws_cli, args.profile, args.region)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Read-only evidence saved to {args.output}; review before any apply")


if __name__ == "__main__":
    main()
