"""Read-only Vultr GPU catalog evidence; never approves or launches resources."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

API = "https://api.vultr.com/v2"


class PreflightError(RuntimeError):
    """A safe-to-display failure without credentials or provider response bodies."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PreflightError("API redirect refused; credentials were not forwarded")


class Client:
    def __init__(self, token: str):
        self.token = token
        self.opener = build_opener(NoRedirect())

    def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{API}{path}"
        if params:
            url += "?" + urlencode(params)
        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with self.opener.open(request, timeout=30) as response:
                result = json.load(response)
        except HTTPError as exc:
            reasons = {
                401: "invalid or missing API credentials",
                403: "access denied; check API source IP restrictions and permissions",
                429: "rate limited; wait before retrying discovery",
            }
            raise PreflightError(
                f"HTTP {exc.code}: " + reasons.get(exc.code, "provider request failed")
            ) from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise PreflightError("API connection or JSON decoding failed") from None
        if not isinstance(result, dict):
            raise PreflightError("API returned a non-object response")
        return result

    def pages(self, path: str, key: str, params: dict | None = None) -> list:
        query = {"per_page": 500, **(params or {})}
        rows = []
        seen = set()
        for _ in range(1000):
            page = self.get(path, query)
            batch = page.get(key)
            if not isinstance(batch, list):
                raise PreflightError(f"API response missing {key} list")
            rows.extend(batch)
            meta = page.get("meta") or {}
            if not isinstance(meta, dict):
                raise PreflightError("Invalid pagination metadata")
            links = meta.get("links") or {}
            if not isinstance(links, dict):
                raise PreflightError("Invalid pagination links")
            next_link = links.get("next")
            if not next_link:
                return rows
            if not isinstance(next_link, str):
                raise PreflightError("Invalid pagination link")
            # Extract only a cursor; never send credentials to an API-supplied URL.
            parts = urlsplit(next_link)
            if (
                parts.scheme not in ("", "https")
                or parts.netloc not in ("", "api.vultr.com")
                or parts.path not in ("", f"/v2{path}")
            ):
                raise PreflightError("Unexpected pagination destination")
            cursor = parse_qs(parts.query).get("cursor", [None])[0]
            if not cursor or cursor in seen:
                raise PreflightError("Missing or repeated pagination cursor")
            seen.add(cursor)
            query["cursor"] = cursor
        raise PreflightError("Pagination limit exceeded")


def collect(client: Client, region_ids: list[str]) -> dict:
    # The catalogs can be public. A successful account GET verifies token access
    # only; account finances and personal details are deliberately not archived.
    account = client.get("/account")
    if not isinstance(account.get("account"), dict):
        raise PreflightError("Account access was not verified")
    plans = client.pages("/plans", "plans", {"type": "vcg"})
    regions = client.pages("/regions", "regions")
    if any(
        not isinstance(row, dict) or not isinstance(row.get("id"), str)
        for row in plans + regions
    ):
        raise PreflightError("Catalog row missing string ID")
    regions_by_id = {row["id"]: row for row in regions}
    selected = sorted(set(region_ids) if region_ids else regions_by_id)
    if any(region not in regions_by_id for region in selected):
        raise PreflightError("Requested region is absent from provider catalog")
    availability = {}
    for region in selected:
        availability[region] = client.pages(
            f"/regions/{quote(region, safe='')}/availability",
            "available_plans",
            {"type": "vcg"},
        )
        if any(not isinstance(item, str) for item in availability[region]):
            raise PreflightError("Availability contains a non-string plan ID")
    gpu_plans = []
    for plan in plans:
        if plan.get("type") != "vcg" and not plan["id"].startswith("vcg-"):
            continue
        # Preserve API fields/units. Missing hourly price or GPU identity remains
        # unknown; do not infer VRAM from SKU text or divide monthly prices.
        gpu_plans.append(
            {
                "provider_plan": plan,
                "available_regions": [
                    r for r in selected if plan["id"] in availability[r]
                ],
                "hourly_cost": plan.get("hourly_cost"),
                "gpu_type": plan.get("gpu_type"),
                "gpu_vram": plan.get("gpu_vram"),
            }
        )
    return {
        "account_api_access_verified": True,
        "regions": [regions_by_id[r] for r in selected],
        "availability": availability,
        "gpu_plans": gpu_plans,
        "limitations": [
            "Catalog stock is not a reservation or two-VM concurrency approval",
            "GPU entitlement, quotas, OS/VPC support and driver fit need verification",
            "Null price/GPU fields require a provider quote; no SKU inference used",
            "Budget and explicit owner launch approval remain required",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        help="Region ID; repeat to select several (default: all)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New evidence file; existing reports are never overwritten",
    )
    args = parser.parse_args(argv)
    report = {
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "api_base": API,
        "gate": "BLOCKED",
        "launch_authorized": False,
    }
    try:
        token = os.environ.get("VULTR_API_KEY", "").strip()
        if not token:
            raise PreflightError(
                "Set VULTR_API_KEY through the local secret environment"
            )
        report.update(collect(Client(token), args.region))
        report["discovery_status"] = "COLLECTED"
    except PreflightError as exc:
        report.update(discovery_status="BLOCKED", error=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Read-only P1 evidence: {args.output}; launch remains BLOCKED")
    return 0 if report["discovery_status"] == "COLLECTED" else 2


if __name__ == "__main__":
    sys.exit(main())
