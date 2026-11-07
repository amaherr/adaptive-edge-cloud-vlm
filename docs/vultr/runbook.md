# Vultr operator runbook

## Current supported operation: read-only P1 discovery

Set `VULTR_API_KEY` in the trusted local secret environment. Never paste the
value into chat, command arguments, a Terraform variable or tracked files.
At the initial check the agent's environment had no key; a shell opened later may not inherit
variables from your separate terminal. You can run the command locally instead.

From the repository root, using a new evidence filename on every invocation:

```bash
.venv/bin/python scripts/vultr/check_plans.py \
  --output artifacts/vultr/p1/discovery-001.json
# To limit availability calls after choosing a provider region ID:
.venv/bin/python scripts/vultr/check_plans.py \
  --region atl --output artifacts/vultr/p1/discovery-002.json
.venv/bin/pytest -q tests/test_vultr_preflight.py
```

The tool uses only GET requests to the official API, verifies account API access,
then paginates GPU plans, regions and selected regional availability. It keeps
provider plan fields with their original units. Null hourly price, GPU model or
VRAM requires independent provider evidence; no monthly-to-hourly conversion or
SKU memory inference is performed. Account response details are not archived.
An exit code of 0 means discovery collected, **not P1 passed**; code 2 means
blocked discovery. Evidence files are exclusive-create to preserve old reports.
HTTP 401/403/429 produce a sanitized blocked report without response bodies.
Fix credentials/source-IP permissions or wait after rate limiting before retry.

Plans and availability may be public even with an Authorization header. Account
API access alone proves neither GPU entitlement nor quota, two concurrent VMs,
VPC compatibility or a stock reservation. Ask provider support for these when
not verifiable through account controls; save its nonsecret response as evidence.

## Required owner inputs before P1 can close

1. Billing/API activation and GPU entitlement status; credential setup as above.
2. Selected region: Atlanta, USA (`atl`). Confirm with the provider that two
   independent GPU VMs can run concurrently there; capacity remains unverified.
3. Total planning budget: approximately $80; alert destination:
   `a.maher1506@gmail.com` (recorded, not active). Proposed single-node
   preflight ceiling: $5, pending explicit approval of a concrete launch.
4. Checkpoint backend preference: existing AWS S3 or new Vultr object storage.
5. Before infrastructure: administrative SSH source CIDR/public key, supported
   GPU OS/image, retained registry/storage decisions and remote-state bootstrap.

The next implementation phase is isolated Vultr Terraform. It remains gated by
P1. There is currently no executable Vultr deploy/destroy path, GPU pilot, durable
remote checkpoint or evidence supporting READY. Refer to the supplied roadmap
for later phases and approval boundaries.

## API sources

- [Official plans API SDK documentation](https://github.com/vultr/vultr-ruby/blob/main/docs/PlansApi.md)
- [Official region availability API SDK documentation](https://github.com/vultr/vultr-ruby/blob/main/docs/RegionApi.md)
- [Provider plan fields](https://registry.terraform.io/providers/vultr/vultr/latest/docs/data-sources/plan)

Reviewed 7 November 2026. Runtime response fields remain authoritative. The tool
retains unknown provider plan fields instead of assuming a fixed GPU SKU schema.
