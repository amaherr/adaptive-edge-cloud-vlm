# Vultr migration status

## P0 — baseline freeze

- Branch: `vultr-cloud-backup`; starting SHA and inherited commits: [decisions](decisions.md).
- Implemented paths: roadmap preserved; decisions and this status report added.
- Baseline: Python 3.12.3 development environment; runtime image remains Python 3.11.
- `.venv/bin/pytest -q`: **218 passed, 1 skipped**, 43.91 seconds. The TCP
  subprocess test skipped because the sandbox prohibits loopback binding.
- `.venv/bin/ruff check .`: passed; `.venv/bin/mypy`: see validation record.
- Terraform state/plan: none for Vultr. Docker digest: none selected for Vultr.
- Plans, region and quote: unselected; no authenticated account evidence.
- Cost: no Vultr paid resources created; no budget authorized.
- P0: baseline tests pass with disclosed sandbox skip; clean-tree requirement
  is satisfied by committing the supplied roadmap and local phase records.
- Next: read-only P1 discovery, followed by owner account/capacity/budget review.

## Gate sequence

P1 remains **BLOCKED** pending account credentials, GPU entitlement/concurrency,
verified plan/OS compatibility, quotes and owner budget/launch approval.
Preferred region: Atlanta, USA (`atl`), selected by the owner; GPU capacity
and two-VM availability there remain unverified.
P2–P10 and full Gate 2 are **NOT RUN**. Local preflight tests do not establish
provider feasibility. Stop at P1 before infrastructure/provisioning work per
roadmap section 14. No paid launch or 8,100-attempt campaign is authorized.

## P1 — read-only tooling and external selection gate

- P0 commit: `16e573e` (`docs(vultr): freeze branch baseline and plan`).
- Implemented: `scripts/vultr/check_plans.py`, `tests/test_vultr_preflight.py`,
  cost worksheet, operator runbook and validation record; README links added.
- Offline preflight tests: 13 passed. Full verification recorded in
  [validation](validation.md), raw output `artifacts/vultr/p1/tests.txt`.
- Missing-token check returned 2 and wrote
  `artifacts/vultr/p1/missing-token.json`; no provider request was attempted.
- Terraform state/plan: none. Docker digest: none selected. Preferred region: Atlanta (`atl`); plan IDs/hourly
  quotes: unknown. GPU entitlement and concurrent two-VM quota: unknown.
- Cost incurred by these actions: no paid Vultr operations. Owner planning
  target: approximately $80 total; individual paid launch approval outstanding. P1: **BLOCKED**. Infrastructure and GPU testing remain NOT RUN.
- Next exact step: owner supplies account/API setup, API credentials; run the discovery command in the runbook with a fresh output
  filename, verify provider GPU access and quote, then review the P1 selection.
- Final P1 checks: Ruff and mypy passed; full pytest coverage run: 231 passed,
  1 sandbox skip, 81.65% coverage. Gate remains BLOCKED by external prerequisites.


## P1 — budget review and public catalog evidence

- Owner planning target: **approximately $80 total**, not a paid launch approval.
- Alert destination: **a.maher1506@gmail.com**; delivery not configured or tested.
- Owner is setting up the API key and will notify the agent when ready.
- Public unauthenticated catalog collected at 11:40 UTC, 7 November 2026:
  only three L40S plans, with hourly rates $1.671/$3.342/$6.684 and empty
  locations. Atlanta returned no available GPU plans (including VPC-only).
  This is not an account entitlement or authenticated capacity check.
- Evidence: [public API snapshot](evidence/public_gpu_catalog_2026-11-07.json).
  SHA-256: `4a51e84dd6c2aa5860427afa5425e78e474da030eeeb8805d11d6b697e4402fc`.
- [Cost estimate](cost_estimate.md): conditional A16/L40S pair at $2.142/hour;
  5 s mean attempt scenario about $59 with existing S3 allowances versus
  about $70 with one month of Vultr storage (additional ancillary costs apply).
  At 10 s mean, about $90/$101 respectively. No Vultr runtime is measured.
- Proposed, not approved: $5 single-node preflight; cumulative alerts at
  $40/$60/$72/$80; checkpoint and verified teardown threshold $72.
- P1 remains **BLOCKED**: authenticated account/GPU verification, actual capacity,
  hardware fit, concrete quote and explicit preflight launch approval outstanding.
- Next: after the owner confirms API setup, perform authenticated Atlanta
  discovery; if still empty, obtain provider guidance before proposing a region
  change. No paid launch or alert was sent. Terraform/Docker/GPU gates NOT RUN.

## Current — controls deployed; single-node preflight preparation

The owner approved S3 reuse, the $5 single-node preflight, alerts/checkpoints/
teardown and destruction of idle campaign VMs. Corrected all-type discovery
finds Atlanta A16 plans at $0.471/$0.942 per hour; the old filtered-empty result
was a tool bug, not a provider capacity refusal. Ubuntu 24.04 LTS x64 is selected
for preflight (API OS ID 2284). No GPU has been launched at this checkpoint.

- Commits: `3392b5b` S3 checkpointing; `1c33af1` corrected GPU discovery.
- Live S3 API and CPU fake-adapter byte-matching restore/resume passed.
- AWS operations: deployed, five-minute EventBridge schedule **ENABLED**;
  independent Lambda/S3 execution lease; SNS subscription **confirmed** and
  labelled test publication accepted; deletion-only CodeBuild worker deployed.
- Hosted empty-state teardown/API verification: **SUCCEEDED** on retry after
  owner allowed IPv4 API access. First attempt's HTTP 401 and the initial AWS
  reserved-concurrency rejection were resolved; neither launched GPUs.
- Separate state keys: `vultr-v1/operations/terraform.tfstate`,
  `vultr-v1/foundation/terraform.tfstate`, `vultr-v1/gpu/terraform.tfstate`.
- Vultr private VPC, dedicated public SSH key and operator-/32 SSH firewall
  applied from a reviewed plan; no public inference rule.
- API key: encrypted SSM; absent from Git/Terraform variables and GPU VMs.
- Tests: Ruff/mypy passed; **267 passed, 1 sandbox skip**, **81.64%** coverage;
  seven Terraform mock safety tests and all three provider schema checks passed.
- Spending: no GPU charges yet; small S3/CodeBuild/control usage is incurred
  within the $7 planning allowance. Exact provider invoice has not been queried.
- Gate P1: candidate plans/account/OS selection verified; concurrent pair
  entitlement and per-device layout remain unverified. Single-node preflight is
  authorized. P2/P3/P5/P10 live GPU acceptance remains **NOT COMPLETE**.
- Next: finalize the immutable control package, register the approved $5 session
  before VM creation, review the one-node Terraform plan, inspect actual driver/
  Docker/GPU, test the immutable application image, then destroy via Terraform
  and independently verify zero campaign nodes. Full Gate 2 is NOT AUTHORIZED.
