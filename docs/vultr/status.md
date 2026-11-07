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
region, verified plan/OS compatibility, quotes and owner budget/launch approval.
P2–P10 and full Gate 2 are **NOT RUN**. Local preflight tests do not establish
provider feasibility. Stop at P1 before infrastructure/provisioning work per
roadmap section 14. No paid launch or 8,100-attempt campaign is authorized.
