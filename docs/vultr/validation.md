# Vultr validation evidence

## P0 baseline

Starting commit: `621e5127d03ae7e0ee89f893e61cc464465e54f4`.
Development environment: existing `.venv`, Python 3.12.3.

- `.venv/bin/pytest -q`: 218 passed, 1 skipped in 43.91 s. The existing
  `test_server_subprocess.py` test skipped because sandbox loopback binding is
  prohibited; it was not counted as a passed network test.
- `.venv/bin/ruff check .`: all checks passed.
- `.venv/bin/mypy`: success. Raw lint/type outputs: `artifacts/vultr/p0/`.
- `git fetch --all --prune`: succeeded after filesystem permission escalation.
- Frozen SHA-256 inputs: [decisions](decisions.md); source files remain unchanged.

## P1 local tooling

Offline tests cover pagination/filter retention, repeated cursors, untrusted
pagination destinations, redirect refusal, sanitized HTTP 401/403/429/500,
region joins, unknown price/VRAM, missing token, append-only evidence filenames,
and successful discovery retaining a BLOCKED launch gate.

The missing-token CLI run records `artifacts/vultr/p1/missing-token.json` and
returns 2. It does not contact the provider. No authenticated inventory, real
quota, quoted GPU pair, container GPU smoke, Terraform plan/apply, network
measurement, remote restore or billing test has been performed for Vultr.

The 13 offline preflight cases passed with
`.venv/bin/pytest -q tests/test_vultr_preflight.py` (1.32 s).

Final P1 module verification:

- `.venv/bin/ruff check .`: passed.
- `.venv/bin/mypy`: success, no issues in 64 source files.
- `.venv/bin/pytest --cov=adaptive_vlm --cov-report=term-missing`:
  **231 passed, 1 skipped in 58.46 s**, coverage **81.65%** (required 80%).
  The same sandbox loopback-binding skip remains disclosed above.
- After pagination metadata hardening, the focused 13 tests passed again in
  1.03 s; full suite also passed with the final implementation.
- `git diff --check`: passed. Frozen model/data/config and all existing
  Terraform files have no changes relative to the starting commit.


## P1 cost review — 7 November 2026

Read-only unauthenticated HTTP GETs fetched public `/v2/plans?type=vcg&per_page=500`
and `/v2/regions/atl/availability?type=vcg`. Sandbox DNS initially failed; the
same read-only requests succeeded with network permission escalation. No API
credential, paid request, provisioning operation or email was used.

Committed raw public evidence:
`docs/vultr/evidence/public_gpu_catalog_2026-11-07.json`, SHA-256
`4a51e84dd6c2aa5860427afa5425e78e474da030eeeb8805d11d6b697e4402fc`.
Source timestamps: 11:40:28/29 UTC. All 3 catalog plans and the empty Atlanta
availability response were preserved; no pagination next link was present.

Cost arithmetic uses 8,100 measured attempts, a hypothetical $2.142/hour pair,
30% duration overhead plus four paired hours, $12 contingency and either $7
hybrid ancillary allowance or $18 one-month Vultr storage. These are explicit
planning assumptions, not measured GPU performance or an approved quote.
Arithmetic and input counts were checked with Python Decimal; no software
implementation changed, so the existing test suite was not rerun for this
pricing/documentation-only change. `git diff --check` passed.
