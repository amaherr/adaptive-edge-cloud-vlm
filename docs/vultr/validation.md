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

## Checkpoint and independent controls implementation

- Full local verification: Ruff/mypy passed; 267 tests passed, one sandbox
  loopback skip, 81.64% coverage in 38.73 seconds. Raw output:
  `artifacts/vultr/validation/tests-final.txt`.
- Terraform 1.10.5 / Vultr 2.33.2 / AWS 5.100.0 schemas validated. Seven mock
  safety tests passed across foundation, disposable GPU compute and operations.
- S3 API/CPU fake-adapter live evidence: [checkpoint acceptance](checkpoints.md).
- Reviewed `control-005.tfplan` created separate operations resources. AWS
  rejected reserved Lambda concurrency because of the account's minimum
  unreserved pool; the failed function was replaced through reviewed
  `control-007.tfplan` using a tested 120-second S3 execution lease instead.
- The confirmed SNS subscription accepted labelled delivery-test message
  `a3914f74-2d0d-51f7-a2d7-2c1d3171b92b`. Receipt is not asserted without owner
  acknowledgement. Manual Lambda invoke returned NO_REGISTERED_SESSION.
- First hosted empty-state worker applied zero resource changes, then failed
  Vultr inventory verification with HTTP 401. The owner permitted IPv4 API
  access. The retry succeeded, including a zero-campaign-GPU API check and
  durable S3 record. See committed hosted control acceptance summary.
- `control-008.tfplan` enabled the independent five-minute schedule and added
  an STS-assumed ECR read-only role. It changed no existing AWS campaign stack.
- Reviewed `foundation-001.tfplan` applied four isolated Vultr VPC/SSH/firewall
  resources, no GPU and no public inference rule.
- No GPU/driver/BF16/model/network or real VM-destruction acceptance result is
  claimed by these control/storage tests. Single-node preflight follows.

## Actual preflight and independent stopped-instance teardown

One reviewed Atlanta A16 / Ubuntu 24.04 allocation was created at 12:58:49 UTC.
All Linux SSH probes timed out; Windows TCP-22 also timed out. Two independent
public-IP checks matched the configured provider/host SSH source /32. Public
GitHub SSH was reachable. The owner reported console connection failure. One
controlled reboot returned HTTP 200; SSH remained inaccessible. No CUDA/driver,
Docker, real model, throughput or quality result was obtained.

The independent watcher adopted the VM automatically and observed stopped,
still-billable compute at 13:16:10 UTC. Its hosted job
`adaptive-vlm-vultr-teardown:d256cf1a-6704-4415-a6ba-0b2a5f79a200` **SUCCEEDED**:
a deletion-only plan removed the GPU and API verification returned zero campaign
instances at 13:22:00 UTC. A separate local API check at 13:28:11 UTC agreed.
The compute lifetime upper bound is 1,391.54 seconds; one-hour-minimum estimate
$0.471 before extras. Local launch inputs were reset to false.

[Nonsecret destruction summary](evidence/preflight_teardown_2026-11-07.json).
Observer diagnostics were independently restored from their S3 SHA-256 index
after the VM was gone; this does not substitute for real GPU benchmark-journal
resume or full Gate 2 acceptance. Full phase/limitations:
[preflight record](preflight_2026-11-07.md). No core software changed after the
last 267-pass/81.64% run; final changes are evidence and operator documentation.

Final closure: **11 observer diagnostic files** were restored after destruction
from immutable S3 index SHA-256
`b7bd8c5931a947a0660c3e971af921decafacbd7d93dc5bf823e6342146cbab9`;
every hash and byte comparison passed. The final GPU Terraform plan with the
local deploy flag false returned **exit 0 / No changes**. The temporary SSH
private key/public copy/known-hosts file were removed after independent zero-GPU
verification. A future retry needs a fresh operator SSH key/reference.
`main` remains `22567a67`; `cloud-infrastructure` remains `621e5127`.
