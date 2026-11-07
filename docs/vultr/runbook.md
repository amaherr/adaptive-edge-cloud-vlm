# Vultr operator runbook

Current phase: [single-node preflight blocked; GPU destroyed](preflight_2026-11-07.md).
The control plane is deployed and no campaign GPU remains. The local launch flag
is false. Resolve access through the [support-request draft](atl_preflight_support_request.md)
before repeating paid work. The draft has not been sent.

## Credentials and read-only discovery

AWS CLI profile: `cloud-infrastructure`. Refresh with
`aws login --profile cloud-infrastructure` when required. The Vultr key is an
SSM SecureString at `/adaptive-vlm/vultr/api-key` in us-east-1. Never paste it into
chat, Terraform input/state, command arguments, source or a GPU image/VM.
To rotate it, run `scripts/vultr/configure_key.py` in the trusted terminal where
`VULTR_API_KEY` is exported. Hosted roles read the encrypted value at runtime.

Read-only discovery from the repo, without printing the key:

```bash
set +x
VULTR_API_KEY="$(aws --profile cloud-infrastructure --region us-east-1 \
  ssm get-parameter --name /adaptive-vlm/vultr/api-key --with-decryption \
  --query Parameter.Value --output text --no-cli-pager)" \
  .venv/bin/python scripts/vultr/check_plans.py --region atl \
  --output artifacts/vultr/p1/FRESH_REPORT_NAME.json
```

Use a new filename on every run. Exit 0 means discovery collected, not launch
approval or P1 passed; exit 2 means blocked discovery. The tool paginates all
plan families and joins ordinary/VPC-only availability. GPU SKUs can have API
type `vdm`; `type=vcg` alone misses A16/A40 stock. Raw GPU fields retain provider
units, missing values remain unknown, and GPU VRAM is not inferred from SKU
strings. Public listing/account API access do not prove concurrency, VPC support,
driver fit or a stock reservation.

## Approved budget and launch prerequisites

Approximately $80 total; at most $5 for the single-node preflight; one failed
allocation has an estimated $0.471 compute cost before tax/control overhead.
Carry that cost into any retry's remaining preflight allowance and campaign
ledger. Do not reset cumulative spend or silently launch a second experiment.

Region Atlanta; preflight candidate `vcg-a16-6c-60g-16vram`, $0.471/hour;
Ubuntu 24.04 LTS x64, OS ID 2284. This OS/plan combination was allocated but its
actual guest/driver/VRAM could not be observed. A 32 GB aggregate A16 plan is
also listed at $0.942/hour; per-device layout and stronger-cloud execution
remain unverified. [Updated cost sensitivity](cost_estimate.md).

Before a retry, verify supported image/access, refresh actual quote/stock,
generate a dedicated SSH key outside Terraform, use the exact operator IPv4
/32, and review a saved Terraform plan. The temporary private SSH key was removed after verified destruction;
prepare a fresh key/reference for a retry. Persist a new approved
session/expiry/quote to independent S3 before GPU creation. See
[operations admission](../../infra/terraform/vultr/operations/README.md).

`scripts/vultr/terraform.sh` loads the encrypted key privately and reuses the
existing AWS temporary-credential bridge. Never use the AWS experiment state
keys for Vultr. Defaults create zero GPUs. GPU compute is isolated from retained
storage/state/networking so destruction cannot erase independent checkpoints.

## Backups, alerts and teardown

- [Checkpoint protocol and restore commands](checkpoints.md): immutable
  hash-verified incremental segments, writer lock, strict contract and original
  runner `--resume`. Cadence is 25 rows/300 seconds between trials plus boundaries.
- [Independent controls](../../infra/terraform/vultr/operations/README.md):
  EventBridge every 5 min, Lambda/S3 execution lease, SNS to
  `a.maher1506@gmail.com`, conservative cost ledger and deletion-only CodeBuild.
- Alerts at $40/$60/$72/$80. $72, session budget/expiry, stopped state or idle
  job activity requests checkpoint/stop, then verified destruction after grace.
  Actual teardown worked for the inaccessible stopped VM without using SSH.
- The SNS subscription is confirmed and publications were accepted; the owner
  has not yet acknowledged actual receipt of the labelled test email.
- Stop measured scheduling, fsync/commit and verify backups before planned
  teardown. On a broken guest, preserve the last committed independent data;
  do not wait indefinitely while compute remains billable. Record any lost
  uncommitted/in-flight work honestly.

The existing published immutable application image is retained in ECR; it does
not yet contain the new checkpoint CLI/pinned AWS CLI integration. Rebuild and
verify the shared image before containerized profiling. No actual CUDA/model
or independent-host correctness result is claimed by storage/control tests.

After teardown query provider inventory independently, verify zero campaign
GPU IDs, record destroyed timestamps and spend, and reset local deploy flags.
Restore into a fresh directory and verify all hashes before any compatible
resume. Preserve physical/AWS campaign evidence and final holdouts.
