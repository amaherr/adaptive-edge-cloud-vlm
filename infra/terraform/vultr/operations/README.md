# Independent Vultr control plane in AWS

This stack uses existing protected S3 buckets; it never owns or destroys them.
State key: `vultr-v1/operations/terraform.tfstate`. GPU state uses the independent
`vultr-v1/gpu/terraform.tfstate` key; VPC/SSH/firewall state uses
`vultr-v1/foundation/terraform.tfstate`. Never reuse the AWS campaign state keys.

The owner authorized the approximately $80 campaign target, $5 single-node
preflight, S3 reuse, email alerts, checkpoint/teardown controls and destruction
of idle campaign VMs. Full Gate 2 launch remains separately gated.

## Architecture

- EventBridge invokes Lambda every five minutes, independent of the GPU VMs and
  the operator's laptop. It creates a 120-second conditional S3 execution lease,
  longer than its 90-second hard timeout. Small AWS accounts need no reserved
  Lambda concurrency quota.
- The API key is an encrypted SSM SecureString at `/adaptive-vlm/vultr/api-key`.
  It is absent from Terraform input/state, source ZIP and GPU VMs. Run
  `scripts/vultr/configure_key.py` in the terminal with the exported key to set
  or rotate it. The owner permitted IPv4 API access for hosted AWS callers;
  validate this from CodeBuild, not just the local terminal.
- SNS emails **a.maher1506@gmail.com**. The confirmed subscription accepted a
  labelled delivery-test publication. API acceptance is not a human delivery
  acknowledgement.
- The guard counts all registered VM lifetime, including powered-off instances,
  rounded up per node/hour. Destroyed lifetimes remain in its ledger. It triggers
  alerts at $40/$60/$72/$80; $72, session budget/expiry, idle/missing heartbeat,
  stopped VMs or approved-plan drift request shutdown.
- It writes STOP into each registered checkpoint prefix, waits at least 180
  seconds (normally until the next 5-minute poll), then starts CodeBuild.
  Failed email publication cannot disable STOP/teardown enforcement.
- CodeBuild sets `deploy_gpu_instances=false`, plans, validates **only deletion
  of exactly the registered campaign GPU IDs**, applies the saved plan, queries
  Vultr independently for zero remaining campaign VMs, and records destruction
  times. It rejects create/update/replacement, unrelated resource deletion,
  missing-state/empty-plan hiding of registered VMs and tag/identity mismatches.
  Its IAM role can delete only the dedicated GPU Terraform state/lock objects;
  it has no deletion permission on retained artifacts.
- Lambda and CodeBuild failure alarms publish to the same SNS topic. The small
  build runs only for teardown, not continuously for cost polling.
- A separate STS-assumed role grants ECR read-only access to the existing app
  repository. It grants no ECR image publication/deletion permission.

## Session admission before GPU creation

Write `BUCKET/research/adaptive-vlm/vultr-v1/control/session.json` **before apply**,
with conditional create/CAS. It must contain:

```text
session_id, approved=true, approved_at, region, campaign_tag,
expires_at, session_budget_usd=5 (preflight), total_budget_usd=80,
teardown_at_usd=72, ancillary_reserve_usd=7, spent_before_usd,
idle_timeout_seconds=600, nodes=[], registration_pending=true,
approved_quotes={edge:{plan_id, hourly_usd}},
checkpoint_prefixes=[dedicated /runs/... prefixes, if running benchmarks],
terraform_variables={the exact reviewed GPU variables, including approved role
                     plans, OS, VPC/firewall/SSH refs, expiry and approval ID}
```

Use a unique `adaptive-vlm-vultr-...` campaign tag for each admitted session.
Before replacing a prior ledger, verify its nodes are destroyed, archive its
final state and carry prior compute charges into `spent_before_usd`; never reset
spending to zero to bypass the campaign ceiling. Include persistent service
charges in the allowance or explicit prior cost. Estimates are not the provider
invoice and cannot guarantee tax/FX or delayed-billing precision.

If the operator dies during `apply`, the watcher adopts only new instances with
an approved role tag/plan/region and creation timestamp inside the approval
window. This closes the post-creation/pre-registration gap. New nodes share
`activity_id=job`: a cloud intentionally idle during the edge-only policy must
not be destroyed while the paired job is active. Additional unregistered,
wrong-role or drifted allocations require inventory review; never silently
adopt unrelated VMs.

Use `scripts/vultr/publish_activity.py` every minute, pointing to the existing
`ADAPTIVE_VLM_HEARTBEAT` job marker. It uploads only recent actual activity,
not a timer-generated unconditional keepalive. Example:

```bash
.venv/bin/python scripts/vultr/publish_activity.py \
  --bucket YOUR_EXISTING_BUCKET --profile cloud-infrastructure \
  --activity-id job --local-marker /path/to/job.heartbeat
```

A stalled/disconnected operator stops producing activity; the independent
watchdog still enforces the idle/expiry deadline. For model bootstrap, publish
actual setup/download progress under the same approved session. After a STOP,
archive, review and create a new approved session; do not silently clear it.

## Packaging and Terraform

`package_ops.py` builds an explicit allowlist ZIP containing only Python control
code and locked GPU Terraform source; no .tfvars, state, credentials, data or
model weights are included. Regenerate and redeploy whenever those files change.
The host app model lock is untouched by the control-plane boto3 dependency.

```bash
.venv/bin/python scripts/vultr/package_ops.py --output artifacts/vultr/control.zip
# Supply ignored local tfvars for artifact_bucket, state_bucket and package_zip.
# Initialize this directory against its own existing-S3 backend key.
# terraform plan -out=... is mandatory before applying a saved plan.
```

Schedule defaults to **disabled** in reusable source. Set `enable_schedule=true`
only after secret/SNS setup and live no-GPU tests. No GPU launch is permitted
while the hosted provider API check or deletion-plan acceptance is failing.

The planned controls are within the $7 ancillary allowance, not asserted free:
Lambda/S3/SSM/SNS/log/alarm usage and occasional CodeBuild minutes accrue charges.
No NAT gateway, EC2 supervisor, retained GPU volume or paid Vultr object-storage
subscription is required by this stack. Retained service lifetime still counts
against the total budget.
