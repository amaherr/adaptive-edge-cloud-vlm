# Vultr migration status

Updated 7 November 2026. Branch: `vultr-cloud-backup`.
**Current gate: BLOCKED at single-node guest access. The backup is not READY.**

## Owner-approved scope

Atlanta (`atl`); approximately **$80 total**; existing AWS S3 reuse;
**$5 maximum single-node preflight**; alerts to **a.maher1506@gmail.com**;
checkpointing, off-VM cost/expiry/idle protection and destruction of idle GPU VMs.
Ubuntu 24.04 LTS x64 (OS ID 2284) selected for preflight. Full Gate 2 spend/run
is not authorized. The API key is encrypted in SSM and absent from Git,
Terraform variables and GPU VMs; the owner permitted IPv4 API access for hosted
AWS callers. The temporary key handoff file was removed.

## Implemented and verified

- `3392b5b`: immutable S3 segments/manifests, conditional writer lock/head,
  strict restore, optional benchmark checkpoint hooks outside inference timing.
  CPU fake-adapter live restore/resume reproduced files byte for byte.
- `1c33af1`: corrected provider discovery across all API families. A16/A40
  `vcg-` SKUs are classified `vdm`; the old `type=vcg` query incorrectly hid
  Atlanta stock. Actual Atlanta candidates: $0.471/hour 16 GB advertised A16
  and $0.942/hour 32 GB advertised aggregate allocation; no per-device VRAM
  pooling or performance advantage is assumed.
- `b3d21cb`: separate Vultr Terraform foundation/compute and independent AWS
  EventBridge/Lambda/SNS/CodeBuild controls. Schedule is **ENABLED every 5 min**.
  SNS subscription is **confirmed**; delivery-test and control publications
  were accepted. Human email-receipt acknowledgement remains unrecorded.
- Separate encrypted/versioned S3 state keys:
  `vultr-v1/foundation/terraform.tfstate`,
  `vultr-v1/operations/terraform.tfstate`, `vultr-v1/gpu/terraform.tfstate`.
  Existing AWS experiment/retained states and physical evidence were preserved.
- Seven Terraform mock safety tests and all three provider schemas passed
  (Terraform 1.10.5, Vultr 2.33.2, AWS 5.100.0). Ruff/strict mypy passed;
  **267 passed, 1 sandbox loopback skip**, **81.64%** coverage.

## First single-node result and cleanup

One `vcg-a16-6c-60g-16vram` VM was created from the reviewed plan at
**12:58:49 UTC**, with independent approval/quote/session registration committed
before creation. The watcher automatically adopted its approved role and ID.
Provider metadata reported active/running and VPC address `10.46.0.3`.

Repeated Linux SSH and Windows TCP-22 probes timed out. Two egress-IP checks
matched the configured /32 allowlist; GitHub SSH was reachable. The owner
reported the web console also failed to connect. One controlled reboot was
accepted; subsequent SSH probes still failed. Guest boot, NVIDIA, CUDA/BF16,
Docker and model fit could not be inspected. **Zero model requests ran.**

The watcher independently detected a stopped-but-billable instance at
**13:16:10 UTC** and initiated cleanup; the session was also expired early at
13:18:30 UTC after diagnostics were archived. Its CodeBuild worker applied the
reviewed deletion-only plan and verified destruction at **13:22:00 UTC**.
A separate operator API check confirmed **zero remaining campaign GPU VMs**.
Local `deploy_gpu_instances` was reset to **false**. Diagnostics were restored
from S3 after destruction with SHA-256 verification; this is observer evidence,
not a GPU benchmark-resume test.

Compute estimate: **$0.471**, before tax/transfer and small AWS control costs.
The $7 ancillary reserve is a planning allowance, not measured spending.
Independent control services, the retained S3/ECR data and non-GPU Vultr
VPC/SSH/firewall resources remain; no paid Vultr GPU, snapshot, block volume,
object-storage or registry subscription was left by this phase.

Evidence: [preflight record](preflight_2026-11-07.md),
[destruction proof](evidence/preflight_teardown_2026-11-07.json),
[diagnostic restore](evidence/preflight_diagnostic_restore_2026-11-07.json),
[checkpoint acceptance](checkpoints.md), [cost analysis](cost_estimate.md).
Raw local output: `artifacts/vultr/`; independent prefix:
`research/adaptive-vlm/vultr-v1/` in the existing artifact bucket.

## Corrected retry

The owner authorized one corrected retry. Raw cloud-init delivery matched the
intended minimal script, but SSH still timed out from the operator and an
independent AWS network. Console also failed over a private browser/hotspot.
One reboot left the VM stopped; explicit start returned HTTP 500.
Zero model requests ran. Temporary AWS probe resources and its exact /32
firewall rule were removed and independently verified absent. The session has
expired and the watchdog entered teardown grace; final deletion is pending.
See the [retry record](preflight_retry_2026-11-07.md).

Combined minimum compute estimate: **$0.942**, before tax/transfer and small AWS
controls/probe costs. The $7 ancillary allowance remains a reserve, not a bill.

## Remaining gates

- P1/P2: one-node provision/destroy verified; two-node entitlement, supported
  GPU image/guest connectivity and actual per-device layout remain unresolved.
- P3: GPU Docker/CUDA/BF16/model execution **NOT RUN**. The existing pinned
  application image was verified present, but no container executed on Vultr.
- P4/P6: portable profiler/provenance, Vultr campaign configs and independently
  measured two-peer netem link are **NOT RUN**.
- P5: local/live S3 fake-adapter recovery passed; real container kill, GPU-journal
  VM-recreation recovery and latency-interference pilot remain **NOT RUN**.
  The new benchmark checkpoint CLI also needs a reviewed image containing the
  updated app and pinned AWS CLI before container use.
- P7/P8: real hardware correctness/memory tests and three-policy pilot **NOT RUN**.
- P9/G2: **NOT RUN**: 8,100 measured attempts plus 27 warmups remain required.
- P10: real off-VM stopped-instance detection and automatic Terraform destruction
  passed without SSH. Full campaign protection/resume acceptance remains open.

Next exact step: verify retry teardown, then escalate guest/console/start failure
using the updated [support request](atl_preflight_support_request.md), which has
**not been sent**. Do not start further paid experiments before resolving access.
Carry both attempts into the $5 preflight limit and $80 campaign ledger;
preserve failed outcomes and immutable evidence.
