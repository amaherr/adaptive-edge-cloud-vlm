# Single-node Vultr preflight — connectivity blocked

The authorized $5 preflight created one Atlanta A16 VM using a reviewed
Terraform plan and the selected Ubuntu 24.04 LTS x64 image. Its approval,
$0.471/hour quote, two-hour expiry and session budget were stored in independent
S3 **before** creation. The independent watcher automatically adopted the
created role/plan/instance into its ledger.

The guest could not be reached over SSH. API metadata reported active/running,
but repeated Linux SSH and Windows TCP port-22 probes timed out. The operator's
current IPv4 matched the exact provider and host SSH allowlist; two independent
egress checks agreed. GitHub SSH port 22 was reachable. The owner independently
reported that View Console failed to connect to the server.

One controlled API reboot was accepted (HTTP 200). Further SSH probes still
timed out. The cause is **not established**: guest boot, provider/hypervisor
console path, routing and security enforcement could not be inspected. Do not
attribute the failure to the model or GPU-memory feasibility without evidence.

At **13:18:30 UTC**, the session expiry was shortened to stop scheduling. All
accessible diagnostic files were uploaded under immutable SHA-256-suffixed keys
and read back from the existing independently retained S3 bucket. No model
request was started; CUDA/BF16, GPU identity, Docker and model-fit checks are
**NOT RUN**. This preflight is **BLOCKED**, not PASS.

Session: `preflight-20261107T125414Z`; instance:
`70b5a674-f823-47a0-aa85-3dc32d6e5c60`; campaign tag:
`adaptive-vlm-vultr-pf01`; creation: **12:58:49 UTC**.

Operator/VM diagnostic artifacts remain under `artifacts/vultr/preflight/`.
Independent archive prefix:
`research/adaptive-vlm/vultr-v1/control/preflights/preflight-20261107T125414Z/`.
The local archive index lists each immutable key and SHA-256; no original
physical experiment artifact, model lock or dataset was overwritten.

The independent watcher detected a **stopped but still billable** instance at
13:16:10 UTC after the reboot. It requested shutdown and launched CodeBuild after
the grace interval. The hosted worker applied a deletion-only Terraform plan,
then independently verified no campaign GPU remained at **13:22:00 UTC**.
A separate operator API query confirmed zero remaining instances. The local
launch flag was reset to false. [Destruction evidence](evidence/preflight_teardown_2026-11-07.json).

Allocation lifetime through verification: **1,391.54 seconds** (about 23.2 min).
The one-hour-minimum compute estimate is **$0.471**, before tax/transfer and small
AWS control-service usage. The provider invoice is authoritative. The alert's
$7.471 estimated campaign total includes the **$7 ancillary planning reserve**;
it is not a measured provider bill. No second GPU or different region was started.

The diagnostic archive was also restored from S3 after destruction and checked
against its immutable SHA-256 index. This preserves **observer diagnostics**;
no GPU benchmark journal was produced, so this is not a real-model VM-resume
acceptance claim. See the committed diagnostic-restore evidence.

A [support-request draft](atl_preflight_support_request.md) records the evidence
and asks about supported GPU image/network/console behavior and the 32 GB
plan's per-device topology. It has not been sent; contact with support requires
the owner's instruction. No further paid experiment should bypass this failed
access gate or claim CUDA/model/Gate 2 readiness.
