# Corrected single-node retry — access still blocked

Owner explicitly authorized this retry after the connection audit. It used the
existing $5 preflight ceiling and carried forward the earlier $0.471 compute
estimate: remaining retry ceiling $4.529, total project target $80. Fresh API
quote confirmed Atlanta A16 at $0.471/hour. No second concurrent GPU was used.

## Setup and observations

- Source fix: `cf9e7dc`; plain user-data, fresh SSH key, exact operator /32
  provider firewall. Minimal cloud-init changed no packages or guest firewall.
- Independent session `preflight-20261107T140644Z` was registered in S3 before
  apply, with an initial expiry of 14:24:44 UTC and 900-second idle protection.
  Preparation allowed a few minutes before the planned 15-minute diagnostic
  window. The hosted watcher adopted the allocation and reported no stop reasons
  while it was running. The first attempt's cost was retained in the ledger.
- Instance `532bb8df-50fa-406a-8256-83d996151db3`, IP `155.138.208.173`, created
  **14:07:09 UTC**, plan `vcg-a16-6c-60g-16vram`, Ubuntu 24.04 OS ID 2284.
- API user-data decoded once matched intended cloud-init exactly; SHA-256:
  `61e1b4e7af86e645ea6f5af69aed464dd248eb5963737f29b635423da2ae37e2`.
  This verifies delivery, not successful in-guest cloud-init execution.
- API showed active/running/ok and correct firewall assignment before verbose
  SSH probes. Operator SSH still timed out before authentication.
- Three TCP-22 probes from a temporary AWS Lambda also timed out. An exact /32
  rule allowed its consistently observed egress IPv4 `34.227.22.59`. HTTPS
  egress-IP agreement is useful evidence, not a packet capture proving the
  precise source of every TCP SYN. The Lambda had no AWS resource permissions
  and received no Vultr or SSH credentials. Its rule, function and IAM role
  were deleted; separate lookups verified cleanup and only the operator rule
  remained. No additional GPU was involved.
- The owner reproduced the console error in a private window over a personal
  hotspot. Neither the console nor SSH provided guest logs or a boot screen.
- Following [Vultr's console FAQ](https://docs.vultr.com/vultr-web-console-faq),
  one API reboot at **14:13:55 UTC** returned HTTP 200. All eight polls through
  **14:15:21** reported stopped/active/ok. One explicit start returned HTTP 500;
  the error response body was not retained. No SSH probe against this stopped
  guest was counted as evidence of failure of a running guest.
- The session was expired early after that failure; local deploy flag reset
  false. Watchdog reported CHECKPOINT_GRACE at **14:16:09 UTC**, for expiry and
  stopped-but-billable state. No control was disabled to accommodate the reboot.

## Result and retained evidence

**BLOCKED at guest access.** Zero model requests. CUDA/BF16, GPU identity,
Docker, model fit, correctness, profiling and Gate 2 remain NOT RUN. The
cloud-init encoding defect is corrected; it did not resolve access on this
allocation. The root cause remains unknown. The updated
[support draft](atl_preflight_support_request.md) has not been submitted.

Raw observer evidence is in `artifacts/vultr/preflight-002/`. Before teardown,
20 diagnostics were archived under immutable SHA-256-suffixed keys in existing
S3 at `research/adaptive-vlm/vultr-v1/control/preflights/preflight-20261107T140644Z/`
and read back/restored byte for byte. These are diagnostics, not a GPU benchmark
journal or a real-model resume acceptance result.

The independent CodeBuild worker
`adaptive-vlm-vultr-teardown:a2da548b-59d3-4bdc-a33c-cb8fb2ca8df6` succeeded.
Deletion-only Terraform teardown verified zero remaining GPUs at
**14:20:15.007540 UTC**. A separate operator API query confirmed zero project
GPU instances. Local deploy flag is false; the final Terraform plan reports
**No changes** against empty GPU state. Temporary SSH private/public key and
known-host files were removed after the independent zero-inventory check.

After destruction, **28 diagnostic files (22,540 bytes)** were restored into a
fresh directory and verified byte for byte and by SHA-256. The final evidence
summary, no-change plan and restore index were separately persisted in retained
S3 and read back. [Committed outcome evidence](evidence/preflight_retry_2026-11-07.json).

Estimated minimum compute for both attempts combined is **$0.942**
(2 × $0.471), before tax/transfer and small AWS control/probe costs. The retry
lifetime through deletion verification was about 13.1 minutes. Provider invoice
remains authoritative; the alert total includes the separate $7 ancillary
planning reserve. No paid GPU remains. Independent controls and retained
S3/ECR/state remain available; the support draft has not been sent.
