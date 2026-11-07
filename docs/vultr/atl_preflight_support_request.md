# Draft Vultr support request — two Atlanta A16 access failures

Prepared for the owner; **not submitted**. Never include API keys, default
passwords, SSH private keys, SSM values or raw Terraform state.

Subject: Atlanta A16: SSH and console inaccessible on two Ubuntu 24.04 VMs; start API returns HTTP 500

Hello Vultr Support,

We ran two short single-node research preflights in Atlanta (`atl`) using
`vcg-a16-6c-60g-16vram` and Ubuntu 24.04 LTS x64 (OS ID `2284`). Neither VM
was accessible over SSH or the web console. Please investigate the host/boot,
public network and console paths, and confirm a supported image configuration.

Latest reproduction on 7 November 2026, UTC:

- Instance `532bb8df-50fa-406a-8256-83d996151db3`, public IP `155.138.208.173`,
  created at **14:07:09**; VPC address `10.46.0.3`.
- API reported `active`, `running`, `server_status=ok` before SSH probes.
- Fresh SSH key supplied at creation. Minimal cloud-init made no package,
  driver or guest-firewall changes. The user-data API round trip matched the
  intended plain cloud-config after one Base64 decode.
- The assigned provider firewall permitted TCP 22 from the operator's exact
  IPv4 /32; two public-IP checks agreed. SSH timed out before authentication.
- Three independent AWS Lambda TCP-22 probes also timed out. Their observed
  egress IPv4 was consistently `34.227.22.59`, temporarily allowed as an exact
  /32 in that same firewall group. The temporary rule/function/role were removed.
- View Console still displayed **“failed to connect to server”** in a private
  browser window over the owner's personal hotspot.
- A documented API reboot at **14:13:55** returned HTTP 200, but eight polls
  through **14:15:21** reported `power_status=stopped`, while server status
  remained `ok`. One explicit start attempt returned **HTTP 500 Internal Server
  Error**. No guest logs or GPU inventory could be obtained.
- We ended the test and used our independent Terraform teardown controls;
  deletion verified at **14:20:15 UTC**, with a separate API check confirming
  zero remaining project GPU instances. All diagnostics were retained in S3.

Earlier instance `70b5a674-f823-47a0-aa85-3dc32d6e5c60`, IP `45.76.63.37`,
created at **12:58:49** and destruction verified at **13:22:00**, had similar
SSH/console failures. Our first configuration double-encoded cloud-init; we
corrected that before the latest reproduction above. Neither attempt ran any
model inference or modified vendor GPU tooling.

Please check why a running allocation had no reachable SSH or console, why
restart left it stopped, and why explicit start returned HTTP 500. Are there
host faults, image/boot issues, or account activation requirements? Is OS ID
2284 appropriate for this A16 plan with supported NVIDIA drivers/toolkit?
Please review billing for these inaccessible allocations if appropriate.

Separately, for future planning: does `vcg-a16-12c-120g-32vram` expose multiple
16 GB devices or one 32 GB device, and can one CUDA process receive greater
compute than on the 16 GB allocation? We do not assume pooled VRAM.

We can provide reviewed diagnostics through a secure support channel.

Latest evidence: [corrected preflight](preflight_retry_2026-11-07.md).
