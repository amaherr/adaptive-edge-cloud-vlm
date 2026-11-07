# Draft Vultr support request — Atlanta A16 guest inaccessible

Prepared for the owner; **not submitted** by the agent. Do not include API keys,
SSM contents, default passwords, SSH private keys or raw Terraform state.

Subject: Atlanta A16 GPU VM reports active/running but SSH and web console fail

We are preparing a university research deployment and ran a bounded one-node
feasibility test through Terraform. We need to verify access before purchasing
extended GPU time.

- Region: Atlanta, USA (`atl`).
- Plan: `vcg-a16-6c-60g-16vram`, listed as API type `vdm`, $0.471/hour.
- OS: Ubuntu 24.04 LTS x64, OS ID `2284`.
- Instance ID: `70b5a674-f823-47a0-aa85-3dc32d6e5c60`.
- Created: **7 November 2026, 12:58:49 UTC**.
- Public IP at test time: `45.76.63.37`; VPC IP: `10.46.0.3`.
- Provider API reported status `active`, power `running`, server `ok`.
- Vultr firewall group allowed TCP 22 only from our operator IPv4 /32.
  Two independent HTTPS egress-IP checks matched that allowlist. The supplied
  cloud-init instructs the host firewall to allow SSH from that /32 and private
  experiment ports from the VPC; actual guest application could not be inspected.
- Repeated Linux SSH connection attempts timed out before authentication.
  A separate Windows TCP connection attempt to port 22 also timed out.
- Public GitHub SSH port 22 was reachable from the operator, so a general
  outbound-SSH block on the operator's connection was not observed.
- The owner opened View Console and received **“failed to connect to server”**.
  We could not inspect boot messages, guest firewall, SSH service or GPU driver.
- One provider API reboot was accepted with HTTP 200. SSH still timed out in
  subsequent probes; final local stop decision was at **13:18:30 UTC**.
- The independent watchdog observed the still-billable VM in a
  stopped power state at 13:16:10 UTC after the reboot. No cause is established.
- No CUDA or model inference was run. No driver/kernel/Docker reinstall was
  attempted. Independent Terraform teardown completed at **13:22:00 UTC**;
  both the hosted worker and a separate operator API check confirmed zero
  remaining campaign instances. All accessible diagnostics were archived.

Our subsequent configuration audit found that Terraform Base64-encoded
cloud-init even though provider 2.33.2 already does so. We have corrected that
double encoding, but have not retried a VM. The intended guest firewall setup
therefore cannot be assumed to have run. Linux and Windows probes also shared
the same operator network; an independent network has not yet been tested.
The reboot was not followed by verified return to running before later probes.
We have not established whether any of these explain the original failure.

Please investigate the host/console path and confirm whether the guest booted
and received working public networking. Is this Ubuntu OS ID the correct
GPU-enabled image for this A16 allocation, including compatible NVIDIA driver
and NVIDIA Container Toolkit? Is any account entitlement or GPU activation
step missing despite the API reporting the instance active?

We also need to confirm two simultaneous GPU VMs in Atlanta with private VPC
connectivity. For `vcg-a16-12c-120g-32vram`, does the guest see multiple 16 GB
GPUs or one 32 GB device, and does one CUDA process receive greater compute
than the 16 GB plan? We do not assume multi-device VRAM is pooled.

Please advise a reproducible supported image/plan configuration before another
paid attempt, and review billing for the inaccessible test allocation if
appropriate. We can provide private diagnostic artifacts through a secure
support channel after reviewing them.
