# Private experiment host AMI

This separate stack defines an EC2 Image Builder recipe for both experiment
roles. `enable_build=false` is the default; no builder GPU is launched until a
reviewed plan explicitly enables it and the regional GPU quota is sufficient.

Verified inputs (5 November 2026):

- Canonical Ubuntu 22.04 amd64 AMI `ami-033ea4ff4a9c726dc`, owner `099720109477`,
  release `ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-20261021`.
- Official AWS GRID 19.5 installer
  `grid-19.5/NVIDIA-Linux-x86_64-580.159.03-grid-aws.run` in
  `ec2-linux-nvidia-drivers`.
- Downloaded installer SHA256:
  `8ea16dc3c9034813a11bf52d8024f5466b48ab36cae6d91a7c95f8e77912e5dd`.

The recipe installs the host driver, Docker/Compose, NVIDIA Container Toolkit,
SSM prerequisites and CloudWatch agent, then records OS packages, kernel,
installer checksums and GPU inventory under `/usr/local/share/cloud-vlm-host`.
It never installs application model packages or weights on the host.
Both roles must use the same resulting private AMI revision after validation.
APT/toolkit/agent packages are resolved at image-build time and recorded; this
is not a byte-reproducible package build. Reuse the resulting immutable AMI,
and bump recipe_version for changes rather than silently rebuilding a campaign.

The temporary builder has public outbound connectivity for package installation,
no inbound access and a separate VPC. Experiment nodes remain private. Image
Builder terminates failed builders; the install step has a one-hour timeout and
starts a transient host shutdown timer capped at two hours of instance uptime.
This stop guard still needs a live test; it does not execute until the build step
starts, and it is not retained in the final AMI.
Tests require the specific driver and host tooling. The temporary instance type
is g6f.2xlarge, so it consumes 8 quota vCPUs while building. Do not overlap it
with the 16-vCPU experiment pair. Builder GPU cost is $0.475/hour, plus temporary
public IPv4/EBS and retained AMI snapshots. Report its reviewed window before use.

Initialize the separate locked S3 backend with the bootstrap bucket and region.
Validate/fmt/mock-test, save and review the plan, then apply. Check whether
AWSServiceRoleForImageBuilder already exists; if it does, import it instead of
attempting a duplicate creation. No host image has been built yet: quota is
pending, and the recipe's installation commands remain unverified on AWS.
The final app's CUDA/BF16/model tests on both nodes remain mandatory after baking.
