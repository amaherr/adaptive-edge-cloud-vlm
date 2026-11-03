# Cloud migration implementation and evidence

Scope: roadmap architecture (phase 0), frozen baseline, Terraform (1A), and
container contract (1B). Campaign `cloud-vqa-v1`; environment
`aws-cloud-emulation-v1`. No cloud measurements have been collected.

## Repository analysis

The existing system already supports raw-image and projected-tensor TCP paths
through `adaptive-vlm adaptive-server`. Keep this shared server rather than
building a second inference implementation. Model precision remains BF16,
generation is deterministic with 32 tokens, and the tokenizer and remote model
code are frozen by the existing artifact contract. The model lock installs the
repository in editable mode; the image preserves that behavior and verifies all
locked distribution versions after installation.

The physical runner is tied to physical network evidence and local orchestration.
It must not simply be renamed and advertised as a cloud runner. Cloud scheduling,
per-attempt journal durability, S3 immutable commits, SIGTERM checkpoint handling,
and recovery remain subsequent work. The edge Compose service currently exposes
the CLI for diagnostics; it does not run the final campaign.

Retain all 100 original requests and images, 60/20/20 request assignments,
training blocks N1/N3/N5/N6/N8, validation N2/N7 and evaluation N4/N9. SLA is
60,000 ms. The full matrix is 8,100 attempts and 27 warmups. Cloud grid copies
only intended impairments: its measured baseline is null and interface is an
explicit unresolved value. Never substitute the old 1.63 ms baseline.

The edge hardware file describes advertised memory rather than measured usable
memory. Full-model weights (~3.85 GB) alone do not establish feasibility in a
quarter L4: activations, CUDA contexts, allocator reserves and KV cache also count.
A failed 6 GB preflight requires a documented environment revision, not silent
quantization or a different model. Independent VMs still have driver, hypervisor,
partition and contention confounders. Network emulation is not real 5G.

## Deployment sequence

1. Use Terraform 1.10.5. Bootstrap the protected state bucket using
   `infra/terraform/bootstrap`; provide `region` and `state_bucket_name`.
   Run init, fmt, validate, save/review plan, then apply that plan. Keep the local
   bootstrap state encrypted with restricted access and securely backed up;
   migrate it to a distinct backend key once the bucket exists. Never place its
   state under the experiment stack key.
2. Copy the experiment tfvars example to ignored `terraform.tfvars`. Supply
   region/AZ, globally unique bucket name, a **user-chosen positive budget** and
   confirmed email, reviewed AMI IDs, runtime/idle limits. Keep compute disabled.
   Initialize with `-backend-config=bucket=... -backend-config=region=...`.
   S3 backend uses native lockfiles, encryption and bucket versioning.
3. Plan/review/apply foundation resources first. Confirm budget email and SNS
   subscription. Seven interface endpoints incur charges even with no GPUs;
   include them, EBS, retained S3 versions and ECR in the cost forecast.
4. Build reviewed AMIs containing the role-appropriate supported NVIDIA driver
   (GRID on G6f), Docker/Compose, NVIDIA Container Toolkit, SSM, iproute2, iperf3.
   There is deliberately no public IP, NAT or internet route. Package installation
   and model downloads cannot run on these hosts without a separately reviewed
   path. Seed model/data EBS via S3/SSM. Cloud-init checks GPU/toolkit; it does not
   download unverified drivers. Freeze AMI/driver/OS versions as provenance.
5. Confirm same-AZ offerings, quotas and current prices; set launch_approved and
   enable_compute true only after the preceding acceptance checks. Review/apply
   a new plan. Export outputs to ignored `artifacts/infra/terraform_outputs.json`.
6. The encrypted data EBS volumes survive instance replacement and are protected
   against Terraform destroy. Identify by volume ID/NVMe serial, format only a
   confirmed EMPTY volume, mount persistently, and prepare directories owned by
   UID/GID 10001. No automatic formatting is attempted. Confirm mounts before
   starting Compose; otherwise paths can silently land on the root disk.
7. Build/push once and deploy by digest per the container runtime contract.
   Validate route/interface, GPU BF16, model hashes and all three inference paths
   on both nodes. Record image digest and deployment outputs together.

## Cost guard and remaining gates

Host watchdog checks uptime and `/var/lib/adaptive-vlm/job.heartbeat` every minute;
update heartbeat only while a controlled job is active (on BOTH nodes). It stops
`cloud-vlm.service` and powers down at runtime/idle ceiling. EC2 shutdown must
retain default stop behavior. Test with short limits in a disposable environment.
Current guard preserves EBS but cannot guarantee an S3 checkpoint: the checkpoint
manager is not yet implemented. Do not claim Gate R0. Budget notifications are
not an immediate spending cap. Reboot resets the per-boot runtime ceiling;
this is not a cumulative campaign spending limit.

CloudWatch currently provisions host status alarms and a retained log group;
log shipping and application checkpoint/GPU/storage metrics are pending the
runner/agent integration. Do not claim a working heartbeat alarm or idle metric
on the strength of a provisioned log group. No full campaign before these controls
and checkpoint crash/restore tests pass.

Terraform protects state, artifacts, ECR and data EBS with prevent_destroy.
A full destroy plan will therefore refuse destruction: teardown must explicitly
retain/remove protected resources from the disposable stack under reviewed state
migration, never delete thesis evidence merely to make destroy succeed. Verify
independent S3 restore first. Archive reviewed plans privately; plans/state may
contain sensitive values. Record emergency stops and reconcile state afterwards.

Gate C0 is locally hash-verified but still needs a committed deployment revision
and matching clean VM checkouts. T0 and D0 remain OPEN until live plans, alerts,
drift, watchdog, driver/GPU and container tests have actual evidence. The local
implementation is not evidence that AWS provisioning or GPU compatibility passed.

References checked during implementation:
- https://developer.hashicorp.com/terraform/language/backend/s3
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
- https://docs.docker.com/compose/how-tos/gpu-support/

## Local validation (3 November 2026)

- Existing Python suite: 189 passed, 1 skipped (sandbox loopback restriction).
- Frozen baseline verifier: all contract hashes, 100 image hashes and split pass.
- Cloud experiment passes the existing configuration loader.
- Terraform recursive formatting and experiment provider validation pass.
- Terraform mocked plans: foundation launches no GPU; unapproved launch rejected
  (2 passed). Mock inputs are test fixtures, not chosen deployment costs/region.
- Compose and cloud-init YAML parse; this does not substitute for Docker's schema
  validation, image build, service startup or GPU testing.
- Ruff passes for the new Python scripts. No AWS plan/apply or paid deployment
  performed; real prices, quota, region/AZ and notification confirmation pending.
- Docker build not run: Docker Desktop WSL integration is unavailable here.

CI repeats Terraform validation and mocked safety plans without AWS credentials.
No CI deployment authority is configured. Existing physical files remain intact.

Initial region recommendation: us-east-1, subject to current same-AZ offerings,
quota and price checks. AWS lists G6f availability in Northern Virginia:
https://aws.amazon.com/about-aws/whats-new/2025/07/amazon-ec2-g6f-instances-fractional-gpus/
The supplied notification email is stored only in ignored local tfvars;
monthly budget is now $20 (user-selected alert threshold).
Both bootstrap and experiment Terraform provider validation passed.

## Foundation cost correction

Interface endpoints are now disabled by default through
`enable_private_endpoints=false`. GPU launch requires them explicitly enabled.
They must be included in the reviewed cost estimate and removed after the test
window when no longer needed. Stopping EC2 does not stop endpoint billing.
The user clarified $20 is a monthly alert threshold; report deployment cost
before creating paid resources. AWS authentication currently fails with
InvalidClientTokenId and Docker Desktop WSL integration needs enabling.
