# Fully Cloud-Hosted Edge–Cloud VLM Research: Migration and Gate 2 Roadmap

**Project:** [`amaherr/adaptive-edge-cloud-vlm`](https://github.com/amaherr/adaptive-edge-cloud-vlm)  
**Version:** cloud-emulation-v2 / Terraform + Docker + resilient checkpointing, 3 November 2026  
**Decision:** Conduct all remaining bachelor-thesis implementation, profiling, calibration, and evaluation using **Terraform-provisioned** cloud infrastructure, **Docker-packaged** model services, and **S3-backed resumable experiments**. Physical devices are **not a dependency**; physical-device validation can be a separately reported future experiment.

> **Gate 2 counting clarification:** The committed protocol describes **100 VQA requests × 9 network blocks × 3 placements × 3 repetitions = 8,100 measured attempts** (plus 27 warm-ups), not 1,800. The recovery design also supports a smaller 1,800-attempt subset, but that would be a separate explicitly declared campaign and would **not** complete the current full Gate 2.

> **Research terminology:** This setup is *real distributed inference on two independently provisioned cloud VMs with a resource-constrained virtual edge and controlled network emulation*. It is **not** a physical Jetson/smartphone deployment, and a simulated 5G profile is not a real 5G radio link. These distinctions must remain explicit in the paper.

## 0. Recommended architecture and why

### Primary recommendation: two distinct AWS EC2 machines

| Role | Proposed type | GPU allocation | Advertised GPU memory | vCPU / RAM | Reason |
|---|---|---|---:|---|---|
| **Virtual edge** | `g6f.2xlarge` | 1/4 NVIDIA L4, fractional vGPU | 6 GB | 8 vCPU, 32 GiB | Real restricted GPU compute and VRAM allocation; large enough to *plausibly* run current BF16 Moondream2, subject to verification. |
| **Virtual cloud** | `g6.2xlarge` | Full NVIDIA L4 | 24 GB | 8 vCPU, 32 GiB | Matched advertised CPU/RAM, substantially more GPU capacity. |

Both are instances in **one AWS region, one Availability Zone, and one VPC**; communicate over **private IPs**. Use Ubuntu 24.04 only after verifying the official GRID/CUDA driver requirements. At the host level, keep the exact existing Python 3.11, pinned model revision, BF16, tokenizer, deterministic decoding, and model-cu128 lock.

AWS G6f has fractional L4 vGPU. `g6f.2xlarge` has a **fraction** of a full L4 with ~5.59 GiB reported GPU memory in detailed EC2 specs, not a full 24-GB device. G6f supports **GRID driver**, not standard Tesla driver; check current AWS documentation and prove PyTorch CUDA works before declaring this setup usable. Its CPU/RAM matches `g6.2xlarge`, but hypervisor, driver, GPU partition and underlying contention are still potential confounders. Freeze machine types and record them.

**Minimum acceptance preflight:** `nvidia-smi`, CUDA BF16 matrix operation, model load, single-request all-edge GPU inference, and vision-only GPU inference must succeed on G6f. The 6-GB tier is a recommendation **not a measured guarantee**. If it fails, test `g6f.4xlarge` (1/2 L4, 12 GB GPU memory), explicitly recording the change as a *new environment revision*. Do not silently quantize, change precision, or swap models just to make a result pass.

**What to avoid:** a single powerful GPU VM with two containers claiming to reproduce distinct edge/cloud devices, or using `torch.cuda.set_per_process_memory_fraction` as evidence of a physically slower GPU. Software restrictions may help diagnostics, but do not reproduce device silicon or memory bandwidth.

### Architecture

```text
                  Experiment orchestrator (runs ON virtual edge)
                  frozen VQA manifest, policy schedule, network blocks
                                         |
                 +-----------------------+-----------------------+
                 |                                               |
                 v                                               v
     EC2 g6f.2xlarge — EDGE                           EC2 g6.2xlarge — CLOUD
     fractional L4 (1/4), ~6GB                          full L4, 24GB
     pinned Moondream2 adapter                          pinned Moondream2 adapter
     all-edge or vision/projector                       raw-image or projected-tensor server
                 |                                               ^
                 +--- private VPC/TCP, tc/netem both directions -+
                         measure with iperf3 + TCP echo

  Terraform (bootstrap + experiment stack) provisions BOTH VMs, networking,
  IAM, encrypted S3 checkpoint/artifact storage, ECR, logs, and budgets.
  One immutable Docker image digest runs separate edge/cloud roles on the VMs.
  Per-attempt local durable JSONL -> periodic immutable S3 snapshots/commits.
  Checkpoint recovery resumes the original frozen campaign after a crash.

 All-edge:      image + prompt -> edge full model -> output (no request inference link)
 All-cloud:     image + prompt -> cloud full model -> output
 Distributed:   edge vision + projector -> BF16 tensor -> cloud language model -> output
```

### AWS setup choices

- Use **On-Demand** instances for the first final benchmark; Spot can be added only after resume/recovery behavior is confirmed and interruptions are logged.
- Choose the region based on **both instance types being available in the same AZ**, quota and verified current prices; AWS lists G6f availability in US/EU regions. Proximity to the student's laptop is irrelevant to inference timing because the benchmark must run entirely between the VMs over private IPs.
- Provision **all** AWS resources using version-controlled Terraform. Create a separate bootstrap state for the Terraform S3 backend; do not manually create VMs, buckets, policies, alarms, security groups, or ECR repositories. Create an IAM role permitting only necessary S3 artifact access and SSM access. Apply security groups so cloud TCP `5050` (inference), `5051` (iperf3) and `5052` (TCP echo) are allowed **only from the edge security group/private IP**. Never expose the repository's unauthenticated, unencrypted TCP protocol to the public Internet.
- Use AWS Systems Manager Session Manager (with appropriate agent/role/network connectivity), or a tightly restricted SSH setup for administration. Management traffic is **not** the measured edge–cloud inference link.
- Give each VM adequate persistent EBS storage; pin the EBS volume ID/size/type. Run the same versioned Docker application image on both VMs, with host-side NVIDIA drivers and NVIDIA Container Toolkit. Persist each attempt locally and checkpoint **during each block** to private, versioned S3; do not rely only on after-block uploads. Stop instances when not in use. Confirm launch quotas and budgets before starting.
- For the first working setup, shaping the primary egress interface is acceptable after the autonomous jobs and out-of-band access are established. For best isolation, later attach dedicated experimental interfaces and use appropriate policy routes; do not silently assume that adding a second interface automatically routes application packets through it.

## 1. Freeze repository baseline and create a new cloud campaign

Inspect and retain the following existing artifacts (do **not** overwrite their physical-run meanings):

- `configs/experiments/physical_vqa_v1.yaml`
- `configs/profiles/physical_network_blocks_v1.json`
- `configs/evaluation/physical_vqa_split_v1.json`
- `configs/model/moondream2.yaml`, `configs/model/moondream2_artifact.yaml`
- `requirements/model-cu128.lock`
- `data/manifests/physical_vqa_v1.jsonl`
- `scripts/run_physical_profile_block.py`
- `scripts/tcp_rtt_probe.py`
- `docs/physical_profile_campaign_v1.md`, `docs/phase2_profile_schema.md`

Create a new Git branch, for example `experiment/cloud-emulation-v1`. Keep source history and past physical evidence unchanged. Suggested new files:

```text
configs/hardware/edge_cloud_g6f_2xlarge_v1.yaml
configs/hardware/cloud_g6_2xlarge_v1.yaml
configs/experiments/cloud_vqa_v1.yaml
configs/profiles/cloud_network_blocks_v1.json
configs/evaluation/cloud_vqa_split_v1.json
scripts/run_cloud_profile_block.py
scripts/apply_cloud_network_block.py
scripts/validate_cloud_block.py
scripts/aggregate_cloud_gate2.py
scripts/checkpoint_campaign.py
scripts/restore_campaign.py
scripts/test_checkpoint_recovery.py
container/Dockerfile
container/compose.edge.yaml
container/compose.cloud.yaml
container/.dockerignore
infra/terraform/bootstrap/
infra/terraform/environments/cloud-v1/
infra/terraform/modules/{network,compute,security,storage,observability,budgets}/
infra/cloud-init/{edge,cloud}.yaml
docs/checkpoint_protocol_v1.md
docs/container_runtime_contract_v1.md
docs/cloud_experiment_protocol_v1.md
docs/cloud_environment_handoff_v1.md
artifacts/phase2/cloud-v1/N1/ ... N9/      # generated, ignored, S3-archived
```

The cloud-specific files should **copy/refer to** the same frozen model, original image/request manifest and 32-token generation settings rather than regenerate them. The existing split is already 60 training, 20 validation and 20 final evaluation requests, and reserves N4/N9 as held-out network blocks. Either preserve that plan *unchanged* as the prospective cloud plan or declare and freeze a new independent split **before profiling**. Do not use final evaluation requests/conditions to choose your predictor.

### Gate C0 — Frozen provenance

- [ ] New branch and commit; clean Git status on both VMs.
- [ ] Identical model artifact hash, tokenizer, Python and dependency lock.
- [ ] All 100 source images and question IDs hash-verified; original dataset not resampled.
- [ ] New cloud campaign ID and explicit environment ID, never reuse `physical-v1` labels.
- [ ] Frozen prospective train/validation/test assignments and SLA.
- [ ] `cloud-v2` infrastructure contract requires Terraform, one Docker image digest, and S3 checkpoint commit/restore tests.

## 1A. Terraform-only Infrastructure as Code: bootstrap, deploy and destroy

**Hard rule:** No EC2 console clicks for creating or editing cloud resources. Provision, update, review, and eventually decommission them through **Terraform**. The only permitted manual actions are initial account/security administration (e.g., establishing AWS SSO/OIDC credentials and verifying an email subscription), troubleshooting reads, or emergency stops that are recorded and reconciled back to code. Never use Terraform to store Hugging Face tokens, AWS keys, or other secrets in plain text or committed `tfvars`.

### Repository structure and ownership

```text
infra/terraform/
  bootstrap/                # Separate state; remote-state bucket and its protection
    versions.tf providers.tf main.tf outputs.tf README.md
  environments/cloud-v1/    # Primary AWS experiment stack; remote S3 state
    backend.tf versions.tf variables.tf main.tf outputs.tf
    terraform.tfvars.example
  modules/
    network/                # VPC, subnets, routes, VPC endpoints, experiment path
    security/               # SGs, instance IAM/SSM, least-privilege bucket access
    compute/                # two pinned EC2 types, EBS, user_data/cloud-init
    storage/                # private versioned S3 artifacts, ECR image repository
    observability/          # CloudWatch logs/alerts, health and stop watchdog
    budgets/                # actual/forecast AWS Budgets alerts
infra/cloud-init/
  edge.yaml                 # host driver/toolkit/Docker/agent, not model weights
  cloud.yaml
```

- **Bootstrap first**: use a minimal separately tracked Terraform configuration to create a dedicated encrypted/versioned Terraform-state S3 bucket, bucket-public-access block, and lifecycle safeguards. Bootstrap can initially use a protected local Terraform state, then migrate it to a distinct protected backend; never accidentally destroy the state bucket through the experiment stack. Keep the state bucket **separate** from experimental data.
- **Primary state**: S3 backend with `use_lockfile = true` (Terraform 1.10+), bucket versioning and encryption. Never commit `.tfstate`, `.terraform/`, backend credentials, private key files, or plan files containing secrets. Pin the Terraform CLI and `hashicorp/aws` provider versions with `.terraform.lock.hcl`. <https://developer.hashicorp.com/terraform/language/backend/s3>
- **Compute**: explicitly fixed `g6f.2xlarge` edge and `g6.2xlarge` cloud types (or approved revision), **same AZ**, separate instance profiles, stable EBS data volumes and private addresses. Set IMDSv2 required. Attach descriptive experiment tags, e.g. `Project=adaptive-edge-cloud-vlm`, `Campaign=cloud-v1`, `Role=edge|cloud`, `ManagedBy=Terraform`.
- **Network**: private VPC security-group-to-security-group ingress; separate an SSM/administration path from the **measured inter-node experimental path**. Prefer a dedicated experimental ENI plus explicit policy routing **only if** routing is validated; otherwise use `tc` traffic classification restricted to inference/probe peer traffic so S3 uploads and management are not unintentionally throttled. Do not silently assume an ENI is on the application data path.
- **Storage**: one S3 bucket for immutable checkpoints/raw campaign outputs, a distinct protected Terraform-state bucket, an ECR repository for application images, persistent EBS for model cache and working logs. Enforce S3 block-public-access, encryption, versioning, minimum IAM rights, and retention that outlives thesis review. Create an S3 gateway endpoint in the same region/VPC where possible; it avoids NAT/internet traffic for S3 without an endpoint hourly fee. <https://docs.aws.amazon.com/vpc/latest/privatelink/vpc-endpoints-s3.html>
- **Access**: IAM instance roles, **never** long-lived access keys in `.env` or image layers; SSM Session Manager where supported. Make image-pull permissions separate from write access to experiment checkpoints. Use OIDC for CI deployment if enabled, not repository-wide AWS secrets.
- **Budget BEFORE GPUs**: Terraform creates one AWS Budgets monthly cost budget with early actual and forecast notifications (e.g. 50%, 80%, 100%). Supply a **user-chosen** `monthly_budget_usd` and notification email via variables (not guessed). Also add EC2-running-hour watchdog/local job limit and alerts on idle instances; **Budgets alerts are delayed, not an immediate hard spending cap**. <https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/budgets_budget> and <https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-best-practices.html>
- **Audit/teardown**: `terraform fmt -check`, `terraform validate`, security checks, `terraform plan -out` and review **before every apply**. Save deployment outputs (instance IDs, AZ, network IDs, ECR digest, S3 bucket ARNs) as provenance without disclosing credentials. Use `terraform destroy` on experiment resources only after artifact/checkpoint restoration is verified. Protect state and thesis artifacts from `destroy`.

Illustrative workflow (directory names are proposed; **do not run until the Terraform files exist**):

```bash
cd infra/terraform/bootstrap
terraform init && terraform fmt -check -recursive && terraform validate
terraform plan -out bootstrap.tfplan
terraform apply bootstrap.tfplan

cd ../environments/cloud-v1
terraform init
terraform fmt -check -recursive && terraform validate
terraform plan -out cloud-v1.tfplan
# Review region, AZ, instance prices/quotas, security rules, roles,
# storage, Budgets, ECR, and all resource creations.
terraform apply cloud-v1.tfplan
mkdir -p ../../../../artifacts/infra
terraform output -json > ../../../../artifacts/infra/terraform_outputs.json
```

**Gate T0 — Terraform reproducibility (must pass before C1):**

- [ ] No manually created experiment infrastructure; remote state works and is locked/versioned.
- [ ] `fmt`, `validate`, security review, and plan pass; reapplying shows **no unintended drift**.
- [ ] Budget actual/forecast alerts are configured with confirmed destinations *before* GPU start; running-hours watchdog and stop procedure tested.
- [ ] Both nodes, IAM roles, buckets, ECR, endpoints, routes and protected security groups exist via Terraform; no public inference ports.
- [ ] Tear-down dry run or reviewed destroy plan proves protected state and retained thesis artifacts are not deleted.

## 1B. Docker packaging and GPU runtime contract

**Hard rule:** use a **single versioned application image** with different entrypoint commands/Compose services for edge execution, cloud service and diagnostic tools. Do not separately `pip install` application packages on each VM after provisioning. Host NVIDIA GRID/driver, kernel, NVIDIA Container Toolkit, EBS, `tc`, `iproute2` and management agent remain host responsibilities; the application environment and pinned Python packages live in the container.

Recommended build/deploy pipeline:

1. Create `container/Dockerfile` and `.dockerignore` with a **digest-pinned** CUDA/PyTorch-compatible Linux base. Install Python 3.11 and the existing `requirements/model-cu128.lock` *inside the image*. Copy the committed application and `pip install` its pinned package. The exact base and CUDA build must be verified on **both** L4/G6f GRID drivers; do not assume a sample CUDA base has Python 3.11 or matches the pinned wheel index. Fail the build on lock mismatch.
2. Embed `GIT_SHA`, model-contract SHA, lock SHA and image build metadata as nonsecret OCI labels. Build once in CI or a controlled builder; publish to Terraform-created **ECR**, tag with the commit for human reference, and **deploy by immutable image digest**, not `latest`. Prefer reproducible CPU-only tests on every PR; do real GPU smoke tests on the two provisioned nodes.
3. Configure **host** NVIDIA driver first (GRID for G6f) and NVIDIA Container Toolkit. Run `sudo nvidia-ctk runtime configure --runtime=docker`, restart Docker and verify a GPU test container with `docker run --rm --gpus all ... nvidia-smi`. This may require verified driver/image combinations; do not execute arbitrary example image tags in the final campaign. <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>
4. Use **Linux host networking** for the measured containers (`network_mode: host`) so application traffic follows the tested VPC host path; this must be audited and held constant. The cloud stack runs `adaptive-server`, `iperf3` and TCP echo as isolated services, with just the approved private ports. The edge stack runs the benchmark orchestrator and local model role. Docker should use `gpus: all` / Docker `--gpus all` with the NVIDIA runtime; do not mount the Docker socket or grant full `--privileged` or `NET_ADMIN` to inference containers.
5. Operate traffic shaping on the **host** with tightly restricted privileged automation (e.g. a root-owned, allowlisted `tc` helper invoked by SSM/systemd); the model containers must remain unprivileged. Check the exact host interface after applying a qdisc and measure actual rates **from inside** the deployed containers.
6. Mount persistent model/cache and output volumes (`/models`, `/data`, `/artifacts`) rather than placing multi-GB weights or live results inside image layers. Freeze the input manifest/model hashes. Configure container health checks, restart policy for long-lived services, a clean SIGTERM handler for experiments, and nonblocking host/service logs to CloudWatch or local persistent volumes.
7. Run the same GPU/correctness smoke with **both** host-local and container runtime configurations; require equivalent outputs/metrics within a predeclared tolerance before switching official benchmark runs to containers. Measure Docker orchestration overhead separately; record container/image digest per trial. Do not mix physical/host-native and container timings in one comparison.

Illustrative commands (proposed files, deployment digest and names must be substituted):

```bash
# On either EC2 host, after Terraform/cloud-init installs drivers and Docker:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -i runtime

# Pull the *same* image digest on both hosts via role-backed ECR login.
# On cloud: expose the GPU and start the inference service privately.
docker compose -f container/compose.cloud.yaml up -d

# On edge: run a frozen one-request smoke before the full campaign.
docker compose -f container/compose.edge.yaml run --rm benchmark --help
```

**Gate D0 — Container equivalence (must pass before C3/C5):**

- [ ] Same immutable image digest, model artifact and dependency lock on both machines; correct G6f GRID GPU exposure inside Docker.
- [ ] All three actual policy paths work from containers and preserve their prior correctness checks.
- [ ] Data/cache/artifact volumes survive container recreation; no model weights or AWS credentials embedded in image.
- [ ] `tc` acts on the proven container-to-container inference path, not the wrong Docker bridge; probe traffic and API traffic show verified equal network conditions.
- [ ] Inference containers do not receive `--privileged`, `NET_ADMIN`, AWS admin rights or public inbound ports.


## 2. Provision and verify two genuinely independent GPU nodes

Via the reviewed Terraform apply (**not through the AWS UI**), provision `g6f.2xlarge` and `g6.2xlarge` in the **same AZ**; maintain separate VM instance IDs and private IP addresses. Avoid an autoscaling fleet, serverless GPU endpoint, or shared notebook runtime, because instance substitution changes performance unexpectedly.

**G6f-specific caution:** AWS says G6f requires NVIDIA GRID driver; ordinary GPU PyTorch DLAMIs may list G6 but *not* G6f. Do not assume a default G6 DLAMI works on G6f. Follow the official G6f GRID 18.4–19.5 support window at the time of setup; use the supported AMI/driver and validate the CUDA runtime. The cloud G6 has different supported driver options. Record both drivers.

On **both hosts**, after **Terraform/cloud-init bootstraps the supported NVIDIA host driver and Docker**, collect OS/GPU inventory **on the host**, then verify the model dependencies **inside the identical Docker image**:

```bash
# Host inventory:
hostnamectl
uname -a
lscpu
free -h
nvidia-smi

# GPU check in the app container; replace image placeholder with a real ECR digest:
docker run --rm --gpus all --entrypoint python IMAGE_BY_DIGEST -c \
  'import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); a=torch.randn(256,256,device="cuda",dtype=torch.bfloat16); print((a@a).shape)'
```

Use the existing **pinned `requirements/model-cu128.lock`** as the dependency source for the image. Do not accept a convenient newer PyTorch, different precision or unpinned tokenizer without a new experiment revision. Host-native Python environments are **not** the official benchmark runtime.

Run the existing Moondream2 smoke inference on each host using the appropriate `--vision-device cuda:0`, `--text-device cuda:0` placement. On the edge, verify full and isolated vision-only execution **separately**, because the repository's resident full-model runner may record memory peaks different from a vision-only component. Capture allocated/reserved CUDA memory and visible device memory.

### Gate C1 — Hardware/model feasibility

- [ ] CUDA and BF16 work on both nodes using the pinned model environment.
- [ ] Both hosts load the verified model artifact.
- [ ] Edge full-model smoke succeeds *or* the failure is documented and a new predeclared edge tier is chosen before the campaign.
- [ ] Peak memory, device free memory, driver, GPU model, vGPU allocation, CPU/RAM, instance ID, AZ and software versions are archived.
- [ ] Host-restart and **container-recreate** smoke confirmed; persistent model cache and experiment workspace survive.

## 3. Use private inter-VM networking and validate actual traffic path

1. Use private IPs for inference and probe traffic. Run `ip route get <PRIVATE_CLOUD_IP>` on edge and reverse route lookup on cloud. Record actual interfaces.
2. Start the dual-path Moondream2 `adaptive-server` **inside the cloud Docker service** on the cloud GPU, using a **cloud-specific** experiment YAML with `mode: cloud`. For example (illustrative; substitute your verified config and any required CLI binding flags):

```bash
.venv-model/bin/adaptive-vlm adaptive-server \
  --config configs/experiments/cloud_vqa_v1.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --device cuda:0 --bind-host 0.0.0.0
```

The cloud service must be private to the VPC, and must load the pinned model only once during steady-state profiling. Run the service through Docker Compose with a declared restart policy, controlled by `systemd` on the host; never rely on an active laptop terminal remaining connected. The shown `adaptive-vlm` CLI is the **process inside the Compose service**, not an instruction to reinstall packages on the host.

3. Start independent network probes **inside isolated cloud Docker services** using host networking (the commands below are the processes configured in those services):

```bash
iperf3 -s -p 5051
.venv-model/bin/python scripts/tcp_rtt_probe.py server \
  --bind-host 0.0.0.0 --port 5052
```

4. From the edge **benchmark/probe Docker container**, verify both directions and TCP RTT against the private cloud IP:

```bash
iperf3 -c <PRIVATE_CLOUD_IP> -p 5051 -t 8 -O 1 -J
iperf3 -c <PRIVATE_CLOUD_IP> -p 5051 -t 8 -O 1 -J -R
.venv-model/bin/python scripts/tcp_rtt_probe.py client \
  --host <PRIVATE_CLOUD_IP> --port 5052 --count 30 --warmup 3
```

Baseline intra-AZ RTT is **measured**, not assumed to be 0 ms or equal to your old 1.635-ms LAN/WSL baseline. Record timestamps, receiver goodput, RTT percentiles, retransmits, packet errors, private addresses, and full command outputs.

### Gate C2 — Connectivity and independence

- [ ] No inference/probe ports exposed publicly.
- [ ] Two distinct instance IDs and VMs, private peer-to-peer TCP route confirmed.
- [ ] Cloud server accepts `cloud/raw_image` and `distributed/projected/synchronous` without restarts.
- [ ] Model fingerprint and protocol checks succeed.
- [ ] Independent bidirectional throughput and RTT baselines captured and repeated **from the actual Docker traffic path**.

## 4. Update the existing runner, not the complete system

Your current `scripts/run_physical_profile_block.py` already performs useful work: frozen manifest check, clean commit check, nine-block lookup, network pre/post `iperf3` + TCP echo, seeded rotation of `edge/cloud/distributed`, `benchmark` for each policy, profile conversion, request-coverage validation, and artifact generation. **Copy and generalize this script** rather than rebuilding the experiment framework. Invoke the generalized script **inside the versioned edge Docker image** with writable persistent artifact mounts, and keep host-level `tc` orchestration separately privileged.

### Required changes to the runner

1. Replace hard-coded `host_pair_id = edge-rtx3050-cloud-rtx2070super-v1` with a required, immutable `cloud-g6f25-cloud-g6-v1` environment identifier from configuration.
2. Replace `source = physical_wsl_netem_measured` with `cloud_vpc_netem_measured`.
3. Replace `cloud_host = 192.168.8.74` in a *new* grid with the current **private cloud IP**, or make the host a required command-line/config input; no silent DNS public-IP fallback.
4. Replace the hard-coded edge interface `eth0` with a verified interface from `ip route get`. Preserve exact queue-state logging.
5. Replace the original `unshaped_echo_median_ms` with a **fresh measured intra-AZ baseline**. Note RTT = baseline RTT + egress-added edge delay + egress-added cloud delay; do not directly call configured per-interface delay the observed RTT.
6. Require cloud-side evidence that the matching `tc` queue was applied, not only the edge queue. Save queue dumps on **both** VMs before and after every block.
7. Keep the current `benchmark --mode edge`, `cloud`, `distributed` CLI and 1 warmup / 3 measured repetitions, `cuda:0` as applicable, and `--resume` semantics. Retain strict dataset/model fingerprints.
8. Record full hardware inventory, Docker image digest/runtime details and GPU memory readings on both VMs. Record impossible/missing metrics as *missing*, not guessed. Add cloud GPU utilization/VRAM telemetry independently from the edge monitor.
9. Make resume repeat live network preflight if network block may have changed; do not treat a previous `network_pre.json` as proof that the current qdisc is still applied.
10. Never overwrite or mutate an already-frozen `plan.json` or previous raw trial rows. If the config/model/host/AZ/grid changes, create a fresh campaign ID and output root.
11. Track cloud VM status/out-of-disk errors; ensure post-measurement and **S3 checkpoint commit** occur after each interval and on normal shutdown, and attempt emergency upload after failed blocks without falsely marking them complete.
12. Keep failed and OOM trials in the 8,100-attempt denominator and mark which placements were infeasible rather than silently dropping them.
13. Add durable checkpoint keys to each request/repetition record, and make `--resume` reconstruct the original deterministic plan exactly from checkpoint state; never resample or quietly reset the seed.
14. Require image digest and infrastructure revision matching as additional resume guards (not only the Git/model/dataset hash).

**One important current limitation:** the repository implements **three placement choices** (`edge`, `cloud`, `distributed`) — not arbitrary model-internal transformer-layer split points. The 8,100-attempt matrix therefore tests *all currently implemented placement policies*, not every possible internal cut. Dynamic internal splitting is a separate proposed research feature and requires implementation and its own matrix.

### Gate C3 — Compatible cloud runner

- [ ] Existing regression tests still pass.
- [ ] Runner produces a valid cloud-specific immutable plan.
- [ ] Fresh cloud probes and edge/cloud qdisc evidence are captured per block.
- [ ] Both GPU inventories and peak-memory metrics are recorded.
- [ ] Killed/failed runs can resume without duplicate or omitted request/repetition keys; **restore from S3 on a fresh container/VM passes Gate R0**.

## 5. Freeze the controlled network matrix

Keep your repository's existing nine impairment profiles as the initial cloud **controlled constant-condition campaign**:

| Block | Egress rate (each host) | Delay added per outgoing host | Nominal added RTT | Declared usage |
|---|---:|---:|---:|---|
| N1 | 2 Mbps | 10 ms | 20 ms | Training |
| N2 | 5 Mbps | 10 ms | 20 ms | Validation |
| N3 | 10 Mbps | 10 ms | 20 ms | Training |
| N4 | 25 Mbps | 10 ms | 20 ms | Final evaluation |
| N5 | 50 Mbps | 10 ms | 20 ms | Training |
| N6 | 100 Mbps | 10 ms | 20 ms | Training |
| N7 | 10 Mbps | 2.5 ms | 5 ms | Validation |
| N8 | 10 Mbps | 25 ms | 50 ms | Training |
| N9 | 10 Mbps | 50 ms | 100 ms | Final evaluation |

The nominal added RTT is **not** total measured RTT. The selected VM type/AZ must achieve the highest throughput target *unshaped*; if 100 Mbps cannot be validated reliably, change the frozen cloud grid **before** collecting trial results, then retain all evidence of the change.

For an initial cloud experiment, an ordinary netem command on each **verified experimental egress interface** is:

```bash
# N1 example: apply on BOTH edge and cloud independently
sudo tc qdisc replace dev <EXPERIMENT_INTERFACE> root netem \
  rate 2mbit delay 10ms

# Verify exact qdisc and then run independent probes
sudo tc -s qdisc show dev <EXPERIMENT_INTERFACE>

# Cleanup on BOTH machines after the block/campaign
sudo tc qdisc del dev <EXPERIMENT_INTERFACE> root
```

Take care: the root qdisc shapes **all egress traffic through that interface**, including management and potentially S3 checkpoints if you use the same NIC. Prefer a dedicated verified experiment ENI or filtered impairment rules matching only edge↔cloud inference and probe traffic. **Checkpoint uploads and management must remain reachable through the unshaped control path**. Validate route selection, qdisc class/filter counters and achieved traffic conditions; never perform blind remote network reconfiguration that could cut off your only administrative path.

**Advanced realism campaign after Gate 2:** use time-varying measured cellular uplink traces, jitter, correlated/bursty loss, link dropouts, and competing traffic. Capture actual achieved conditions and keep the trace-controller independent from the adaptive scheduler. This adds ecological realism but does not recreate cellular radio protocols, mobility handovers, or edge power characteristics.

### Gate C4 — Network validation

- [ ] All nine blocks verified with independently measured bidirectional receiver goodput and TCP echo RTT.
- [ ] Original block acceptance (80–120% of target rate; median RTT within declared tolerance) re-evaluated for cloud baseline.
- [ ] Unshaped baseline and shaped pre/post measurements stored; no hidden deviations.
- [ ] Two qdisc snapshots prove impairment was applied at the measured time **and that checkpoint/SSM traffic is unaffected by experiment shaping**.
- [ ] Constant blocks are not mislabeled as continuously dynamic cellular traces.

## 6. Verify functional correctness on NEW hardware before timing

Run the frozen 12-request correctness manifest across both **Docker-hosted** cloud VMs for `edge`, `cloud`, and `distributed`. Compare model/prompt contracts, output IDs and decoded text, framing and transmitted byte checksums.

- If all 12 triples match exactly, retain the result.
- If GPU kernel/driver differences cause generation divergence, **do not silently loosen** an existing strict equality gate. Investigate numerical equivalence and transport correctness; predeclare an alternative scientific quality criterion in a new version and separately report strict cross-placement mismatch rates.
- Test the real 100 VQA items in a smoke subset before the full 8,100-attempt campaign.
- Assess data/model privacy: image bytes and intermediate features travel only across the VPC security-group-restricted link.

### Gate C5 — All placements work

- [ ] 12-request full correctness audit completed on the cloud host pair.
- [ ] Full-model cloud, edge full and edge vision-only GPU execution tested.
- [ ] 100-request real VQA smoke coverage verified; no missing raw images.
- [ ] Every mismatch and resource limitation documented before profiling.

## 6A. Required crash-safe checkpoint design: local journal + immutable S3 commits

The existing benchmark already creates `plan.json`, `run_manifest.json`, an **fsynced** `requests.jsonl`, `summary.json`, and a `--resume` path. **Preserve these existing contracts**. Extend the orchestrator with durable S3 checkpointing; do not replace working per-attempt accounting with a periodic high-level summary. The core requirement is **restart after a failed/terminated GPU VM without losing an entire network block**.

### What is checkpointed and when

**Local journal:** after each trial (success, timeout, error, or OOM), write one unique result to `requests.jsonl`, flush and `fsync` it; maintain an append-only per-trial sequence and stable trial key. This is local durability but is **not** enough when an EBS volume is lost or deleted.

**Remote checkpoint frequency:** upload an immutable **incremental** checkpoint at whichever happens first: **25 newly completed attempts or five wall-clock minutes**, plus after each policy, at the end of each network block, and upon orderly `SIGTERM`/shutdown. The five-minute interval is a *target recovery-point objective*, not a guarantee if an upload fails or is still pending; record the last committed checkpoint time and backlog. Keep model inference independent from S3 upload, and cap upload concurrency. On failure to persist, visibly stop/hold further work after a bounded grace period rather than silently accumulate unlimited unprotected work.

**Important:** network blocks can last hours. Do **not** wait until N1 finishes before backing up. Checkpoint inside each block and retain enough prior immutable checkpoints for postmortem analysis.

**S3 key design** (the bucket name is produced by Terraform; **not hard-coded**):

```text
s3://<terraform-artifact-bucket>/
  campaigns/cloud-v1/gate2/<run_id>/
    contract/plan.json
    contract/config-manifest.json            # SHA256 digests + image digest + host pair
    N1/edge/checkpoints/000001-<sha>.tar.zst
    N1/edge/checkpoints/000002-<sha>.tar.zst
    N1/edge/commits/000001.json             # committed verified segment chain
    N1/edge/commits/000002.json
    N1/cloud/checkpoints/...                # distinct placement stream
    N1/distributed/checkpoints/...
    N1/validation-report.json
    ...
    N9/...                                  # each of 9 blocks
    gate2-final/aggregate.json
    gate2-final/complete.marker.json
```

You may use compressed JSONL segments instead of TAR files, but **each committed object must be immutable, content-addressed and independently verifiable**. Store no AWS keys or HF tokens. Keep model weights and licensed datasets out of the checkpoint archive unless your redistribution/storage terms explicitly allow them; store verified artifact IDs and restore access instructions instead.

### Checkpoint manifest contract

Each **S3 commit manifest** records at minimum:

| Field | Purpose |
|---|---|
| `campaign_id`, `run_id`, `block_id`, `policy` | Stable identity; no accidental mixed environment |
| `git_sha`, `docker_image_digest`, `terraform_revision`, `host_pair_id` | Reproduce infrastructure and execution code |
| `model_contract_sha256`, `dataset_manifest_sha256`, `split_sha256`, `network_grid_sha256` | Protect research validity and train/test separation |
| `seed`, `schedule_sha256`, `plan_sha256`, `checkpoint_sequence` | Recreate the exact planned trial order |
| `completed_trial_keys`, `status_counts`, `last_trial_key` | Resume accurately; failures count as attempts |
| `segment_s3_keys`, `segment_sha256`, `byte_sizes`, `committed_at_utc` | Prove content and durability |
| `network_evidence_ref`, `container_id/image_digest`, `environment_hash` | Link measured conditions and execution host |

Trial identity is the composite `(campaign_id, block_id, policy, request_id, repetition_index)`. Use a distinct `attempt_id` for any **explicit** retrial, without replacing or hiding the original failed row. For the existing Gate 2 schedule, a failed completed attempt still occupies its planned trial key: only an interrupted *uncommitted in-flight* attempt is eligible for replay after a crash. Never convert an absent row into a success or silently retry failures and count only the favorable outcome.

### Atomicity, exactly-once records and recovery

1. Commit local trial row with a stable key and `fsync` before advancing the schedule.
2. Seal the next small segment locally (temporary file → `fsync` → atomic rename); calculate SHA-256 and upload to a **new** content-addressed S3 object. Never overwrite existing historical segments. S3 supports conditional writes (`If-None-Match: *`) and strong read-after-write consistency for object operations; use conditional writes or immutable unique keys, and verify the remote object and checksum before publishing a commit pointer. <https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html> and <https://aws.amazon.com/s3/consistency/>
3. Write a new **immutable commit manifest** which lists the entire verified segment chain, plan hash, and last completed trial key. This commit is the authoritative latest recoverable state. Only advance a mutable optional `latest.json` pointer *after* the immutable commit succeeds; support discovering the highest verified commit if the pointer is stale. Enable S3 versioning and block-public-access.
4. On resume, stop any competing writer, find the newest **valid** manifest, download the referenced segments, verify all SHA-256 digests/row keys and full frozen campaign contract, rebuild `requests.jsonl`/plan into a fresh persistent workspace, and invoke the existing `benchmark --resume` on the exact same schedule. Restore the **live** network shape on both peers and run fresh `iperf3`/RTT preflight; cached `network_pre.json` is not evidence of live shaping.
5. On intentional restart, keep both the local and S3 copy. If EBS has more verified trials than the last S3 commit, reconcile by immutable keys/segment hashes before proceeding; never merge incompatible branches or adopt divergent settings. Log any retried in-flight work as such. For *VM loss*, resume from last committed S3 state (work after that commit may be rerun but must not duplicate the authoritative output).
6. After each policy/block, generate and hash the official existing `profile.jsonl`, `summary.json`, and `validation_report.json`, and write a **final block commit**. Final Gate 2 completion is a separate aggregate commit; it requires independent proof of 8,100 planned measured keys, including failures. A commit is not a scientific success gate unless coverage and measurement rules also pass.

**Checkpoint frequency vs experiment bias:** the S3 uploader must not compete with controlled GPU timings or inference-link shaping. Use the independent S3 gateway/control path, a single bounded low-priority upload worker, and capture timestamps. If unavoidable, schedule uploads between measured requests, not inside the model's timed `infer` call. Validate A/B impact on latency; report checkpoint cost/overhead separately. Docker/application process metrics may include journal work while the inference timer must continue to cover the documented stable boundaries.

### Deterministic crash-injection validation (mandatory)

Start with a **small fake-adapter** matrix and then one tiny pinned-GPU matrix. For every test save the before/after list of unique trial keys and the recovered commit hash:

- Kill the edge benchmark process/container with `SIGKILL` after a committed request but before the next checkpoint; verify S3 recovery loses at most the uncommitted window and does not duplicate an authoritative row.
- Terminate during multipart or segment upload; ensure incomplete objects are uncommitted and never treated as restorable state.
- Stop/reboot the edge EC2 machine mid-block, then recreate the container on the same EBS; recover with `--resume`.
- Destroy/recreate the **experiment** EC2 node in a test environment (keep the protected S3 artifacts); reconstruct image/cache and resume from S3 alone, validating the reconstructed manifest.
- Temporarily deny S3 egress/permissions (in a safe test environment), observe the checkpoint failure/alarm, restore access, and verify eventual immutable commit; do not falsely mark checkpoint success.
- Alter a test plan SHA or Docker image digest and assert `--resume` refuses the incompatible recovery; test a corrupted segment/checksum similarly.

**Gate R0 must pass before running the complete 8,100-attempt Gate 2 campaign.** New scripts `scripts/checkpoint_campaign.py`, `scripts/restore_campaign.py`, and `scripts/test_checkpoint_recovery.py` are **proposed** and do not exist yet. Prefer a small module integrated with the existing benchmark's trial-recording callback and `--resume` semantics over a second competing scheduler.

### Gate R0 — Checkpoint durability and restore (must pass BEFORE the full Gate 2)

- [ ] New checkpoint manager is tested first with fake adapter and then with the pinned GPU model through the Docker execution path.
- [ ] Every completed trial is written/flushed locally; S3 commit manifests are immutable and content-hash verified.
- [ ] Recover successfully after killing a benchmark container mid-policy, stopping/restarting the edge EC2 instance, and recreating the edge instance/container with only S3 artifacts and Terraform-deployed configuration.
- [ ] Recovery rejects wrong Git/image/model/dataset/split/network-grid/infrastructure revisions and corrupted/incomplete snapshots.
- [ ] No missing/duplicate authoritative trial keys; warmups are not mistaken for measured attempts and failed attempts remain accounted for.
- [ ] S3 outages cause explicit degraded/paused state and safe checkpoint retries; no false success/commit marker is written.
- [ ] Upload traffic is isolated from measured inference traffic and does not contaminate results; periodic checkpoint overhead is separately measured.

## 7. Execute Gate 2: the full fixed-policy factorial matrix

Your repository's declared development matrix remains:

```text
100 frozen VQA requests
  × 9 independently verified network blocks
  × 3 fixed placements (edge/cloud/distributed)
  × 3 measured repetitions
  = 8,100 measured inference attempts

+ 9 blocks × 3 placements × 1 warmup = 27 warmups.
```

The **Dockerized** experiment orchestrator runs **on the virtual-edge VM**; the laptop is not in the request/response critical path. The dual-path server remains running on the virtual-cloud VM. Use systemd-controlled Docker Compose, durable EBS and periodic S3 checkpoint commits to keep work recoverable across SSH disconnections, container crashes, host reboots and VM replacement. **Merely running inside `tmux` is not a checkpoint strategy.**

After the **cloud-specific** runner is implemented, illustrative command for a block:

```bash
# On edge VM after BOTH hosts are shaped to the chosen N1 state
.venv-model/bin/python scripts/run_cloud_profile_block.py \
  --block N1 \
  --config configs/experiments/cloud_vqa_v1.yaml \
  --grid configs/profiles/cloud_network_blocks_v1.json \
  --split configs/evaluation/cloud_vqa_split_v1.json \
  --output-root artifacts/phase2/cloud-v1
```

**Warning:** `scripts/run_cloud_profile_block.py` is a **proposed new script**; it does not yet exist in your repository. Do not run this command until the implementation and dry-run gates pass. The existing `run_physical_profile_block.py` can serve as the starting implementation.

Order N1…N9 need not be chronological. Within each block use the current seeded policy-order permutations and per-policy seeded request permutations. Keep start/end GPU temperature/load telemetry if available, and avoid simultaneous profiling jobs. Existing runner executes each policy in a separate process, rather than interleaving each request across policies; it balances order **by block**, not by request. State this as a limitation and optionally run an independent crossover-confirmation subset with interleaved policy order if memory residency and model reload permit.

If a block crashes, repair the infrastructure cause, **restore the latest committed S3 checkpoint, verify every hash and completed trial key**, and use `--resume` without duplicating trials. Mark server outage/Spot interruption explicitly; do not turn a runtime failure into a successful omitted row.

### Outputs per block

```text
artifacts/phase2/cloud-v1/N1/
  plan.json
  environment.json
  network_pre.json
  network_post.json
  network_evidence.json
  edge/
  cloud/
  distributed/
  edge_gpu_samples.csv
  cloud_gpu_samples.csv
  checkpoints/                  # local manifest + immutable segments + receipts
  checkpoint_recovery_report.json
  profile_specification.json
  profile/profile.jsonl
  profile/summary.json
  validation_report.json
```

During **every** block, persist each completed attempt locally, checkpoint immutable segments to S3 on a time/attempt threshold, and commit an S3 recovery manifest. After each validated block, archive full raw logs and generated outputs to the same private versioned bucket. Record content hashes and verify restore on a separate machine. Do not commit raw model weights, licensed image data, credentials, or huge experimental JSONL files to GitHub.

### Gate C6 — Complete Gate 2

- [ ] 8,100 planned measured trial keys are accounted for, including failed attempts.
- [ ] 27 warmups excluded from measured statistics.
- [ ] All 9 network blocks pass independent pre/post checks.
- [ ] No model/dataset/config/host-pair hash mismatches.
- [ ] Each placement has measurement coverage and explicit failure/OOM accounting.
- [ ] No unexplained negative timing residuals or corrupt byte accounting.
- [ ] GPU peak/available memory and server-side resource evidence archived.
- [ ] Profile converter and final aggregate audit both pass.
- [ ] Every block has a **validated final S3 checkpoint/commit**, a successful restore test, and an independently verified final archive; Gate 2 summary and experimental limitations reviewed.

## 8. Proceed with the complete thesis without physical devices

**After Gate 2:**

- **G3 — Predictor calibration:** Fit only using training requests and training network blocks. Refit for the **new cloud host pair**. Never reuse old physical/localhost performance coefficients. Validate on held-out requests and N2/N7; freeze model/predictor before touching final evaluation.
- **G4 — Selector/oracle:** Compare selected placement against retrospective fastest *feasible* placement and fixed baselines; report regret, SLA violations, exclusions, prediction ranking accuracy, failures, and uncertainty. Use predeclared quality and memory constraints.
- **G5 — Dynamic runtime:** Run the already-implemented adaptive benchmark with a newly frozen cloud replay and its matching calibration, only after G3/G4 gates pass (or mark diagnostic runs explicitly).
- **G6 — Time-varying network traces:** Replay unseen realistic measured bandwidth/jitter/loss sequences; compare adaptive vs fixed placement and a static oracle. Verify achieved network timing during the run; evaluate decision stability and overhead.
- **Ablations:** No network adaptation; no resource constraint; no request-aware state; alternative network predictors. Quantify contributions without overfitting test conditions.
- **Final accuracy:** Use the **official VQA v2 scoring procedure** rather than presenting the repository's provisional exact-match diagnostic as official benchmark accuracy.
- **Publication:** State both the scope and limits of cloud-hosted edge emulation, and report hardware profile, network emulation trace sources, seed, software provenance, per-request latency and uncertainty.

**Optional only after thesis:** rerun a frozen subset on the physical RTX 3050 + RTX 2070 SUPER pair (or a Jetson + remote server) and compare calibration transfer. Do **not** label this a missing thesis deliverable.

## 9. Budget, reliability, and operational protocol

Illustrative AWS US East Linux On-Demand compute rates as found in public catalogs on 3 Nov 2026: `g6f.2xlarge` approximately **$0.475/hour** and `g6.2xlarge` approximately **$0.9776/hour**, totaling approximately **$1.4526/hour** while both run. These are *indicative* US-region figures, not a quote and exclude storage, public IPv4, Internet egress, taxes and other services. Availability, actual rates, and GPU limits must be confirmed before launch. The thesis should not pay for continuous idle instances. Add EBS, S3 objects/requests, ECR storage/image scans, CloudWatch ingestion, SSM/VPC endpoints (if billed), NAT gateways if used, and data transfer to the forecast. S3 gateway endpoints themselves have no hourly charge, but other VPC endpoints/NAT might. Artifact versioning also increases retained storage unless lifecycle rules are designed deliberately. Track the overall AWS bill, not just GPU instance prices.

**Budget is infrastructure, not a dashboard task:** define `monthly_budget_usd`, `experiment_max_runtime_hours`, `idle_timeout_minutes`, and notification recipients as Terraform inputs. Terraform should provision an AWS Budgets cost budget with **50%, 80%, and 100% actual** notifications plus a **100% forecast** notification, and CloudWatch alerts for failed uploads, missing checkpoint heartbeat, inactive benchmark with EC2 still running, GPU/CPU utilization anomalies, and storage exhaustion. Monitor the **S3 commit heartbeat** from the checkpoint manager: if no valid commit appears within the target interval while work continues, raise an immediate operational alarm.

**Budget alerts are NOT a billing kill switch**: AWS Budget data and notifications are not instantaneous (billing data refreshes at least daily). Implement a separate runtime/idle watchdog that checkpoints, terminates the controlled experiment, and requests/initiates safe EC2 stop when predeclared run-hour ceilings are reached. Exercise the watchdog before long experiments. Use a manual emergency EC2 stop only as a last resort and document Terraform drift. <https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-best-practices.html>

Controls:

1. **Terraform budget-first:** apply and confirm the budget/alerts/roles/S3/ECR/state infrastructure *before* enabling GPU nodes; review Terraform plans for unexpected or publicly accessible resources. Verify destination email/SNS subscriptions as needed.
2. **Container smoke:** begin with fake and GPU model correctness tests in the final digest-pinned Docker image, not the complete 8,100-attempt campaign.
3. **Reliability preflight:** run the S3 checkpoint crash tests (Gate R0), including an instance/container restart and restoring a segment from S3 to a new workspace. Audit S3 IAM access, versioning, lifecycle, integrity verification and checkpoint upload costs.
4. **Run budget guard:** cap intended per-block hours and total campaign GPU instance-hours, schedule automated idle stops and make the watchdog checkpoint before orderly shutdown. Do not rely on delayed AWS Budgets notifications as a real-time cap.
5. **On-Demand for controlled baseline:** keep the paired instances On-Demand while measured; add Spot only after recovery passes and interruption effects are tracked as a separate cohort.
6. **S3 during, not after:** sync immutable checkpoint segments at ≤25 new attempts/5 min and finalize a verified S3 archive at every block boundary. Encrypt buckets/volumes, use restrictive roles, and set lifecycle/retention to keep enough history until thesis acceptance.
7. **Persistent storage:** use EBS for the model cache and working data, not a model-image layer. Stopping EC2 generally stops compute billing but **does not** stop EBS/S3/ECR and some networking/storage charges.
8. **Services and logs:** manage Docker Compose through systemd, enforce container health/restart policy, capture GPU/CPU and checkpoint metrics, log rotation and storage-use alarms.
9. **After each block:** confirm the terminal block commit, reconcile expected/measured row keys, kill any hung benchmark processes, test restore, and then stop nodes if another block is not starting.
10. **Stable experimental hardware:** maintain identical instance types, AZ, image digests and model contracts throughout one campaign. An instance replacement or driver change is documented as a possible confounder and may require a distinct environment revision.

### Fail-safe rollback and cost stop

- Ask the Dockerized orchestrator to stop accepting new trials and flush/finalize its local journal; attempt an immutable S3 checkpoint commit and report the last durable commit, including any unprotected tail.
- Restore the experiment qdiscs/classes on **both VMs**; confirm baseline routing and that S3/SSM management traffic was never part of the shaped scientific path.
- Stop Compose services via systemd; preserve the experiment's partial run directories and cloud/edge logs. If the VM is already gone, restore solely from the last **verified S3 commit**, not an imagined in-memory state.
- Verify S3 segment hashes/commit linkage, `plan.json`, result keys and checkpoint recovery metadata; separate the state bucket from thesis data.
- Stop the two Terraform-managed EC2 instances (an operational stop, **not** `terraform destroy`). Confirm billing/resources after the stop; later `terraform destroy` only the disposable experiment stack after validation and retention review.
- If an alarm triggers when no checkpoint can be uploaded, stop accepting new requests and preserve local data on EBS; do not delete failed/incomplete data or mark a block as finished.

## 10. Updated execution order and mandatory gates

| Order | Step | Acceptance / checkpoint |
|---|---|---|
| 1 | **Terraform bootstrap**: protected remote state (versioning/locking), identity and permissions | `terraform init/fmt/validate/plan/apply` pass without console provisioning |
| 2 | **Terraform budget/security/storage**: Budgets, private S3, ECR, roles, VPC endpoints and CloudWatch | Budget/notifications/roles/alarms pass **T0** before GPU launch |
| 3 | **Terraform compute**: g6f virtual edge + g6 virtual cloud, same AZ, persistent EBS | Correct instance IDs, price/quota and supported drivers verified |
| 4 | **Build/push Docker**: pin the CUDA base, Python 3.11, model requirements, code and ECR digest | GPU visible on both nodes, identical digest, volumes retained; **D0** |
| 5 | **Cloud functional smoke** on three partitioning methods with 12 frozen requests | Correctness, numerical/contract checks and 6-GB feasibility; **C1/C2/C5** |
| 6 | **Implement S3 checkpoint integration** in existing runner; fake and GPU dry runs | Durable per-trial journal, immutable segments and commit manifests |
| 7 | **Inject failures and restore** on same EBS and new VM from S3 | Test SIGKILL/partial upload/EC2 restart/corrupt hashes; **R0** |
| 8 | **Measure fresh unshaped baseline, then all nine controlled blocks** from Docker app path | `iperf3`, echo RTT, qdisc paths and independence validated; **C4** |
| 9 | **Run mini-Gate 2**: 2 requests × 3 placements × 1 network, with forced checkpoint and recovery | Check provenance, stable runtime, correct resume, no double count |
| 10 | **Full Gate 2**: 8,100 measured attempts (+27 excluded warm-ups), with checkpoints during every block | All keys/statuses, network evidence, GPU memory and profile audits; **C6** |
| 11 | **Commit final S3 archive and verify independent restore** | Final aggregate + complete marker; no abandoned trial window |
| 12 | **Proceed with G3/G4/G5/G6**, then safe shutdown and documented Terraform teardown | No physical-device dependency; controlled research validity preserved |

### Final completion definition

Gate 2 is complete **only** when it has scientifically valid profiling data **and** reliable infrastructure evidence: (a) the full frozen trial matrix accounted for, (b) independently verified network conditions, (c) pinned Docker/model/host provenance, (d) verified S3 checkpoint chain and independent restore, and (e) reviewed Terraform/expense state. Infrastructure reproducibility and recoverability are new **required** gates, not optional DevOps extras.

## 11. Source links and repository basis

**Inspected repository files** (main branch viewed 3 Nov 2026):

- https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/README.md
- https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/docs/physical_profile_campaign_v1.md
- https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/docs/physical_dataset_protocol_v1.md
- https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/docs/physical_network_measurement_v1.md
- https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/scripts/run_physical_profile_block.py
- https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/configs/profiles/physical_network_blocks_v1.json
- https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/configs/evaluation/physical_vqa_split_v1.json

**Infrastructure references**:

- Terraform S3 remote state and native locking: https://developer.hashicorp.com/terraform/language/backend/s3
- Terraform AWS Budgets resource: https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/budgets_budget
- AWS Budgets alert frequency: https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-best-practices.html
- S3 conditional writes: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html
- S3 strong consistency: https://aws.amazon.com/s3/consistency/
- S3 VPC gateway endpoints: https://docs.aws.amazon.com/vpc/latest/privatelink/vpc-endpoints-s3.html
- NVIDIA Container Toolkit for Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- Docker GPU support: https://docs.docker.com/engine/containers/gpu/
- AWS G6f fractional GPU instances: https://aws.amazon.com/about-aws/whats-new/2025/07/amazon-ec2-g6f-instances-fractional-gpus/
- AWS EC2 accelerated instance specifications: https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html
- AWS NVIDIA driver support: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
- AWS GRID drivers: https://docs.aws.amazon.com/en_eu/AWSEC2/latest/UserGuide/nvidia-GRID-driver.html
- AWS VPC security groups: https://docs.aws.amazon.com/vpc/latest/userguide/security-group-rules.html
- AWS Systems Manager Session Manager: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html
- Linux `tc-netem`: https://man7.org/linux/man-pages/man8/tc-netem.8.html
- Current public instance-price catalogs: https://instances.usage.ai/aws/ec2/g6f.2xlarge and https://instances.usage.ai/aws/ec2/family/g6

**Final scientific statement:** The Terraform-managed, Docker-packaged and S3-checkpointed cloud arrangement gives reproducible, recoverable, independently scheduled heterogeneous compute and realistic *emulated* network impairment; it does not prove performance on physical edge hardware or on a live 5G radio network. State these limitations, validate achieved conditions independently, and use physical validation later if available.
