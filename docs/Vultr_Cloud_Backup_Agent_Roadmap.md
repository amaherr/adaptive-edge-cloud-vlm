# Vultr Backup Deployment — Agent Implementation Roadmap

**Project:** [amaherr/adaptive-edge-cloud-vlm](https://github.com/amaherr/adaptive-edge-cloud-vlm)  
**Document revision:** `vultr-backup-plan-v1` — 7 November 2026  
**Intended branch:** `feature/vultr-cloud-backup` (or the user's chosen Vultr branch)  
**Purpose:** Complete the bachelor's thesis using **two independent, Terraform-provisioned Vultr GPU VMs**, Dockerized model execution, controlled Linux network emulation, and resumable, independently stored experiment artifacts. Preserve AWS as a parallel option. No physical university device is required for thesis completion.

> **For the coding agent:** This is an execution specification, **not** proof that Vultr capacity, billing limits, GPU CUDA support, or particular GPU plan IDs have been approved. Resolve those dependencies with a small preflight. Do not start paid GPU resources or the full 8,100-attempt run without the owner's explicit approval of the instance plan, projected spend, and launch. Do not claim an unperformed test has passed.

---

## 0. Hard rules, scope and known facts

### 0.1 Non-negotiable requirements

1. **Separate branch and clean isolation.** Do not modify or delete the existing AWS Terraform environment, AWS branch, or previously collected physical experiment artifacts. Never force-push or rewrite `main`.
2. **Terraform for infrastructure.** Provision Vultr GPU instances, private networking, firewall, SSH-key resources, container registry, and supported storage resources as code. `terraform plan` is mandatory before `apply`. Avoid UI-based instance provisioning. Account registration/API credential issuance may still require human action.
3. **Docker application.** Run the *same immutable application image digest* on edge-role and cloud-role VMs, with different roles/configurations. Use NVIDIA GPU pass-through to containers. Keep driver/kernel management on the host.
4. **Two independent VMs.** One appropriately constrained virtual edge and one more capable virtual cloud, preferably same Vultr region and VPC; no single-host/two-container experiment masquerading as independent hardware.
5. **Freeze the research contract.** Same pinned Moondream2 artifact, BF16, preprocessing, tokenizer, greedy decoding, max 32 new tokens, VQA request IDs and answer annotations, seeds, and the three existing fixed policies. Changing hardware creates a **new campaign**; do not silently pool old and new data.
6. **Network control + independent measurement.** Shape both directions of the *actual inference link* with `tc/netem`. Verify achieved rates with `iperf3` and actual round-trip timing with the repository's TCP echo probe. Protect control plane, checkpoint uploads, and SSH from data-plane shaping.
7. **Crash-safe checkpointing.** Existing per-attempt `requests.jsonl` remains authoritative locally. Upload immutable hash-verified segments/commit manifests to an independent object store every **25 completed trials or 5 minutes**, whichever occurs first, plus policy/block transitions. Restore and `--resume` must work after a destroyed VM.
8. **Budget protection.** Never count a powered-off Vultr GPU instance as cost-free. Destroy/recreate GPU VMs outside sessions while retaining checkpoints, image, Terraform state, and required persistent resources; prove the teardown process works before extended experiments.
9. **No invented results.** Actual performance, GPU fit, network capacity, costs, and validation status must be measured and archived. The physical RTX 3050 / RTX 2070 SUPER results are historical reference only.
10. **No scheduler development until profiling works.** Gate 2 is the fixed-placement performance matrix. G3 predictor / G4 selector / later dynamic traces follow after Gate 2.

### 0.2 Repository components to reuse (verified on `main`)

| Existing path / command | Purpose | Migration action |
|---|---|---|
| `configs/model/moondream2.yaml` and `configs/model/moondream2_artifact.yaml` | Frozen model contract and verified artifact | **Reuse unchanged** |
| `requirements/model-cu128.lock` | Pinned Python/PyTorch runtime | Use inside Docker, audit host driver compatibility |
| `data/manifests/physical_vqa_v1.jsonl` | Frozen 100 VQA requests | Reuse content/hash; keep locally resolvable image paths |
| `configs/evaluation/physical_vqa_split_v1.json` | 60 train, 20 validation, 20 final evaluation requests; frozen block holdouts | Reuse same split; copy only if new campaign naming requires it |
| `configs/profiles/physical_network_blocks_v1.json` | N1–N9 impairment settings | Copy to Vultr-specific grid; replace host/interface/unshaped baseline |
| `configs/experiments/physical_vqa_v1.yaml` | Benchmark's common manifest and transport configuration | Copy into `vultr_vqa_v1.yaml` with new hardware/host/paths |
| `scripts/run_physical_profile_block.py` | Runs three placements, probes network, builds profile | Refactor into a provider-agnostic runner; retain physical compatibility |
| `scripts/tcp_rtt_probe.py` | Bidirectional application-path RTT checks | Reuse |
| `scripts/fetch_vqa_subset_images.py` | Materialize immutable official COCO subset | Reuse and verify SHA-256 |
| `adaptive-vlm benchmark --mode edge/cloud/distributed` | Frozen benchmark runner; `--resume` | Reuse; extend only for durable checkpoint events |
| `adaptive-vlm profile-runs` / `compare-runs` | Validate trial set, bytes, timings, token equality | Reuse with Vultr campaign specification |
| `docs/Gate2_Cloud_Migration_Roadmap_v2.md` | Existing AWS cloud design | Keep unchanged as AWS reference; extract common code only |

**Actual checked repository detail:** The physical runner hard-codes `edge-rtx3050-cloud-rtx2070super-v1`, its network `source` is `physical_wsl_netem_measured`, and it assumes WSL's `eth0` layout. Those values **must not** appear in Vultr-generated evidence. It also requires a clean Git tree and verifies configured qdiscs. Its `--resume` behavior must be preserved.

**Branch discovery caveat:** GitHub's visible default branch was `main` when this plan was prepared; a separate AWS development branch was not visible through the connected repository listing. The agent must inspect the *local* repository and the user's actual remotes to locate AWS work instead of assuming its exact branch name or content.

### 0.3 Do not confuse the experimental phases

- **Gate 2 / fixed profiling:** 100 frozen requests × 9 network blocks × 3 fixed placements × 3 measured repetitions = **8,100 measured attempts**, plus **27 warmups**. Every attempt, including errors/OOMs, must have an accounted-for outcome.
- **G3/G4:** Fit/evaluate the predictor/selector only after fixed profiling, using declared request and network holdouts.
- **Dynamic trace campaign:** Separate later evaluation with time-varying links. A constant `netem` impairment is *controlled network emulation*, not a real cellular handover or a real edge device.
- **Physical hardware validation:** Optional future work after thesis completion, never a dependency for this cloud-hosted plan.

### 0.4 Acceptance vocabulary

A gate is `PASS` only with a pointer to evidence (raw command output, test report, JSON, SHA-256 manifest, Git commit, Terraform plan, Docker image digest, provider quote). `BLOCKED` is not `PASS`. Budget, GPU capacity, and account approvals are external gates requiring the user or provider.

---

## 1. Target topology and provider-boundary design

```text
                   Git repository / CI / operator
                     Terraform + budgets + approval
                                |
              +-----------------+-----------------+
              |      Vultr (same region)         |
              |                                  |
              |  GPU VM E: VIRTUAL EDGE          GPU VM C: VIRTUAL CLOUD
              |  smaller genuine GPU/vGPU         larger genuine GPU/vGPU
              |  Docker: experiment runner        Docker: dual-path VLM server
              |  Docker: edge/full or vision      GPU: full/cloud or text-only
              |       |                                   ^
              |       +--- private VPC inference link -----+
              |            tc/netem on BOTH egress NICs
              |            iperf3 and TCP echo probes
              +------------------+-----------------------+
                                 |
                 separately routed CONTROL / artifact path
                                 |
                    S3-compatible object storage
                   immutable journal/checkpoint commits
                     + remote Terraform state
                                 |
                      independent cost watchdog
                    + destruction/recreation runbook
```

**Design choice:** Keep inference and `iperf3`/echo on the VPC's experiment interface. Keep SSH/control, image pulls, artifact backups, and billing watchdog traffic on a separately verified unshaped route. If this cannot be achieved with the chosen plan and VPC topology, do not run the benchmark: redesign the impairment point (dedicated interface, namespace/router, or traffic classifiers) and validate packet routing first.

**Data-plane ports:** 5050 inference server, 5051 `iperf3`, 5052 TCP echo, private peer only. Public exposure of any is forbidden. The current model protocol performs integrity checks but **does not implement TLS or authentication**; private network restrictions are mandatory. Optional overlay/private VPN is permissible only if its overhead and route are measured and archived.

**Compute realism:** Two Vultr virtual machines are physically independent compute allocations but remain datacenter hardware. Fractional vGPU is a real quota of shared GPU hardware, **not a Jetson simulator**. Report real CPU/RAM/GPU identity, actual VRAM, occupancy and GPU clocks; do not present virtual-edge energy consumption as embedded-device power.

---

## 2. Stage P0 — Branch strategy, code inventory, and freeze

### Work

1. Inspect Git status, remotes, branches, recent commits and tags. Preserve all changes before switching.
2. Prefer a Vultr feature branch off `main` **after** finding whether AWS branch has completed provider-neutral Docker/checkpoint code:
   - If provider-neutral features exist on an AWS branch, cherry-pick only independently tested commits or merge extracted files into Vultr branch.
   - **Do not merge** AWS Terraform resources, IAM, VPC naming, CloudWatch budgets, or EC2-specific startup scripts into Vultr.
   - If the existing Vultr branch already exists, switch to it without recreating it.
3. Record the main baseline commit and AWS feature commit(s) separately. Obtain a clean tree and make sure existing tests pass before edits.
4. Preserve and link the prior cloud roadmap. Create this document under `docs/vultr_backup_agent_roadmap.md` in the new branch if appropriate.

### Suggested terminal commands (operator/agent, local)

```bash
git fetch --all --prune
git status --short
git branch -a
git log -n 8 --oneline --decorate
git tag --list
# If branch doesn't exist and user agrees:
git switch main && git pull --ff-only
git switch -c feature/vultr-cloud-backup
# Else: git switch <already-existing-vultr-branch>
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest -q
```

### Required records

`docs/vultr/decisions.md`: branch base, reused AWS commits, independent scope, baseline Git SHA, pinned model contract and dataset hashes. `docs/vultr/status.md`: current stage and gates.

**Gate P0:** Clean branch, tests pass, AWS work preserved, no old experiment file overwritten.

---

## 3. Stage P1 — Provider/account and GPU capacity feasibility (NO expensive launch)

This is a **selection gate**, not a presumption about a provider's current inventory. Vultr uses Cloud GPU (`vcg` family), and the Terraform resource is `vultr_instance`. Exact plan IDs and regional stock vary and must be obtained from the authenticated account/API, not copied from an example.

### 3.1 Human/account prerequisites

- [ ] Vultr account registration, verification, billing/payment method, API activation and GPU deployment entitlement checked.
- [ ] User supplies a **scoped Vultr API token** through environment/secret manager, never chat, repo, Terraform state, Docker image or logs. Restrict API source IP if supported.
- [ ] User chooses desired maximum *total experimental spend*, a smaller preflight authorization, alert destination, and the maximum simultaneous GPU instance count (**two**).
- [ ] Confirm account can create **two** requested GPU VMs concurrently in the **same** region. Request account approval/support if unavailable; do not repeatedly buy unsuitable instances.
- [ ] Confirm offered OS image supports NVIDIA/CUDA for chosen GPU type, docker runtime, Linux CAP_NET_ADMIN, and private VPC NIC.

### 3.2 Enumerate actual plans through the API

```bash
# On trusted local terminal. Never print/store the key in logs.
export VULTR_API_KEY='***SET_IN_SECRET_ENVIRONMENT***'
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${VULTR_API_KEY}" \
  'https://api.vultr.com/v2/plans' > /tmp/vultr-plans.json
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${VULTR_API_KEY}" \
  'https://api.vultr.com/v2/regions' > /tmp/vultr-regions.json
# Inspect the response schema, pagination and actual vcg plans.
# Verify target region via the region availability API or Vultr CLI.
```

Implement `scripts/vultr/check_plans.py` (proposed): parse *all pages*, filter GPU plans, join region support, print price/hour, vCPU, RAM, GPU type, GPU memory, disk, network allowances, and available plans. Handle `HTTP 403/429`, missing GPU permissions and region stock without claiming approval. **Do not hardcode a plan ID** in the example environment `.tfvars` until it has been verified.

### 3.3 Hardware selection rubric

**Virtual edge:** smallest *genuine* GPU/vGPU allocation that can run the frozen **entire** Moondream2 BF16 path with measured safe memory headroom. Target 6–8 GB GPU memory *only if an actual purchasable plan exists*, otherwise an explicitly documented 16 GB GPU allocation may be the feasible constrained baseline. Your historical physical `edge/full` reserved roughly 4.2–4.45 GiB despite a 4 GiB physical card; the new VM needs a fresh memory test. Do not confuse GPU-memory size with compute throughput.

**Virtual cloud:** larger GPU allocation (for example a full or larger fractional A40/L40S class, **subject to actual availability**) that independently runs the complete VLM, with ample VRAM and practical hourly price.

**Do not claim** a specific Vultr 8 GB A40 SKU exists until the API and vendor quote confirm it. Vultr advertises fractional GPU offerings; its public overview lists an A16 starting configuration with **16 GB advertised GPU memory**, not proof of an available 8 GB tier. A16 vs A40 introduces both GPU-model and capacity differences; record them explicitly.

| Criterion | Edge acceptance | Cloud acceptance |
|---|---|---|
| Device | NVIDIA CUDA with supported vGPU compute if fractional | NVIDIA CUDA with more compute/memory |
| Model | full BF16 model + vision-only both fit reliably | full model and text-only role fit reliably |
| Software | host driver + Docker CUDA passes | same application digest + compatible host driver |
| Resource isolation | distinct VM, verifiable GPU allocation | distinct VM, verifiable GPU allocation |
| Networking | private VPC attachment and root/CAP_NET_ADMIN | same |
| Billing | quote within explicitly approved budget | quote within explicitly approved budget |

### 3.4 Budget worksheet before provisioning

Prepare `docs/vultr/cost_estimate.md` with **quoted** variables, not fabricated prices:

```text
edge_hourly = verified Vultr account quote
cloud_hourly = verified Vultr account quote
compute_estimate = (edge_hourly + cloud_hourly) * paired_session_hours
storage_estimate = object storage + registry + retained disk + snapshots
network_estimate = outbound/public transfer + other applicable charges
safety_margin = user-approved percentage/amount
total_estimate = compute_estimate + storage_estimate + network_estimate + safety_margin
```

Use a tiny **single-edge-node preflight** (model and driver) before any two-node launch. User signs off on the two-node quote separately. Preserve screenshot or API JSON with timestamp, but never expose account secrets.

**Gate P1:** Actual provider plan IDs, region, inventory/approval, account limits, per-hour costs, and explicit owner authorization recorded. If any are unknown: `BLOCKED`, not bypassed.

---

## 4. Stage P2 — Terraform infrastructure, secure state, and disposable GPU nodes

### 4.1 Proposed repository structure

```text
infra/
  terraform/
    vultr/
      bootstrap/                 # state backend + minimal persistent prerequisites
      environments/vultr-v1/
        versions.tf
        providers.tf
        variables.tf
        main.tf
        outputs.tf
        terraform.tfvars.example
        README.md
      modules/
        networking/              # vultr_vpc, firewall, SSH controls
        gpu_instance/            # reusable role-specific VM + cloud-init
        object_storage/          # subscription / bucket integration
        container_registry/      # private registry
        visibility/              # outputs + monitoring integration
container/
  Dockerfile
  compose.edge.yaml
  compose.cloud.yaml
  entrypoint.sh
scripts/vultr/
  check_plans.py
  preflight.sh
  deploy.sh
  checkpoint_control.py
  restore.sh
  cost_guard.py
  teardown.sh
  verify_destroy.py
  run_block.sh
configs/hardware/vultr_{edge,cloud}_v1.yaml
configs/experiments/vultr_vqa_v1.yaml
configs/profiles/vultr_network_blocks_v1.json
docs/vultr/
  decisions.md
  status.md
  cost_estimate.md
  runbook.md
  security.md
  validation.md
```

These are **proposed files**, not a claim that the current repo already contains them. Reuse any genuinely shared AWS Docker/checkpoint code instead of cloning divergent implementations.

### 4.2 Terraform resource design

- Pin `hashicorp/terraform` and `vultr/vultr` versions and commit `.terraform.lock.hcl`. Use the official provider schema for every argument.
- Use `vultr_instance` with `plan = var.edge_plan_id` / `cloud_plan_id`, region, verified OS ID/GPU image, SSH key IDs, VPC IDs, firewall group IDs, labels/tags, and **minimal, nonsecret** cloud-init.
- Use the **current** documented `vultr_vpc` resource (the older `vultr_vpc2` resource is marked deprecated in the provider's docs). Confirm both GPU types can attach before selection.
- Use `vultr_firewall_group` / `vultr_firewall_rule` and host `ufw`/nftables defense-in-depth. Restrict 5050–5052 to the **peer private IP/subnet**; SSH only from the approved administrative source or private VPN. **Do not rely on public firewall alone to protect private-interface traffic.**
- Use `vultr_ssh_key` where useful. Never store a private SSH key in Terraform or Docker.
- Use `vultr_container_registry` if feasible, otherwise an explicitly approved registry with immutable image digests. Private by default; account for extra recurring registry cost.
- Use `vultr_object_storage` if selecting Vultr Object Storage. This resource provisions an **object-storage subscription**; bucket provisioning/versioning still needs a supported S3 API management path. Confirm whether Terraform's S3-compatible provider setup can manage bucket settings safely; otherwise implement and test an idempotent declarative bootstrap step. Do **not** claim all storage is Terraform-managed if an API bootstrap remains external.
- Use separate persistent storage for Terraform state and experiment artifacts. Root Terraform state is secret-bearing; never put it inside the destructible GPU node or Git.
- Variable `deploy_gpu_instances` defaults to **false**. Deployment must require explicit approval, cost precheck and `terraform plan` review.
- Output private addresses, hardware/region IDs, storage endpoint and registry URL as appropriate. **Never output** access keys/passwords in a normal CI/job log; note Terraform state itself may include secret outputs from resource providers.

**Illustrative, non-executable until provider/schema and plan IDs are verified:**

```hcl
variable "region"              { type = string }
variable "edge_plan_id"        { type = string }
variable "cloud_plan_id"       { type = string }
variable "deploy_gpu_instances" { type = bool; default = false }

resource "vultr_vpc" "inference" {
  region      = var.region
  description = "vultr-v1-private-inference"
}

resource "vultr_instance" "edge" {
  count  = var.deploy_gpu_instances ? 1 : 0
  plan   = var.edge_plan_id
  region = var.region
  os_id  = var.verified_gpu_os_id
  label  = "vlm-vultr-v1-edge"
  # Populate from verified network/SSH/firewall resources.
  vpc_ids = [vultr_vpc.inference.id]
}
```

The snippet deliberately omits necessary security/SSH definitions; an agent must implement and validate full modules, not paste this as finished production infrastructure.

### 4.3 Remote state choice (do not mix provider billing and experiment state)

**Preferred when already available:** existing **AWS S3 remote Terraform state** only, even if AWS's EC2 GPU quota remains at zero. This does not require AWS GPU instances. Use bucket versioning, encryption, least privilege and proven locking. Confirm AWS S3 availability and any ongoing cost with owner.

**Independent fallback:** Vultr S3-compatible object storage or another backend that provides safe **concurrency locking and atomicity** with the chosen Terraform version. Verify real backend behavior including parallel writer rejection; S3 compatibility alone is *not* evidence that Terraform's exact locking operations work. Use a single CI/operator lock as a defensive additional measure. Keep the state backend protected from `terraform destroy`.

Never commit `terraform.tfstate`, `.tfvars` with secrets, provider API token, bucket secret, registry credentials, or Hugging Face tokens. Add CI secret scanning, Terraform formatting/validation, and a plan-only GitHub Actions workflow. Do not let a pull request automatically create paid GPU instances.

### 4.4 Teardown contract (Vultr-specific: CRITICAL)

**Power-off is not teardown.** Vultr bills servers from deployment until destruction even when powered off. The runbook must use Terraform to remove the GPU VM resources from desired state while retaining object storage, remote state, container image/registry, and any independently billed persistent disk intentionally approved.

Recommended Terraform lifecycle:

```bash
# Review change with the GPUs disabled
terraform plan -var='deploy_gpu_instances=false' -out=disable-gpus.tfplan
terraform show disable-gpus.tfplan
# After checkpoint verification and explicit owner/operator approval:
terraform apply disable-gpus.tfplan
# Verify via API that ZERO GPU instances remain with this campaign tag.
```

Do not use an unconditional `terraform destroy` on a stack also managing checkpoints/state. In an emergency, the operator may use the vendor API to terminate leaking instances, then immediately reconcile/import/refresh Terraform state; record this exceptional recovery. Recreating machines must be reproducible from Terraform + immutable Docker image + remote checkpoint.

**Gate P2:** `terraform fmt -check`, `terraform validate`, static security scan, reviewed plan with `deploy_gpu_instances=false`, state locking, two-node plan configuration, and test create/**destroy** lifecycle pass. No unrestricted public inference ports and no secrets in source or logs.

---

## 5. Stage P3 — Immutable Docker application + GPU runtime

### 5.1 One image, two roles

- Build an image from a pinned and security-reviewed Linux/CUDA runtime compatible with the chosen Vultr host driver and the repository's **PyTorch 2.7.1+cu128** lock / Python **3.11** requirement.
- Install the pinned dependency lock and the package entry point; avoid mutable `pip install -U` at runtime. Record base image digest, Python lock hash, source commit, build platform and final image digest.
- **Do not bake** model weights, source VQA images, API secrets or licensed assets into the image. Mount verified read-only model cache and dataset materialized by a bootstrap job.
- Run the **same image digest** on both VMs. Edge role is runner + edge/full + edge-vision; cloud role is the dual-path `adaptive-server` on TCP 5050; probes are separate lightweight services with dedicated security boundaries.
- For a remote cloud server, the dual-path server must accept both raw image and projected BF16 tensor protocols without restarts. Keep it resident during a block.
- Use NVIDIA Container Toolkit (`--gpus all` / Compose GPU device requests). On Vultr GPU-enabled images, check preinstalled driver, Docker and toolkit **before reinstalling**: vendor docs warn that indiscriminate Docker installation can conflict with GPU tooling.
- A non-root application container should not receive `--privileged`; place controlled `tc` operations in a minimal host-level privileged script/service or a dedicated network controller with only `CAP_NET_ADMIN`, audited interface/peer allowlists and a cleanup trap.
- Use systemd to supervise `docker compose`, and mount persistent paths for logs/artifacts/cache. Running in `tmux` alone is not a durability solution.

### 5.2 Minimum container contract

```bash
nvidia-smi
# The agent must verify driver/CUDA compatibility first.
docker version
docker info
# Validate the NVIDIA runtime (actual command may differ by runtime version):
docker run --rm --gpus all <VERIFIED_CUDA_IMAGE_DIGEST> nvidia-smi
# Then validate the pinned research app image:
docker run --rm --gpus all <IMMUTABLE_RESEARCH_IMAGE_DIGEST> \
  python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Inside the research image: run BF16 matrix smoke, model artifact verification, edge/full single real COCO request, edge vision-only and cloud text-only inference; inspect memory. Reproducibility is *same image software*, not guaranteed identical GPU kernels across different GPU models.

### 5.3 Image release flow

`build → dependency/security tests → pinned model smoke → immutable registry push → digest verification on both VMs → deploy`.

Do not use `latest` for experiments. Never pull a mutable branch during an active campaign. Registry credentials must be short-lived/scoped where possible.

**Gate P3:** Both VMs run the exact image digest, CUDA/BF16/model paths work, persisted mounts and restart services work, no secret leaks, and containerized request outputs match the established frozen contract or documented GPU-specific numerical differences.

---

## 6. Stage P4 — Portability refactor: configurations, runner, manifests

### 6.1 Configs and naming

**Never modify** `configs/profiles/physical_network_blocks_v1.json` or `configs/experiments/physical_vqa_v1.yaml` in place. Create:

```text
configs/hardware/vultr_edge_v1.yaml
configs/hardware/vultr_cloud_v1.yaml
configs/experiments/vultr_vqa_v1.yaml
configs/profiles/vultr_network_blocks_v1.json
configs/evaluation/vultr_vqa_split_v1.json   # only if needed; same frozen IDs
```

The machine inventories must say **verified** only after live discovery (`nvidia-smi`, OS, CPU, RAM, GPU vGPU profile, driver, CUDA, local disk, actual interface/private IP, network route). Inventories are immutable within a campaign. No assumed A40/A16 plan IDs; replace after provider API quote and preflight.

### 6.2 Generic profiling runner

Extract shared logic from `scripts/run_physical_profile_block.py` to a provider-neutral module (preferred):

- Accept `--campaign-id`, `--host-pair-id`, `--network-source`, `--grid`, `--edge-interface`, `--cloud-host`, `--config`, `--split`, `--output-root`, `--checkpoint-backend`, `--image-digest`.
- **Remove WSL-specific qdisc shape assumptions**; inspect `tc` actual format and NIC name on Vultr. Match configuration and achieved traffic, not hard-coded first-line text only.
- Preserve deterministic policy schedule, one warm-up + three repetitions, per-policy process separation, clean Git checks, protocol/manifest hash checks, `--resume` and `profile-runs` validation.
- Identify edge and cloud GPU devices via actual runtime instead of assuming `cuda:0` always maps identically; require explicit recorded mapping.
- Verify the cloud-side qdisc via agent or authenticated health/probe endpoint; *edge verification alone does not prove both sides were shaped*. Record both outputs and timestamps.
- Add provider-specific environment metadata as **new fields** rather than changing existing physical profile semantics.
- Account for `nvidia-smi` telemetry being unavailable on some vGPU configurations; use PyTorch memory, process telemetry and log the null reason, while making hardware identity mandatory.
- Keep existing `adaptive-vlm benchmark --mode ...` CLI and output directory schema wherever possible; do **not** fork the benchmark logic to a second runner with different timestamps/measurement definitions.
- Do not include model weight downloads or Docker startup time in steady-state per-request inference timing. Cold start is a separately reported metric.
- Include exact container digest, OS/driver, Terraform revision, region, instance/plan IDs, vGPU profile, host pair, model, dataset, split/grid hashes and seed in each run manifest.

### 6.3 Reproducibility tests

- Run all existing CPU unit tests on the new branch.
- Run fake-adapter whole scheduling test offline with unchanged hashes.
- Run `adaptive-vlm config validate` on all newly introduced YAMLs.
- Run 12-request correctness pair comparison on new hardware before any full profiling.
- Reject resuming a job if provider/host pair, model, image, dataset, split, schedule or network grid changed. A replaced instance can be accepted only when its immutable **logical role and hardware contract** are compatible, and the replacement is documented with a fresh host identity.

**Gate P4:** Provider-neutral runner preserves existing physical mode, produces Vultr-specific provenance, rejects incompatible resumes, and passes tests and smoke comparisons.

---
## 7. Stage P5 — Checkpointing and independent durable storage

### 7.1 Storage decision

**Preferred for minimal migration:** keep the existing S3 checkpoint implementation and AWS S3 bucket if available. An AWS **EC2 GPU quota** refusal does not, by itself, imply an S3 account restriction. Using AWS S3 for artifacts while renting GPU VMs from Vultr is technically possible, but measure transfer charges and the availability of credentials and endpoints. Keep AWS state/artifacts buckets independent of the Vultr teardown.

**Vultr-only alternative:** Vultr Object Storage is S3-compatible and documents bucket object versions, bucket lifecycle and multipart uploads. **Not every S3 API capability is identical**. Validate conditional-object uploads, checksums, versioning, list consistency and required REST semantics using a disposable test bucket. Do not silently reuse an AWS S3-specific atomic-write assumption that Vultr does not implement. Vendor's `vultr_object_storage` Terraform resource creates an object-storage subscription; arrange bucket initialization and settings through tested declarative S3 management, and document any bootstrap command not directly expressible in Terraform.

**Choose and freeze a checkpoint backend before the full campaign.** Storage is a control-plane service, not part of inference-link bandwidth. The benchmark remains runnable if the user's laptop disconnects.

### 7.2 Existing local durability contract

The existing benchmark already writes a unique run directory containing `plan.json`, `run_manifest.json`, an fsynced `requests.jsonl` and `summary.json`; it supports strict `--resume`. **Reuse this system.** Every measured outcome (success, OOM, timeout, error) gets an immutable composite trial key:

```text
(campaign_id, network_block_id, policy, request_id, repetition_index)
```

If a process dies while a request is in flight, label the attempt as interrupted/unknown until recovery; never mark it as successful just because a remote server may have processed it. A failed *completed* attempt stays in the denominator; reruns for diagnostics are separate explicitly labeled runs.

### 7.3 Immutable remote checkpoint layout

```text
<bucket>/research/adaptive-vlm/
  campaigns/vultr-v1/gate2/<run_id>/
    contract/
      plan.json
      hashes.json                # Git, model, dataset, split, grid, Docker, Terraform
      node-inventory-edge.json
      node-inventory-cloud.json
    N1/
      edge/
        segments/000001-<sha256>.jsonl.zst
        segments/000002-<sha256>.jsonl.zst
        commits/000001.json
        commits/000002.json
      cloud/segments/...           # same structure
      distributed/segments/...
      network_pre.json
      network_post.json
      profile/summary.json
      validation_report.json
    ...
    N9/...
    final/
      aggregate.json
      gate2-complete.json         # only after final science/coverage validation
```

Checkpoint cadence: **25 newly completed trial rows OR 5 minutes**, whichever is first, plus at policy/block end, on graceful shutdown, and after validating a block. Upload a bounded number of immutable objects per batch; prefer background uploads between inference calls to avoid contaminating the latency measurement. Record upload start/end timestamps, duration, bytes, failures and last **committed** checkpoint; checkpoint intervals are targets and do not guarantee recovery of the last unuploaded trials.

### 7.4 Two-phase commit / integrity

1. Write completed trial locally using append + flush + `fsync` and stable keys.
2. Seal a local segment, hash SHA-256, compress, upload to a **new** immutable object key. Read back or check contents/hashes using supported S3-compatible APIs.
3. Write a **new immutable commit manifest** only after **all** listed segment objects are confirmed. A manifest includes:
   - campaign/run/block/policy IDs, sequence number and previous commit hash;
   - exact request/repetition keys and status counts;
   - Git SHA, clean status, Docker digest, Terraform revision, host pair/VM GPU profile;
   - model/artifact, dataset manifest, split, network grid, schedule and plan hashes;
   - network evidence digests, segment keys, segment SHA-256, compressed byte counts, UTC commit timestamp.
4. Optionally publish a mutable `latest` pointer **after** manifest success; recovery can also list and verify immutable commit manifests when the pointer is stale. Enforce one active writer through an actual lease/lock and reject concurrent campaign writers.
5. On recovery, select the latest **fully validated** manifest, verify every referenced object and trial key, reconstruct the runner's original files, check the unchanged campaign contract, independently reapply/measure the current network shaping, and continue using existing `--resume`.
6. If provider storage lacks trustworthy compare-and-swap/conditional write semantics, use unique commit keys, a single authoritative writer, and a tested external lock/lease. Do not claim exactly-once remote state without testing concurrent writers and crash boundaries.

### 7.5 Mandatory failure-injection tests

- [ ] Kill the benchmark container mid-block using `SIGKILL`; restore from last committed checkpoint; check no duplicate authoritative keys.
- [ ] Kill during object upload; incomplete segment must **not** be considered committed.
- [ ] Temporarily block object-store access; checkpoint component must alert and stop or hold after bounded grace; it must **not** silently proceed unprotected indefinitely.
- [ ] Supply corrupted segment/incorrect SHA; recovery refuses it.
- [ ] Change model hash or Docker digest; strict resumed campaign refuses the mismatch.
- [ ] Destroy a disposable GPU VM through Terraform; recreate it with the *same approved hardware contract*, rehydrate cache/dataset and restore from object storage alone.
- [ ] Recover a resume that was interrupted between final trial and `summary.json` creation; rebuild derived summaries without rerunning successful measured keys.
- [ ] Test rollback when remote checkpoint has older state than local fsynced journal; reconcile by verified keys and preserve provenance, never overwrite newer valid data without review.
- [ ] Test a scheduled upload **after** the timed inference region; compare latency distributions with uploads enabled/disabled on a small pilot to detect timing interference.

**Gate P5:** Remote checkpoint recovery passes all crash and VM-recreation tests, immutable manifests verify end-to-end, and remote recovery never requires a surviving old GPU instance or university device.

---

## 8. Stage P6 — Network topology, controlled shaping and validation

### 8.1 Freeze a new Vultr network grid

Do not change the conceptual nine-block protocol. Copy the block values from the physical grid, replacing only provider/route-specific settings after measurement:

| Block | Configured rate, per direction | Added egress delay on EACH endpoint | Expected **added** full RTT |
|---|---:|---:|---:|
| N1 | 2 Mbps | 10 ms | 20 ms |
| N2 | 5 Mbps | 10 ms | 20 ms |
| N3 | 10 Mbps | 10 ms | 20 ms |
| N4 | 25 Mbps | 10 ms | 20 ms |
| N5 | 50 Mbps | 10 ms | 20 ms |
| N6 | 100 Mbps | 10 ms | 20 ms |
| N7 | 10 Mbps | 2.5 ms | 5 ms |
| N8 | 10 Mbps | 25 ms | 50 ms |
| N9 | 10 Mbps | 50 ms | 100 ms |

**Important:** The configured added RTT is **not the full measured RTT**. Actual RTT is approximately `measured unshaped baseline RTT + edge added egress delay + cloud added egress delay`, plus processing/jitter. For N7 the target is baseline + 5 ms, **not an absolute 5 ms**. Verify each block from the actual VPC route and do not guess the baseline.

Actual link throughput must be enough to achieve N6 100 Mbps both directions. If VLAN/VPC tenancy, GPU plan link speed or background traffic prevents an achieved rate, **stop** and freeze a formally versioned reduced/alternative network grid **before** research profiling; never change N6 mid-campaign while still calling results the original grid.

### 8.2 Select and verify experiment interfaces

On both VMs:

```bash
ip -br address
ip route
ip route get <PEER_PRIVATE_IP>
ethtool <EXPERIMENT_INTERFACE> || true
sudo tc qdisc show dev <EXPERIMENT_INTERFACE>
```

The `tc` interface must be the one carrying cloud↔edge packets on ports 5050/5051/5052. Prefer a **dedicated VPC interface**, with object storage/control using a distinct route. Validate using `tcpdump`/interface counters in a tiny test. Never shape an unspecified `eth0` by habit on a remote server. Save the exact route/interface proof in the environment report.

### 8.3 Minimal commands for ONE preflight block

```bash
# Example ONLY; run on each host's verified data NIC, never blindly on SSH NIC.
sudo tc qdisc replace dev <EXPERIMENT_INTERFACE> root netem \
  delay 10ms rate 2mbit
sudo tc -s qdisc show dev <EXPERIMENT_INTERFACE>

# From virtual edge, with cloud probe daemons bound on its private IP:
iperf3 -c <CLOUD_PRIVATE_IP> -p 5051 -t 8 -O 1 -J > network_forward.json
iperf3 -c <CLOUD_PRIVATE_IP> -p 5051 -t 8 -O 1 -R -J > network_reverse.json
python scripts/tcp_rtt_probe.py client \
  --host <CLOUD_PRIVATE_IP> --port 5052 --count 10 --warmup 2 \
  > network_echo.json

# Cleanup AFTER completion: both hosts, on the verified experiment NIC.
sudo tc qdisc del dev <EXPERIMENT_INTERFACE> root
```

Use a privileged host-level network controller with an allowlist of only declared block settings, idempotent `apply`, `status`, `verify`, and `restore` actions, and independent proof from **both** hosts. If automatic cleanup fires after network loss, it must not delete essential host networking or lock out the operator.

### 8.4 Acceptance and evidence

- Achieved **receiver** goodput on both directions measured by `iperf3` before and after each block; the original runner accepts **80–120%** of rate target. Preserve raw JSON and retransmissions, not only averaged Mbps.
- Actual median full-path TCP echo RTT before and after block; original runner uses tolerance `max(5 ms, 20% of configured added RTT)` around the measured baseline + added RTT.
- Test that 25/5-minute checkpoint upload and SSH do **not** go through the shaped VPC inference data path. Use route/interface evidence and bandwidth comparison; if they do, modify routing/filtering.
- Independently verify block schedule and exact qdisc on **both** VMs, not just client.
- Include unshaped baseline (`N0`) measurement, timestamp, test duration, background traffic, network provider route metadata and repeat of at least N3 after N9.
- For extra realism, later use separately frozen *real measured 4G/5G trace replay* with rate, RTT, jitter, correlated loss and dropouts, using the same independent probe strategy. Never claim synthetic static netem blocks are physical wireless links.

**Gate P6:** All nine network blocks measurably achievable and repeatable at least in a pilot; route/probe/shaping control paths isolated; evidence shows both directions and actual RTT. No false claim of real radio behavior.

---

## 9. Stage P7 — Functional correctness + full GPU preflight

1. Confirm pinned `.model-cache` on each side with official artifact manifest verification; do not substitute a different Moondream revision, precision, quantization or token cap to make a chosen GPU fit.
2. Record idle GPU memory; in fresh processes measure peak `allocated`, `reserved`, available VRAM, steady-state residency for:
   - edge/full on virtual edge;
   - edge vision encoder + projector on virtual edge;
   - cloud/full on virtual cloud;
   - cloud text-only for distributed path on virtual cloud.
3. Test BF16 support including actual model execution, not merely `torch.cuda.is_available()`. For NVIDIA vGPU, confirm correct driver/profile/license status and usable CUDA compute.
4. Perform original `data/manifests/correctness.jsonl` **12-request, three-placement** correctness campaign inside containers. Compare token IDs, decoded outputs, prompt tokens, bytes and GPU kernel-dependent divergence.
5. If exact cross-placement equality fails, investigate host GPU/BF16 numerics separately from protocol/serialization correctness; record a documented *new* criterion if scientifically necessary. Do not falsely mark the old strict equality gate passed.
6. Test representative real VQA inputs from `physical_vqa_v1.jsonl` on all three paths. Ensure every COCO image SHA resolves; the checked-in manifest omits the large raw images.
7. Prove wall-clock watchdog and idle shutdown/teardown logic on disposable instance(s) before the long run. A two-hour per-boot runtime would **interrupt** a lengthy block unless jobs are subdivided and checkpointed; design the watchdog to stop safely at a verified checkpoint and destroy the VM, not kill an uncheckpointed workload.

**Gate P7:** All three functional inference modes tested on both selected Vultr GPUs, memory feasibility documented, output comparison validated or divergences predeclared, Docker/server preflight passed, and operating spend controls tested.

---

## 10. Stage P8 — Pilot the complete experimental workflow (cheap first)

Before the 8,100-attempt matrix, run a **separate diagnostic campaign**, never misrepresented as final Gate 2:

- One network block (N3) with 3–5 distinct frozen request IDs across all three fixed placements, one repetition each. Validate model/cache, network, Docker roles, request pairing, stage timing and profile generation.
- A second block contrasting N1 vs N6 to test the expected bandwidth effect on the **2,985,984-byte BF16 projected tensor** versus the typically smaller compressed source image. Do not assume split is faster: it transmits far more bytes than cloud-only on many images.
- Collect cold model-load time and per-request steady-state time separately.
- Run full pipeline with checkpoint uploader enabled, and execute at least one crash/restore before marking pilot complete.
- Estimate total runtime and expense from **measured pilot** edge/cloud/split runtimes plus warmups, model loads, network probes, checkpoint transfer, cold provisioning, registry/object-store charges and safety margin.
- Reconfirm with user any proposed new GPU size/plan or an expected increase above the approved cost cap.

**Gate P8:** A real Docker/Vultr end-to-end pilot demonstrates all dependencies, reproduces expected scientific measurement shapes, survives crash recovery, and yields a reviewed **revised cost budget**. Full campaign requires explicit operator approval.

---

## 11. Stage P9 — Execute the exact Gate 2 fixed-placement matrix

### 11.1 Immutable experiment declaration

Before collecting final research data, write a **frozen `vultr-v1` experiment declaration** containing:

```text
campaign_id: vultr-v1-gate2
request_count: 100
network_blocks: [N1,N2,N3,N4,N5,N6,N7,N8,N9]
placements: [edge,cloud,distributed]
repetitions: 3
warmups: 1 per policy/block
measured_attempts_expected: 8100
warmups_expected: 27
randomization_seed: 1707 (policy order), benchmark seed per frozen config
fixed_model: Moondream2 pinned model and verified model/artifact hashes
max_new_tokens: 32, BF16, greedy, batch=1
hardware: verified Vultr VM plans and vGPU profiles
provenance: Git SHA, Docker digest, Terraform revision/state change ID
checkpoint: independent object storage, 25 rows or 5 min, immutable commits
```

Maintain exact user research data contract:

- Training requests: 60 IDs; validation: 20 IDs; held-out evaluation: 20 IDs.
- Training network blocks: **N1, N3, N5, N6, N8**.
- Validation blocks: **N2, N7**.
- Final held-out blocks: **N4, N9**.
- SLA: **60 seconds**; frozen bootstrap seed **1707**, **2,000 resamples**, **95% confidence level** as in existing protocol.
- All blocks may be profiled to create the *prospective* matrix, but **G3/G4 calibration, thresholds and model selection must not inspect or fit the final held-out outcome rows**. Store access-control/review boundaries and seal held-out artifacts appropriately.

### 11.2 Execute by network block

For each block, in the frozen order or a predeclared seeded order:

1. Confirm cloud dual-path server ready on private IP, model fingerprint and pinned image digest.
2. Check current GPU headroom, thermals/utilization and no competing inference workload.
3. Apply network block settings to both role-specific egress interfaces.
4. Verify actual **bidirectional goodput** + full-path RTT; save raw preflight and qdiscs on both VMs.
5. Run policy order from existing deterministic `policy_order` rotation. Each policy executes **100 requests × 3 measured repetitions**, plus **one warmup** as in the checked runner. A warmup is excluded from the 8,100 measured denominator.
6. After **each attempt**, fsync result locally; every 25 attempts/5 minutes make immutable remote checkpoint commit; also checkpoint at policy transitions.
7. Log every failure/OOM/timeout. Do not silently skip failure rows to improve latency statistics. If an entire block is corrupted by network mismatch, archive/label invalid and rerun under a distinct campaign revision; do not selectively delete slow trials.
8. Measure network again after block, preserving independent goodput and RTT files. Unshape both hosts.
9. Validate expected row key counts, network evidence hashes, GPU-memory evidence, timings/negative residuals, per-policy completeness, model equivalence and network state.
10. Publish a verified block-level remote commit, and optionally teardown/recreate GPU instances between sessions **only after** checkpoint and restoration readiness are verified.

One block: `100 × 3 × 3 = 900 measured attempts + 3 warmups`.

### 11.3 Example agent-facing wrapper (proposed, NOT yet existing)

```bash
# From the virtual edge, after the trusted Vultr network agent configures BOTH peers:
bash scripts/vultr/run_block.sh N1

# OR (preferred consistent naming) the refactored existing Python runner:
python scripts/run_profile_block.py \
  --campaign-id vultr-v1-gate2 \
  --block N1 \
  --config configs/experiments/vultr_vqa_v1.yaml \
  --grid configs/profiles/vultr_network_blocks_v1.json \
  --split configs/evaluation/physical_vqa_split_v1.json \
  --output-root artifacts/phase2/vultr-v1 \
  --checkpoint-backend s3-compatible
```

**These are proposed interfaces:** the agent must implement them and align actual CLI options with tested code. The existing command `python scripts/run_physical_profile_block.py --block N1` is the concrete reference to refactor, not something to replace wholesale.

### 11.4 Artifacts to archive for each block

```text
artifacts/phase2/vultr-v1/N1/
  plan.json
  environment.json               # both VM inventories + identity
  docker_and_terraform.json      # digests/versions
  network_pre.json
  network_post.json
  network_evidence.json
  edge/requests.jsonl
  edge/run_manifest.json
  edge/summary.json
  cloud/requests.jsonl
  cloud/run_manifest.json
  cloud/summary.json
  distributed/requests.jsonl
  distributed/run_manifest.json
  distributed/summary.json
  edge_gpu_samples.csv
  cloud_gpu_samples.csv
  profile_specification.json
  profile/profile.jsonl
  profile/summary.json
  validation_report.json
  checkpoint_index.json
```

For measurements, retain timestamped per-trial E2E latency, actual model stage timings, serialized feature size and wire bytes, transport time and residual, prompt/output token counts, response token hash, GPU memory and utilization, achieved pre/post network measurements, failure status and cause, and immutable provenance.

### 11.5 Gate 2 completion criteria

- [ ] Exactly **8,100 planned measured trial keys** accounted for, plus **27** excluded warmups; zero uninvestigated missing keys or duplicate authoritative keys.
- [ ] All nine N-blocks have independently validated pre/post network measurements in BOTH directions.
- [ ] Three fixed placements compared on the same frozen request IDs/repetition schedule, with seeded policy ordering documented.
- [ ] All failed attempts retained; completed-but-failed attempts stay in reliability denominator and are excluded appropriately from successful-latency medians.
- [ ] For valid successful measurements, profile schema checks pass, timings have consistent units/definitions, and any negative transport residuals are investigated rather than clipped.
- [ ] Exact image/model/dataset/split, container digest and hardware provenance verified.
- [ ] GPU memory feasibility, throttling and numerical divergence documented; no invented equality or VQA task score.
- [ ] Official VQA evaluation used for public-dataset accuracy claims; repository's provisional in-run consensus score alone is **not** a publication claim.
- [ ] Remote immutable final artifact archive and independent hash audit complete; at least one full S3-compatible restore produces byte-matching profile data.
- [ ] Separate final `gate2-complete.json` marker issued **only** when all these checks pass.

**Gate P9 / G2:** Complete, immutable, validated, provider-labeled profiling dataset ready for G3 calibration.

---

## 12. Stage P10 — Cost guard, alarms, unattended operation, and teardown

### 12.1 Design two independent cost controls

**A. Hard operational controls (prevent runaway spending):**

- Set deployment guard `deploy_gpu_instances=false` by default in Terraform.
- Require per-run approval on GPU IDs, estimated maximum duration, total cost ceiling and network block count.
- Add host `systemd` timer / local supervisor that requests safe checkpoint and stops measured scheduling after the allowed wall-clock budget; do not use a blind kill that can corrupt a segment.
- Add a **separate** control-plane automation (outside the GPU VMs) that detects expired `expires_at` metadata and invokes the tested Terraform disable/destroy procedure even if SSH/benchmark dies. A local laptop cron alone is **not** sufficient if the laptop is disconnected; use an appropriately authenticated CI runner/small durable supervisor/hosted schedule, and keep secret scope minimal.
- Limit concurrent GPU VM count to exactly two and require a fresh approved plan for scaling/modifying types.
- Test auto-teardown on a disposable one-node session. After teardown independently query Vultr API; zero active **campaign-tagged GPU VMs** must remain.

**B. Budget alerts (helpful but not a hard cap):**

- Estimate accrual from actual VM create/destroy timestamps and price quotes, plus registry, storage, block storage, snapshots and transfer. Compare with provider usage/billing APIs when available.
- Notify a verified email/webhook at configurable percentage thresholds (e.g. 50%, 75%, 90%, 100%). Where Vultr lacks native AWS-Budgets-equivalent enforcement for the selected product, implement a **self-hosted budget watchdog** and disclose that estimates/notifications are not guaranteed real-time hard caps.
- Persist watcher decisions/estimated spend independent of GPU VM and log alert-delivery success/failure. The provider bill is authoritative.
- At risk threshold, stop scheduling new work, verify latest checkpoint, invoke approved teardown path; require explicit new owner authorization for resumption if ceiling reached.

### 12.2 Vultr billing consequence

Vultr's documented server billing continues after stopping an instance. **Destroy** the instance to cease compute charges. Deleting an instance may also delete its local disk; therefore persistent object storage/checkpoints are not optional. Explicitly track any retained bootable block volumes, snapshots, registry subscriptions, object-storage subscriptions, reserved IPs, and other separately billed services. A GPU VM created by Terraform is replaceable; benchmark data must not be.

### 12.3 Emergency and planned shutdown runbook

```text
1. Pause experiment queue; do not launch new measured trials.
2. Finish or mark in-flight trial correctly; fsync requests.jsonl.
3. Create verified immutable remote checkpoint + update status.
4. Stop Docker benchmark/dual-path server and unshape experiment NICs.
5. Review Terraform plan with deploy_gpu_instances=false.
6. Apply, then query Vultr API to verify GPU VM destruction.
7. Leave only explicitly approved persistent S3-compatible bucket/registry/state.
8. Record spend-to-date estimate, resource inventory, last commit SHA, resume command.
```

**Gate P10:** The user can disconnect for a full session without corrupting results, spend is bounded by effective teardown automation rather than a merely advisory email, and no unintended GPU compute charges continue between approved sessions.

---

## 13. Stage P11 — Resume research after Gate 2 without physical hardware

1. **G3 predictor:** Fit physical-performance model only on the frozen **training request IDs and training N-blocks**, evaluate on validation split, seal final evaluation results. Recalibrate **from Vultr profiles**; do not reuse a predictor trained on localhost or RTX 3050/2070 measurements. Quantify APE, p95, intervals, fastest-choice ranking and comparisons against a constant-placement/constant-median predictor.
2. **G4 selector:** Compare optimal measured oracle vs predictor choice under memory feasibility and 60-s SLA; measure regret, top-choice agreement, SLA violations and uncertainty/fallback behavior.
3. **G5 adaptive execution:** Use the *same two Vultr VMs* and the single Docker image, with bounded live state monitoring and adaptive selection. Include selector overhead and cost of probes; no future network states as inputs.
4. **G6 dynamic traces:** Replay measured 4G/5G traces with bandwidth, RTT, jitter/loss/dropouts on controlled VPC link. Use unseen traces/task IDs and predeclared fairness/control baselines. Do not claim actual wireless radio performance.
5. **Final thesis analysis:** Include edge-only/cloud-only/fixed-split/adaptive, paired E2E latency p50/p95, throughput and reliability, bytes transferred, CPU/GPU memory, network state, quality and uncertainty. Explain cloud-hosted edge emulation and missing power/physical hardware validity explicitly.
6. **Optional post-thesis:** Repeat representative results on real edge+cloud hardware as a new host-pair experiment if hardware becomes available.

**Thesis novelty caution:** Your current three policies are **full edge / full cloud / vision-encoder–projector edge + cloud LLM**. The implemented `distributed` policy is one internal split, not arbitrary runtime partitioning at every transformer layer. Do not claim a general per-layer dynamic split algorithm unless the code actually implements and evaluates it. Your publishable claim should match the supported execution configurations and actual scheduler logic.

---

## 14. Work breakdown for the agent — exact order and commits

| Step | Suggested commit | Key deliverable | Gate |
|---|---|---|---|
| 0 | `docs(vultr): freeze branch baseline and plan` | Decision log, dependency inventory, baseline tests | P0 |
| 1 | `feat(vultr): add plan and quota preflight` | Authenticated GPU plan/region/price verifier; no launch | P1 |
| 2 | `feat(vultr): add Terraform foundations` | Provider, remote state, VPC, SSH, firewall, registry/storage modules | P2 preflight |
| 3 | `feat(vultr): add disposable gpu nodes and teardown` | GPU VMs opt-in, `deploy_gpu_instances=false`, dry run/cleanup | P2 |
| 4 | `feat(containers): reuse immutable gpu image across providers` | Dockerfile, Compose, NVIDIA driver/runtime smoke | P3 |
| 5 | `refactor(profiling): make host/network provenance portable` | New Vultr configs, generalized runner, regression tests | P4 |
| 6 | `feat(checkpoints): support S3-compatible durable resumes` | Incremental immutable segments, commit manifests and restore tests | P5 |
| 7 | `feat(network): validate Vultr private link and netem` | Dedicated route, 9-block grid, probe automation | P6 |
| 8 | `test(vultr): document gpu and correctness preflight` | Hardware + BF16 + 12-request comparisons | P7 |
| 9 | `test(vultr): prove small profiling pilot` | 3-policy pilot, failure injection, corrected cost estimate | P8 |
| 10 | `feat(ops): protect budget and enforce teardown` | Off-VM cost guard, explicit alerts, destroy verification | P10 |
| 11 | `research(vultr): freeze and run gate2` | 8,100-attempt campaign + verified checkpoint archive | P9/G2 |
| 12 | `docs(vultr): archive reproducible handoff` | Commands, results, validation, limitations, spend, next steps | Final |

**Review cadence:** Stop at every `BLOCKED` or failed gate. Report actual error, reason and suggested resolution. Run CI on each module. Code review before provisioning network/security changes and before full paid GPU campaign. Do not ask to approve steps that are purely local non-paid validation; do ask for paid resource creation, account charges and destructive cleanup of retained data.

### Agent must produce a status report after every phase

```markdown
## Vultr migration status — <phase>
- Branch / commit:
- Implemented paths:
- Terraform state / plan location (no secrets):
- Docker image digest:
- Selected GPU plans / verified region / quoted hourly cost:
- Tests and exact commands executed:
- Evidence paths / checksums:
- Cost incurred / remaining approved budget (estimate):
- Current gate: PASS | FAIL | BLOCKED
- Unresolved risks:
- Next exact step:
```

---

## 15. Definition of DONE

The **Vultr backup environment is READY** when:

- [ ] Actual provider capacity/permissions and quoted pair cost verified.
- [ ] Terraform can provision **and destroy** both GPU instances and preserve independent data/state, without UI provisioning.
- [ ] Both roles execute the same verified immutable Docker image with working CUDA/BF16 and pinned Moondream model.
- [ ] Private inference path, controlled `tc` shaping and independent `iperf3`/RTT measurements pass; admin/backup paths remain reachable.
- [ ] Cloud GPU memory and correctness checks pass or well-supported exceptions are documented.
- [ ] S3-compatible checkpointing survives container kill and instance destruction with exact plan/hash enforcement.
- [ ] All nine N-blocks are independently measured and a full small pilot proves profiler semantics.
- [ ] User has approved the full Gate 2 spend and the cost guard/automatic destroy path has been tested.

The **Vultr Gate 2 research is DONE** when:

- [ ] All **8,100 planned measured trial keys**, including failed outcomes, and 27 warmups are accounted for.
- [ ] Every block's network, model, dataset, GPU, Docker, Terraform and raw-result hashes are linked to an immutable archive.
- [ ] `profile-runs`, comparisons and independent audit pass; official VQA evaluation used where relevant.
- [ ] Results are clearly labeled **Vultr virtual-edge/cloud**, not physical edge results.
- [ ] GPU VMs are destroyed after safe final checkpoint, and no unapproved paid resources remain.
- [ ] A fresh machine can restore archived evidence, run validation, and regenerate the summary without hidden local files.

---

## 16. Source and implementation references (checked 7 November 2026)

These are implementation references, not promises about your account's current access or a specific region's available capacity.

1. [Vultr Cloud GPU provisioning](https://docs.vultr.com/products/compute/instances/cloud-gpu/provisioning) — fractional/full GPU VMs, Terraform provisioning, GPU-enabled images and bootable volumes.
2. [Vultr GPU variants](https://docs.vultr.com/products/compute/instances/cloud-gpu/explore-gpu-variants) and [Cloud GPU overview](https://docs.vultr.com/product-overviews/cloud-gpu) — A16/A40/A100/L40S hardware classes; marketing specifications are not live account inventory.
3. [Vultr Terraform provider](https://registry.terraform.io/providers/vultr/vultr/latest/docs), [`vultr_instance`](https://docs.vultr.com/reference/terraform/resources/instance), [`vultr_vpc`](https://docs.vultr.com/reference/terraform/resources/vpc), [`vultr_object_storage`](https://docs.vultr.com/reference/terraform/resources/object_storage), and [`vultr_container_registry`](https://docs.vultr.com/reference/terraform/resources/container_registry).
4. [Vultr plan API example](https://docs.vultr.com/how-to-provision-cloud-infrastructure-on-vultr-using-terraform) and [Vultr CLI region availability](https://docs.vultr.com/reference/vultr-cli/regions/availability) — discover current plan IDs/regions; confirm account entitlement.
5. [Vultr private VPC on Cloud GPU](https://docs.vultr.com/products/compute/instances/cloud-gpu/networking/vpc) — independent VMs with private connectivity.
6. [Vultr NVIDIA GPU-enabled images](https://docs.vultr.com/products/compute/instances/cloud-gpu/gpu-enabled-images), [Docker installation warning](https://docs.vultr.com/how-to-install-docker-on-ubuntu-2204), and [vGPU management](https://docs.vultr.com/how-to-manage-vgpu-on-vultr-cloud-gpu-instances) — driver/CUDA/GPU licensing/installation cautions.
7. [Vultr Object Storage S3 compatibility matrix](https://docs.vultr.com/products/storage/object-storage/s3-compatibility-matrix) — supports bucket versions/multipart uploads but not all AWS S3 features; test your precise checkpoint operations.
8. [Vultr GPU server billing](https://docs.vultr.com/support/platform/billing/how-am-i-billed-for-my-servers) — **stopped GPU VMs continue to incur hourly charges**; destroy to stop compute billing.
9. Original repository: [README](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/README.md), [physical profiling runner](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/scripts/run_physical_profile_block.py), [network grid](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/configs/profiles/physical_network_blocks_v1.json), [VQA split](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/configs/evaluation/physical_vqa_split_v1.json), [prior cloud migration roadmap](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/docs/Gate2_Cloud_Migration_Roadmap_v2.md).

---

## 17. Copy-paste agent kickoff directive

> Implement the Vultr backup roadmap in `docs/vultr_backup_agent_roadmap.md` in the existing `feature/vultr-cloud-backup` branch (create it from the correct clean baseline only if it does not exist). Inspect the current repository and identify any usable provider-neutral Docker/checkpoint code already present in the AWS branch; reuse it without importing EC2-specific infrastructure. Begin with P0 and P1. Do not assume a specific Vultr GPU plan/region has capacity. Before any paid GPU launch, report verified account plan IDs, GPU VRAM/compute/driver compatibility, proposed Terraform plan, and full preflight spend, and get explicit authorization. Implement reproducible Terraform infrastructure, one immutable Docker image for both roles, S3-compatible immutable checkpointing with full VM-recreation recovery, a distinct controlled network data path, and a provider-neutral profiling runner. Keep the exact frozen three placements, nine network blocks, 100 VQA requests, three repetitions, 27 warmups, model hashes and holdouts. Do not run or claim Gate 2 finished until all 8,100 attempts and scientific validation gates are complete. Commit logical changes, run tests, provide evidence and gate reports after each phase, protect AWS/main, and destroy unused Vultr GPU VMs after checkpoint verification because powered-off VMs still accrue charges.
