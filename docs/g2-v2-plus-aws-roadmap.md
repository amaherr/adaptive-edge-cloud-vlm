# G2 v2+ on AWS: roadmap

**Status: plan, 17 November 2026; account inventory added the same day (section 8).** The GPU quota appeal for the AWS pair was granted. This roadmap moves the G2 v2+ campaign ([plan, B0–B9](g2-v2-plus-plan.md)) from the physical RTX 3050 / RTX 2070 SUPER pair to two AWS VMs, built with the Terraform and container work on the `cloud-infrastructure` branch. It **reverses the earlier "AWS/Vultr out of scope" decision** recorded in the Predictor V2 specification; record that reversal in the protocol (amendment A2).

The physical results are unaffected: G2 and its analysis stay as they are, as the real-hardware baseline. The physical G2 v2 campaign is not run; its frozen prompt, long task, dataset, splits and tooling carry over unchanged.

## 1. Target environment

| | Virtual edge | Cloud |
| --- | --- | --- |
| Instance | `g6f.2xlarge`: ¼ NVIDIA L4 (time-sliced vGPU, ~5.6 GiB usable), 8 vCPU, 32 GiB; fallback `g4dn.xlarge` (full T4, 4 vCPU). `g6f.4xlarge` needs 16 vCPU and doesn't fit the 16-vCPU quota next to the cloud (section 8) | `g6.xlarge`: full L4, 24 GiB, 4 vCPU, 16 GiB (protocol A2a-2); `g6.2xlarge` only if phase D's CPU headroom guard fails |
| Driver | GRID (required on G6f), baked into a reviewed AMI | Tesla/datacenter driver, baked into a reviewed AMI |
| Software | One digest-pinned image from ECR, run as the edge and cloud roles | Same image |
| Data | Encrypted data EBS volume per host, holding the model cache, images and manifests, seeded from S3 | Same |
| Network | Same VPC and AZ; experiment ports 5050–5052 open only from the edge security group; **no inbound from the internet**; outbound through an auto-assigned public IPv4 instead of interface endpoints (D9) | Same |
| Management | SSM Session Manager and `ssm send-command` over outbound HTTPS; S3 through the free gateway endpoint | Same |
| Shaping | `tc` on the **host**, classified by peer IP and ports 5050–5052, so management, S3 and SSM bypass netem | Same, mirrored |

Moondream2's BF16 weights are about 3.85 GB. The ¼-L4 edge may only fit the full model with **swap residency**, which we already built and verified for the 4 GiB RTX 3050. Phase D decides residency from measurements.

This is *real distributed inference on two independent VMs, with a resource-constrained virtual edge and emulated network conditions*. It is not a physical edge device, and the thesis must say so. In exchange it gives a representative capability gap, with the cloud much stronger than the edge. The physical pair had that inverted: its Turing "cloud" was slower than the edge at vision and prefill.

## 2. What carries over and what changes

| Area | Status |
| --- | --- |
| Prompt P1, long task E1, dataset, splits, network grid conditions | **Unchanged** (protocol sections 3–5 stay frozen) |
| Interleaved runner, supervisor, sub-block gates, references, derived tables, predictor and oracle pipeline | Reused; adapted in phase A |
| Terraform, AMI recipe, container contract, watchdog, image publish and smoke scripts, classified shaping (branch `cloud-infrastructure`) | Reused; ported to `main`'s layout in phase A |
| Edge-load levels, residency, spill gate, winner estimate inputs, references, int8 quality, cloud-load level, run estimate | **Redone on AWS** (phases D–E) |
| Windows-specific code (PowerShell GPU counters, WSL NVML workaround, keepalive, portproxy) | Kept for the physical setup; switched off by a platform setting on AWS |

## 3. Decisions

Recommendations first; each becomes part of amendment A2.

| # | Decision | Recommendation |
| --- | --- | --- |
| D1 | Region and AZ | The AZ where the quota was granted and both instance types are offered (`account-preflight` checks this) |
| D2 | Edge instance | `g6f.2xlarge` if it passes the stability and fit criteria in phase D; otherwise `g4dn.xlarge`. `g6f.4xlarge` only if the quota is raised. Criteria are in A2a rule 10 |
| D3 | Remote control from the supervisor | An **SSM runner**: `ssm send-command` to the cloud host, behind the same `Runner` interface as SSH. It keeps the no-SSH, no-public-IP design, and the edge role only needs `ssm:SendCommand` on the cloud instance. The alternative, intra-VPC SSH on port 22 from the edge security group only, is simpler but adds keys and a port |
| D4 | Where the supervisor runs | In an **edge orchestrator container** with host networking and `NET_ADMIN`, scoped to the edge role. This is a revision of the container contract. It launches the benchmark as a subprocess in the same container (GPU attached), and shapes the edge host's interface directly |
| D5 | "Same source on both hosts" check | Compare the **image digest and its `org.opencontainers.image.revision` label** on both hosts instead of `git status`. The image is built from a clean commit |
| D6 | Checkpoints | Each validated sub-block (and the campaign plan, status and logs) is copied to S3 with SHA-256 receipts as soon as it validates. The data EBS volume remains the primary copy |
| D7 | Budget | **A $180 ceiling for everything on AWS, including G5 later.** Alerts at $60, $110 and $150. An AWS Budgets action **stops all project instances at $165** (a Budgets action stops instances; it is not a billing cap) |
| D9 | Network egress | **No interface endpoints**: they cost $0.07/h even with compute off, about $51 a month. Instead, an auto-assigned public IPv4 per instance ($0.005/h, only while running) with **no inbound rules**, plus the free S3 gateway endpoint. SSM, ECR and S3 keep working. The trade-off is that the hosts can reach the internet outbound, which the old design forbade; installation from the internet remains forbidden by procedure |
| D10 | Paying only for what runs | Start only the VM a step needs (section 6). Stop instances whenever idle (watchdog idle stop after 15 min). **No spot**: the spot G/VT quota is 0, and a request isn't worth the delay for the hours involved. Everything runs on demand |
| D11 | Storage | Small volumes: a 30 GiB root and a 20 GiB data volume per host (100 GiB in all, about $8 a month), against the old 180 GiB. Scratch artifacts go to S3, and nothing large stays on EBS after archiving |
| D8 | Watchdog ceilings | `experiment_max_runtime_hours` must cover one campaign session (about 14 h). Idle stop stays on; the supervisor writes the watchdog heartbeat on both hosts |

## 4. Phases

```
A (port + platform code) ─┬─────────────────────────────┐
B-code (B0a,B1,B2,B4,B5,B7,B8) ─┤                        │
C (Terraform, AMIs, image, data) ─ D (acceptance) ─ E (recalibration) ─ F (freeze, refs, pilot, run) ─ G (after)
```

A, B-code and C can run in parallel. D needs C, plus enough of A to run the image. E needs D. F needs everything.

### Phase A: port and platform code (no GPUs; about 3 days)

**Status (17 November 2026): done** on branch `feature/aws-phase-a`. Tested locally with fakes; not yet run on AWS.
- **A1:** infrastructure ported (`infra/`, `container/`, `scripts/cloud/`, `docs/cloud/`).
- **A2:** `campaign/remote.py` (SSM, SSH and local runners).
- **A3:** `ClassifiedNetwork`.
- **A4:** the `platform` setting (spill metric: Windows dedicated or NVML used memory; an L0 CUDA OOM invalidates a sub-block).
- **A5:** image build record as source identity; same image digest on both hosts; `campaign/checkpoint.py` (S3 sync with receipts, edge and cloud heartbeats).
- **A6:** `orchestrator` service; `adaptive-vlm-inference` container name; `iproute2` and `awscli` in the image; [runtime contract v2](cloud/container_runtime_contract_v2.md); `configs/campaigns/g2v2_aws.example.json`.
- **A7:** tests.
- **Also:** the correctness smoke now runs a few **training IDs of the campaign manifest** through the campaign config instead of the fixture config. The old version would have failed preflight: its fixture IDs aren't in the v2 reference, and it pointed at the physical cloud's IP.
- **Still open:** the AWS grid (`configs/profiles/aws_network_grid_v2plus.json`) is created in B6.

1. **A1 Port the branch.**
   - Bring `infra/`, `container/`, `.github/workflows/cloud-infrastructure.yml` and the cloud scripts (into `scripts/cloud/`) onto `main`.
   - Bring its docs into `docs/cloud/`, marked as historical where they describe the 8,100-attempt v1 campaign.
   - Re-implement its small `src` changes (`activity.py` heartbeat, `shutdown.py`, runner and benchmark hooks) against the current runner instead of merging them.
   - Drop the superseded `cloud_vqa_v1` campaign files.
2. **A2 Runner abstraction.** Add `campaign/remote.py` with `SshRunner`, `SsmRunner` (send, poll, stdout and exit code, timeout) and `LocalRunner`. Supervisor, network and cloud load take a runner, not an SSH target.
3. **A3 Classified shaping.** Port the branch's peer- and port-classified netem into `campaign/network.py` as a `ClassifiedNetem` mode: a prio qdisc with filters on the peer IP and ports 5050–5052, feeding a netem child. Drop counters come from the netem child. Keep root netem for the physical pair.
4. **A4 Platform setting** (`platform: physical-wsl | aws-linux` in the settings):
   - AWS spill gate: NVML used memory below the usable limit, and no CUDA OOM, instead of the Windows dedicated counter;
   - no PowerShell sampler;
   - no WSL keepalive;
   - Linux `MemAvailable` for the RAM checks.
5. **A5 Source identity, checkpoints and watchdog:** D5 image identity in preflight; D6 S3 sync of validated sub-blocks; supervisor heartbeat for the host watchdog (D8).
6. **A6 Container:**
   - add `iproute2`, `iperf3` and the AWS CLI/SSM client to the image;
   - define the edge orchestrator service (host network, `NET_ADMIN`, GPU) and the cloud services (adaptive server with `--report-resource-state`, the iperf3 and echo probes);
   - update the runtime contract (D4).
7. **A7 Tests.** Fake SSM runner, classified-shaping command generation and drop parsing, the platform switch, image-identity preflight, and S3 sync with a stub.

*Done when* the loopback rehearsal passes in `aws-linux` platform mode with fake runners, and the physical mode is unchanged.

### B-code: configuration space, int8, prompt-first, cloud load (no GPUs; about 5 days, alongside A)
Follow [g2-v2-plus-plan.md](g2-v2-plus-plan.md) for B1, B2, B4, B5, B7 and B8, with these changes:
- **B0 is split.**
  - **B0a** (now, before any measurement): configuration space, int8 encoding and quality rule, C1 procedure and target, scheduler inputs, and the **edge-instance acceptance criteria (D2)**.
  - **B0b** (after phase E): AWS hardware contract, calibrated levels, cells, counts and run estimate.
- **B5 cloud load** uses the A2 runner. On AWS, the worker runs **inside the cloud container** (`docker exec` through SSM, or a dedicated `cloud-load` service), so the WSL `--detached` and NVML workarounds aren't needed there, but keep them for the physical pair.
- **B3 int8 quality** moves to phase E, because it runs on the AWS edge GPU.

### Phase C: infrastructure (about 1–1.5 days, mostly waiting on applies and builds)
1. **C1 Account preflight and inventory.** Run `account-preflight`: quota (on-demand **and spot**), both instance types offered in the chosen AZ, current prices.
   - **First, before anything else:** list what still exists from 5 November (endpoints, volumes, ECR, buckets, budgets) and **destroy whatever is accruing cost and not needed**. Seven forgotten interface endpoints alone cost about $1.68 a day.
2. **C2 Foundation.** Refresh the `cloud-v1` plan:
   - set the budget, alerts and stop action (D7), and the ceilings (D8);
   - remove the interface endpoints, keep the S3 gateway endpoint, and enable public IPv4 with no inbound rules (D9);
   - shrink the volumes (D11);
   - add the IAM permission for D3 and the ECR pull;
   - review the plan, then apply.
3. **C3 AMIs.** Rebuild the host images: the GRID driver for G6f, the datacenter driver for G6, Docker, the NVIDIA Container Toolkit, SSM agent, `iproute2` and `iperf3`. Record the driver and OS versions as provenance.
4. **C4 Image.** Build from a clean commit, push to ECR, and record the digest.
5. **C5 Data.** Upload the model cache, the 180 COCO images and the manifests to the artifact bucket with SHA-256 receipts, then seed both data volumes through SSM. The hosts have no internet.
6. **C6 Compute.** Set `enable_compute` and `launch_approved`, review the plan, then apply. Export the outputs to `artifacts/infra/terraform_outputs.json`.

*Done when* both instances are running the pinned image digest, the data volumes verify against the receipts, and the watchdog is live.

### Phase D: acceptance on AWS (about 1 day; both VMs)
1. **D1 Nodes:** `nvidia-smi`, a BF16 matrix operation and the model load on both nodes. On the edge, test **resident and swap residency**, and record peak memory and whether each fits.
2. **D2 Paths:** edge/full, cloud/raw, distributed sync, plus prompt-first and int8 once B lands; outputs and wire bytes checked.
3. **D3 Edge stability (decides D2).** Run 100 edge/full requests three times at different times of day, edge otherwise idle. Pass if the coefficient of variation of the per-run median is at most 3%, the within-run p90/median is at most 1.15, and the model fits. Otherwise move to the next edge instance and record a new environment revision.
4. **D4 Network:**
   - unshaped iperf3 and echo baseline;
   - classified shaping at one grid condition, checking that rate and RTT land within tolerance, the drop counters work, and SSM, S3 and SSH traffic stay unshaped;
   - the 200 Mb/s check.
5. **D5 Resource state.** Check which NVML fields the GRID vGPU exposes (clocks, power and utilization may be missing), then adapt the edge-state input list in B8 to what exists.
6. **D6 Fault drill on AWS:**
   - kill the supervisor, the edge load worker and the cloud worker;
   - stop and start the edge instance in the middle of a sub-block;
   - resume, with no gaps, no duplicates and no stray load.

### Phase E: recalibration on AWS (about 1.5 days)
1. **E1 Residency**, from D1. Use resident if it fits with margin; otherwise swap.
2. **E2 Edge load L1/L2**, using the W5 procedure on the chosen edge. Targets are re-derived for a strong cloud:
   - L1: edge/full median about equal to the cloud/raw median at 35/20;
   - L2: about 1.5× that;
   - each reproduced within 2% over three runs.
3. **E3 Cloud load C1** (B5 procedure, now off site): the smallest duty giving 1.4–1.6× the cloud server time.
4. **E4 int8 quality (B3)** on the edge VM, under rule B0a.
5. **E5 Stage profile and winner estimate.** Run a short profile (20 requests per configuration, at L0/L1/L2 and C0/C1, unshaped). Feed it into `estimate_g2v2_winners.py` with AWS inputs, then confirm or revise the long-task and C1 cells so each split keeps winner changes. Revising draft sections is allowed; sections 3–4 stay frozen.

### Phase F: freeze, references, pilot, run
1. **B6** grids and pilot; **B0b** amendment A2 with the AWS contract, calibrated values, cells, counts and run estimate.
2. **B9** tests and the fault rehearsal (loopback, plus the D6 drill rerun on AWS).
3. **Freeze:** commit, build and push the image, and deploy the digest on both hosts.
4. **References:** per task, the same-device run on both VMs, the synchronous and int8 controls, then the bundles.
5. **Pilot:** go/no-go per the plan's section 6, then freeze the protocol and pin its hash.
6. **Run:** two sessions, about 11–12 h each. Stop the instances between sessions (EBS keeps the state). Validated sub-blocks go to S3 as they finish.
7. **Teardown:** stop compute. Keep the data volumes, the artifact bucket and ECR until the analysis is archived, then destroy the disposable stack.

### Phase G: after the run
Derive the tables, then archive under `results/aws-g2v2plus-<date>/` with `SHA256SUMS` and the S3 receipts. Then G3 (Predictor V2 r2) and G4 (selector against the oracle). **G5, the dynamic evaluation, also runs on this AWS pair**: no site visit, the same images, and conditions switched during the run.

## 5. Schedule

| Week | Work |
| --- | --- |
| 1 | A1–A7 and B0a, B1, B2, B4, B5 (code), in parallel; C1–C5 (infrastructure) as applies allow |
| 2 | C6, D1–D6, E1–E5; B7, B8; B6 and B0b |
| 3 | B9, freeze, references, pilot, run (two sessions), teardown |

That is about 12–14 working days, with the critical path through A → D → E → F. The run itself is 2–3 calendar days of wall time.

## 6. Cost: a $180 ceiling

On-demand prices from the 5 November review, to be confirmed in C1: `g6f.2xlarge` $0.475/h, `g6.xlarge` $0.805/h, so **$1.28/h for the pair**. Spot is typically 50–70% lower where available.

**Which VM each step needs:**

| Step | Edge | Cloud | Hours (estimate) |
| --- | :---: | :---: | ---: |
| D1–D2 fit and paths | ✓ | ✓ | 3 |
| D3 edge stability (3 runs, 2 h apart; stop the edge in between) | ✓ | — | 3 |
| D4 network, D6 fault drill | ✓ | ✓ | 3 |
| E2 edge load | ✓ | — | 3 |
| E3 cloud load (client runs on the cloud host) | — | ✓ | 1.5 |
| E4 int8 quality (single process on the edge) | ✓ | — | 2 |
| E5 stage profile | ✓ | ✓ | 1.5 |
| References: same-device per host, then the controls | ✓ / ✓ | ✓ | 2 + 1 + 1.5 |
| Pilot | ✓ | ✓ | 2 |
| Campaign, two sessions | ✓ | ✓ | 24 |
| Debugging and reruns allowance | ✓ | ✓ | 12 |

**Estimate (on demand, all of G2 v2+):**

| Item | Cost |
| --- | ---: |
| Pair-hours (~50 h × $1.28) | ~$64 |
| Edge-only hours (~10 h × $0.475) | ~$5 |
| Cloud-only hours (~2.5 h × $0.81) | ~$2 |
| Public IPv4 (~110 instance-hours × $0.005) | ~$0.6 |
| EBS, 100 GiB gp3 for ~5 weeks | ~$10 |
| S3, ECR and CloudWatch | ~$4 |
| **G2 v2+ total** | **~$86** |
| Reserved for G5, the dynamic evaluation (~24 h pair plus setup) | ~$50 |
| Contingency | ~$44 |
| **Ceiling** | **$180** |


**Rules that keep it under the ceiling:**
- Phases A and B are built and tested **locally** (free); AWS only starts at C.
- Nothing runs while a person isn't working or a job isn't active: the watchdog stops idle instances after 15 min, and the supervisor's heartbeat keeps them up during runs.
- Stop instances between sessions, and destroy the disposable stack as soon as the run is archived. EBS snapshots are not kept.
- Check the spend on the console before every phase. If actual spend exceeds the phase's line in this table by 30%, pause and replan before continuing.

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| ¼-L4 timing noise from the time-sliced vGPU | D3 criteria, decided in advance; fallback edge instances |
| Model doesn't fit in about 5.6 GiB | Swap residency (already built); then `g6f.4xlarge` |
| GRID vGPU hides NVML fields | D5; the edge-state input list adapts; nothing gates on unavailable fields |
| SSM command latency or limits during a run | Runner timeouts and retries; per-sub-block calls only (shape, check, load start and stop), never per request |
| Shaping hits management traffic | Classified netem (A3); the D4 check that SSM, S3 and SSH bypass it |
| Instance retired or interrupted mid-run | On-demand instances (not spot); resume from EBS and the S3 checkpoints |
| Cost overrun | Budget alerts, the idle watchdog, stopping between sessions, teardown checklist |
| Results not comparable with physical G2 | Expected: G2 v2+ is a new baseline. Report physical G2 as the real-hardware reference, and the hardware ratio explicitly |

## 8. Account inventory (17 November 2026, read-only)

| Item | Found | Cost | Action |
| --- | --- | --- | --- |
| EC2 instances, volumes, snapshots, AMIs, Elastic IPs, NAT gateways | none | — | — |
| VPC `10.42.0.0/16` and its S3 gateway endpoint | present | free | keep (reused by C2) |
| Interface endpoints | none | — | — |
| S3: state bucket and artifact bucket (a few KB of old Vultr acceptance records) | present | cents | keep |
| ECR `adaptive-vlm-cloud-v1`: two copies of a 4.0 GB image from 5 November | **deleted 17 November** (repository kept, now empty) | was ~$0.80/month | — |
| Budget `adaptive-vlm-cloud-v1` ($20) and SNS alerts | present | free | replace with the $180 budget and stop action in C2 |
| Old Vultr watchdog: Lambda every 5 min, EventBridge rule, CodeBuild teardown project, 2 alarms, SNS topic, SSM SecureString holding a **Vultr API key** | **deleted 17 November** | was ~$0.20/month | revoke the key in the Vultr account |
| Vultr leftovers not deleted: IAM roles `adaptive-vlm-vultr-{image-pull,teardown,watchdog}`, two log groups (~0.6 MB), ~2,900 KB-sized acceptance records under `research/adaptive-vlm/vultr-v1/` in the artifact bucket | present | free or cents | optional cleanup |
| On-demand G/VT vCPU quota | **16** | — | `g6f.2xlarge` (8) + `g6.xlarge` (4) uses 12 |
| Spot G/VT vCPU quota | **0** | — | no spot |
| Spend | October $0.00; 1–17 November **$0.20** (ECR $0.14, S3 $0.03, tax $0.03) | — | — |

**Cheaper cloud, adopted 17 November (protocol A2a-2).** `g6.xlarge` has the same full L4 with 4 vCPU and 16 GiB, at $0.805/h against $0.978/h. That saves about $9–11 and frees 4 vCPU of quota. The instance types are Terraform variables restricted to the protocol's allowed types (`edge_instance_type`, `cloud_instance_type`).
