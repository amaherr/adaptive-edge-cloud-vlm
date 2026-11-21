# AWS campaign runbook

How the two AWS campaigns were run, and how to run one again: G2 v2+ (the static baseline, 18–19 November 2026) and G5 (the live adaptive policy, 20–21 November 2026). Both use the same infrastructure, image, supervisor and session procedure; they differ in the grid and the settings file.

To **regenerate the thesis tables and figures** from the archived data you need none of this: see [Reproducing the results](../reproducibility.md). This runbook is for collecting new measurements.

> **State on 21 November 2026.** The disposable `cloud-v1` stack (VPC, instances, roles, watchdog) is destroyed. The retained stack still holds the artifact bucket, the ECR repository, the budget and the two 20 GB data volumes (`prevent_destroy`). A new run starts at step 1.

Related documents:
- [Container runtime contract](../design/container-runtime.md): the image, the services and their permissions.
- [G2 v2+ AWS roadmap](../g2-v2-plus-aws-roadmap.md): why the setup looks like this (decisions D1–D11) and the acceptance results.
- [Protocol](../protocols/g2-v2-protocol.md) and amendments [A3](../protocols/g2-v2-amendment-a3.md), [A4](../protocols/g2-v2-amendment-a4.md), [A4a](../protocols/g2-v2-amendment-a4a.md), [A4b](../protocols/g2-v2-amendment-a4b.md): what a valid campaign is.
- [Adaptive runtime](../design/adaptive-runtime.md): the G5 policy.

## The pair

| | Edge | Cloud |
| --- | --- | --- |
| Instance | `g6f.2xlarge`: ¼ NVIDIA L4 (L4-6Q vGPU, 6,144 MiB), 8 vCPU, 32 GiB | `g6.xlarge`: NVIDIA L4 (23,034 MiB), 4 vCPU, 16 GiB |
| Address | 10.42.1.10 | 10.42.1.20 |
| Role | Orchestrator, edge model (resident), in-process edge load, edge side of the shaping | `adaptive-server` (port 5050), iperf3 (5051), echo (5052), cloud load worker |
| Inventory | [`configs/hardware/aws_edge_g6f_2xlarge.yaml`](../../configs/hardware/aws_edge_g6f_2xlarge.yaml) | [`configs/hardware/aws_cloud_g6_xlarge.yaml`](../../configs/hardware/aws_cloud_g6_xlarge.yaml) |

Both run one AMI (Ubuntu 22.04, GRID driver 580.159.03, Docker, NVIDIA Container Toolkit, SSM agent) in us-east-1d, in one private subnet with no inbound rules. Management is SSM only; there is no SSH. The hosts reach S3 through the gateway endpoint and SSM and ECR through an auto-assigned public IPv4.

## Prerequisites (on the workstation)

- AWS CLI v2 with an SSO profile (the scripts use `cloud-infrastructure`; log in with `aws sso login --profile cloud-infrastructure`). Sessions expire: log in again before long steps.
- Terraform 1.10.5 (checked against HashiCorp's `SHA256SUMS`), always run through `scripts/cloud/terraform_cloud.sh`, which bridges the SSO login into the pinned provider.
- The repository's `.venv` (for `adaptive_vlm.campaign.remote.SsmRunner`, which the steps below use to run a shell script on a host: `SsmRunner(instance_id, region=…, profile=…)(["bash", "-c", script, …])`).
- `artifacts/infra/terraform_outputs.json`: the instance IDs, bucket and ECR repository, exported after each apply (`terraform output -json`). It is git-ignored; never commit plans, state or outputs, which hold the account ID.

**Budget.** The ceiling is $180 for everything (D7), with alerts at $60, $110 and $150 and a Budgets action that **stops both instances at $165**. The pair costs about $1.28/h on demand; the watchdog stops an instance after 15 minutes without a heartbeat. Stop the instances between sessions.

## 1. Infrastructure

Terraform lives in `infra/terraform/`, in four independent states:

| Stack | Holds | Notes |
| --- | --- | --- |
| [`bootstrap`](../../infra/terraform/bootstrap/README.md) | The Terraform state bucket | Protected; created once |
| [`account-preflight`](../../infra/terraform/account-preflight/README.md) | The G/VT on-demand vCPU quota request (16) | A quota is not capacity |
| [`environments/cloud-retained`](../../infra/terraform/environments/cloud-retained/README.md) | Artifact bucket, ECR repository, budget and alerts, data volumes | `prevent_destroy` |
| [`host-image`](../../infra/terraform/host-image/README.md) | The EC2 Image Builder recipe for the host AMI | Builds on demand |
| `environments/cloud-v1` | VPC, subnet, security groups, S3 gateway endpoint, IAM roles, both instances, watchdog, budget stop action | Disposable |

For every change: `plan -out`, **review the plan**, then apply that saved plan. From `infra/terraform/environments/cloud-v1`:

```bash
bash ../../../../scripts/cloud/terraform_cloud.sh plan -input=false -out=../../../../infra/local/compute.tfplan
bash ../../../../scripts/cloud/terraform_cloud.sh apply ../../../../infra/local/compute.tfplan
```

Copy `terraform.tfvars.example` to the ignored `terraform.tfvars` first. Compute starts only with `enable_compute = true` and `launch_approved = true`. Interface endpoints are refused (`enable_private_endpoints = false`, decision D9).

**Capacity.** `g6.xlarge` in us-east-1d often returns `InsufficientInstanceCapacity` on start. Retry, or create a one-instance capacity reservation in the instance's AZ, start into it, and cancel the reservation once the instance runs.

## 2. Image and data

1. **Source archive.** Commit, then upload `git archive` of the commit to `s3://<bucket>/cloud-v1/data/source/<commit>.tar.gz` with its `.sha256`.
2. **Build on the edge.** Debian's package mirrors need HTTP, which the hosts don't normally have. Open it only for the build: plan and apply with `-var build_http_egress=true`, build, then plan and apply with `false` again.
3. Run `scripts/cloud/bootstrap_host.sh` on each host through SSM, as root:
   - `bootstrap_host.sh edge <commit> <bucket> <commit-iso-date> <ecr-repository-url>` unpacks and checks the source archive into `/opt/adaptive-edge-cloud-vlm`, seeds `/srv/adaptive-vlm/data` from `s3://<bucket>/cloud-v1/data/dataset` (checked against its `SHA256SUMS`), fills the model cache through the adapter's pinned download, builds the image exactly as `build_cloud_image.py` does, verifies the cache with `container/verify_cache.py`, and pushes to ECR. It prints one JSON line with the image digest.
   - `bootstrap_host.sh cloud <commit> <bucket> <repository@sha256:…>` does the same but pulls the image by digest.
   - The 30 GB root holds one image plus one build: prune old images first, keeping the current digest.
4. **Write `/etc/adaptive-vlm.env`** on both hosts (`APP_IMAGE`, `MODEL_DIR`, `DATA_DIR`, `ARTIFACT_DIR`; see the [contract](../design/container-runtime.md#host-layout)).
5. **Start the cloud services:** `docker compose --env-file /etc/adaptive-vlm.env -f container/compose.cloud.yaml up -d --pull never`. Wait for the `"ready": true` line in `docker logs adaptive-vlm-inference`, with both `raw_image` and `projected_synchronous` in `supported_paths` and `reports_resource_state: true`.

Every session starts the same way: start both instances, keep the heartbeat fresh (a short `systemd-run` loop that touches `/srv/adaptive-vlm/artifacts/job.heartbeat` every minute), refresh the source to the frozen commit if needed, write the env file, and (cloud) start the services.

## 3. Correctness references

Preflight refuses to start without a same-device reference bundle built from the **frozen image**. Rebuild it whenever `src/` changes. `scripts/campaign/build_g2v2_reference.py` has four steps; run them inside the image (edge: `docker run --gpus all --network host --user 10001:10001` with the four mounts; cloud: `docker exec adaptive-vlm-inference python …`), with `--config configs/experiments/aws_vqa_v2.yaml --cache-dir /models` (long task: `aws_vqa_v2_long.yaml`, output under `reference-long/`):

| Where | Step | Output |
| --- | --- | --- |
| edge and cloud, in parallel | `runtime` | `<dir>/{edge,cloud}-runtime.json` |
| edge and cloud, in parallel | `same-device` | `<dir>/{edge,cloud}/` (120 short requests per GPU; about an hour on the edge) |
| — | copy the cloud's folder and runtime file to the edge through S3 | |
| edge (cloud service running, no shaping) | `control`, then `control --payload int8` | `<dir>/control/`, `<dir>/control-int8/` |
| edge | `bundle --output <dir>/bundle --edge … --cloud … --control … --int8-control … --edge-runtime … --cloud-runtime …` | `<dir>/bundle/index.json` |

`<dir>` is `/artifacts/g2v2/reference` (G2 v2+) or `/artifacts/g5/reference` (G5). The bundle step verifies everything exactly as preflight will; any same-device mismatch stops it. Its index path goes in the settings' `same_device_reference_evidence`.

## 4. Settings

Copy the example into `/artifacts/settings/` on the edge and fill in the placeholders (`REPLACE-cloud-instance-id`, `REPLACE-artifact-bucket`):

| Campaign | Example | Grid | Output |
| --- | --- | --- | --- |
| G2 v2+ | [`g2v2_aws.example.json`](../../configs/campaigns/g2v2_aws.example.json) | [`aws_network_grid_v2plus.json`](../../configs/profiles/aws_network_grid_v2plus.json): 196 sub-blocks, 43,200 attempts | `/artifacts/g2v2plus/campaign` |
| G2 v2+ pilot | [`g2v2_aws_pilot.example.json`](../../configs/campaigns/g2v2_aws_pilot.example.json) | [`aws_pilot_grid_v2plus.json`](../../configs/profiles/aws_pilot_grid_v2plus.json) | separate folder |
| G5 | [`g5_aws.example.json`](../../configs/campaigns/g5_aws.example.json) | [`aws_g5_traces.json`](../../configs/profiles/aws_g5_traces.json): 80 segments, 14,400 attempts | `/artifacts/g5/campaign` |
| G5 pilot | same example, grid and output changed | [`aws_g5_pilot.json`](../../configs/profiles/aws_g5_pilot.json) | `/artifacts/g5/pilot` |
| Fault drill | [`aws_drill.example.json`](../../configs/campaigns/aws_drill.example.json) | [`aws_drill_grid.json`](../../configs/profiles/aws_drill_grid.json) | never campaign data |

The main fields:

| Field | Meaning |
| --- | --- |
| `platform: aws-linux` | NVML spill gate instead of the Windows counter; no PowerShell sampler or WSL keepalive |
| `cloud_access` | `{"kind": "ssm", "instance_id": …}`: cloud control through SSM |
| `shaping: classified` | Netem on experiment traffic only (peer IP and ports 5050–5052) |
| `edge_residency: resident` | The model stays on the edge GPU (swap residency does not work on the GRID vGPU) |
| `checkpoint.s3_uri` | Every valid sub-block is copied here with a SHA-256 receipt |
| `heartbeat` | The watchdog heartbeat on both hosts |
| `max_restarts_per_subblock`, `max_subblock_attempts`, `max_failure_rate`, `stall_seconds` | Recovery limits (below) |

The grid pins the protocol's SHA-256 (`protocol_sha256`); preflight refuses a grid whose protocol doesn't match. The G5 grid also pins the frozen calibration, sensing and task-feature files under [`configs/g5/`](../../configs/g5/).

## 5. Rehearse, calibrate and pilot

Before any campaign, on the workstation:

```bash
.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q
.venv-model/bin/python scripts/campaign/rehearse_g2v2_faults.py --settings configs/campaigns/g5_loopback_rehearsal.json --fresh
```

The loopback rehearsal runs the full supervisor with fake hosts and injected faults (kill and resume, a killed load worker, a cloud worker killed mid sub-block); every check must be `true`. Use `g2v2plus_loopback_rehearsal.json` for the static grid.

On AWS:
- **G2 v2+ calibration (phase E, done once):** edge load L1, cloud load C2, int8 quality and the winner estimate. Results and commands: [phase E record](../../results/aws-phase-e-20261118/README.md).
- **G5 edge-probe calibration (A4, A4a):** inside the image on the edge, `scripts/campaign/calibrate_edge_probe.py --edge-load-config configs/edge_load/aws_levels_v1.json --level L1 --device cuda:0 --samples 60 --sensing-template configs/g5/sensing_template.json --output /artifacts/g5/edge_probe_calibration.json --sensing-output /artifacts/g5/sensing.json`. It writes the threshold into the sensing file and reports go or no-go.
- **Pilot:** run the pilot settings exactly like a campaign (step 6), then apply the go/no-go rules of the protocol (A2b rule 9 for G2 v2+, A4 section 5 for G5). A no-go needs a dated amendment before anything else is collected. Pilot data is never campaign data.
- **Freeze:** pin the image digest, the protocol and amendment hashes, the reference bundle and (G5) the sensing file. The G5 freeze record is [`results/g5-aws-20261120/freeze/g5_freeze.json`](../../results/g5-aws-20261120/freeze/g5_freeze.json).

## 6. Run a session

On the edge, from `/opt/adaptive-edge-cloud-vlm`:

```bash
. /etc/adaptive-vlm.env
CAMPAIGN_SETTINGS=/artifacts/settings/g5_aws_campaign.json docker compose --env-file /etc/adaptive-vlm.env \
  -f container/compose.orchestrator.yaml run -d --rm --name g5-campaign-run orchestrator \
  run --settings /artifacts/settings/g5_aws_campaign.json
```

Replace `run` with `preflight` (in the foreground, without `-d`) to check everything without measuring. `run` repeats the preflight on every start, including resumes:
- the pinned protocol hash and the same image digest on both hosts;
- the cloud ports, health and model fingerprint;
- free disk and GPU memory, the reference bundle and runtime versions;
- the GPU correctness smoke (`cloud_container_smoke.py`: 12 training requests through every configuration).

Then, per sub-block (or G5 segment), the supervisor shapes both hosts, measures the link (`network_pre.json`), starts the edge and cloud loads, runs the interleaved benchmark, stops the loads, measures again (`network_post.json`), clears the shaping, validates the sub-block and copies it to S3.

**Monitor** from the workstation, through SSM: `/artifacts/g5/campaign/status.json` (`state`, `stage`, `measured_attempts` out of `target_attempts`, `valid_subblocks`), the count of `subblocks/*/validation_report.json` with `"valid": true`, and `docker logs g5-campaign-run`. A heartbeat line appears every 30 s.

**End a session:** `docker stop -t 200 g5-campaign-run` (the supervisor stops its benchmark and loads and clears the shaping), check that no benchmark is left (`ps -ww`; torch processes are named `pt_main_thread`) and that the qdisc is back to default, then stop both instances. **Resume** by running the same command: valid sub-blocks are re-verified and skipped, and the interrupted sub-block is run again under the same plan.

The G2 v2+ campaign took three sessions (about 22 h); G5 took two (about 8 h).

## 7. Recovery

| Event | Automatic handling | What to do |
| --- | --- | --- |
| CUDA context failure on the edge | Benchmark exits 75; sub-block resumed, up to `max_restarts_per_subblock` | Nothing, unless restarts run out |
| Edge load died | Benchmark exits 76; resumed with a new generator | Nothing, unless restarts run out |
| Cloud load worker died | The sub-block fails its gate (load not alive in the responses), moves to `invalid/` and reruns | Nothing, unless it fails `max_subblock_attempts` times |
| Sub-block fails a gate (drops, probes out of tolerance, spill, failures above `max_failure_rate`) | Moved to `invalid/<id>-<time>` and rerun whole | Read `reasons` in its `validation_report.json` |
| No committed attempt for `stall_seconds` | Child process tree stopped; campaign fails | Check the cloud service and the link, then rerun |
| Orchestrator killed | Its children exit with the container; leftovers are reaped at the next start | Rerun |
| Instance stopped (watchdog, budget action, capacity) | Nothing runs; EBS keeps the state | Start, prepare, rerun |
| SSO session expired | SSM calls from the workstation fail; the campaign on the edge continues | Log in again |

Never delete lock files, `shaping_active.json`, journals or `invalid/` to force progress.

## 8. Campaign output

`/artifacts/<campaign>/campaign/` contains:
- `campaign_plan.json`: the frozen contract, including the sub-block order (G5: traces shuffled per repetition, segments in order). A resume with changed inputs is refused.
- `network_baseline.json`, `host_preflight.json` and `preflights/`.
- `subblocks/<cell>-r<rep>/` (G5: `<trace>-s<i>-<cell>-r<rep>/`), each with `plan.json`, `network_pre.json`, `network_post.json`, `qdisc_before.json`, `qdisc_after.json`, `run/` (`interleave.json` and one folder per placement with `plan.json`, `run_manifest.json`, `requests.jsonl`, `summary.json`), `restarts.jsonl` and `validation_report.json`.
- `invalid/`, `stray_processes.jsonl`, `status.json` and `campaign.log`.

## 9. After the run

1. **Copy the data volumes to S3** (`cloud-v1/{edge,cloud}/ebs-final/`), then **mirror the bucket** to `artifacts/aws-s3-mirror/` and verify every object by key, size and MD5. The [archive record](../../results/archive-20261121/README.md) shows the result for both campaigns.
2. **Derive and analyse:** the commands per table are in [Reproducing the results](../reproducibility.md).
3. **Commit the record** under `results/<campaign>-<date>/` with `SHA256SUMS`, and tag the campaign commit.
4. **Tear down:** plan `-destroy` in `cloud-v1`, review, apply. The retained stack refuses to delete the bucket, ECR and data volumes; remove those only after the thesis is accepted and the off-machine backup is verified.
