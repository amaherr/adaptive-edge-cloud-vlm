# Phase 1A / 1B acceptance record

An implementation, mocked plan or YAML parse is not a passed deployment gate.
Store real evidence under ignored artifacts/infra and archive it privately to S3.
Live foundation evidence is recorded in cloud_execution_2026-11-05.md.
GPU-dependent T0/D0 acceptance remains open while the applied quota is zero.

| Gate item | Current evidence | Still required |
|---|---|---|
| T0 remote state | Live encrypted/versioned backends and lock-contention test | Keep bootstrap, quota, retained and disposable states separate |
| T0 cost controls | Live $20 budget, four notifications, confirmed SNS, published delivery test | Test receipt confirmed; exercise host stop guard |
| T0 provisioning | Foundation deployed; paired AZ offerings verified; clean plan | AWS quota approval, baked/tested AMI and reviewed paired-node apply |
| T0 operations | Mount, telemetry, log agent and stop code | Confirm mounted EBS IDs, received logs/metrics, idle/max-runtime stop tests and no unintended drift |
| T0 teardown | Retained-state migration, reviewed disposable destroy plan, S3 receipt restore | Repeat the plan with actual GPU nodes before final T0 acceptance |
| D0 image | Successful pinned build, installed-lock check and offline CPU smoke | ECR digest publication and identical digest deployed on both GPU nodes |
| D0 GPU | Diagnostic script and pinned model | GRID/driver inventory, BF16 on both nodes, full edge and isolated vision feasibility |
| D0 correctness | Existing CPU suite; diagnostic smoke script | Actual 36 GPU attempts on 12 training requests, exact text/token outputs across all three placements |
| D0 timing | No cloud evidence | Host/container correctness comparison; declared numerical tolerances where applicable; separate orchestration overhead measurement |
| D0 lifecycle | SIGTERM/resume CPU test; persistent-mount code | Actual container recreate and host reboot retain data/cache/results; SIGTERM and timeout behavior tested |
| D0 network | Peer/port-specific helper and unit checks | Apply on both hosts; route/qdisc/filter counters and in-container forward/reverse throughput + RTT; S3/SSM unaffected |
| D0 isolation | Nonroot/drop-cap/read-only Compose | Inspect actual containers and SGs; confirm no privileged mode/NET_ADMIN/socket/public inference |

## Read-only account discovery

Run `scripts/cloud_aws_preflight.py --profile cloud-infrastructure --output
artifacts/infra/aws-preflight.json`. If using the Windows CLI from WSL, pass
`--aws-cli '/mnt/c/Program Files/Amazon/AWSCLIV2/aws.exe'`. This collects identity,
account-specific AZ offerings/IDs, GPU specifications, quota and pricing products;
it never creates infrastructure. Offerings do not guarantee current capacity.
The Windows login cache is not automatically a Linux Terraform credential source;
a supported credential bridge or Linux SSO session must be configured and tested
without writing keys to the repository. Fresh CLI login is currently required.

## Build and deploy

Enable Docker Desktop WSL integration, then run the build script from a clean
commit: `python scripts/build_cloud_image.py`. It resolves the base from the
Dockerfile, supplies provenance hashes, builds Linux/amd64 and records the local
image config digest. That digest is not an ECR deployment manifest digest.
Push to the Terraform ECR repository using temporary builder credentials, then
retrieve the ECR manifest digest. Use exactly that reference as APP_IMAGE on
both nodes. Never treat a tag or a self-reported environment variable as proof:
`validate_cloud_runtime.py` checks the pulled image RepoDigests and OCI labels.

Install the relevant systemd unit as cloud-vlm.service on each host and its
nonsecret environment file. Prestart validation requires EBS mounted at
/srv/adaptive-vlm and existing model/data/artifact directories under that mount.
The host AMI must include Python (stdlib tools only), AWS CLI v2, CloudWatch agent,
GRID/supported NVIDIA driver, NVIDIA Container Toolkit, Docker/Compose, SSM,
iproute2, curl, util-linux and ext4 tools. These dependencies remain an AMI
acceptance requirement; they have not been built or verified on AWS yet.

For diagnostics, override the edge entrypoint to run
`python scripts/cloud_container_smoke.py --output-root /artifacts/UNIQUE-ID` in
the shared GPU image. It selects the first 12 frozen *training* IDs, verifies
image hashes and runs actual edge/cloud/distributed paths with zero warmups and
one repetition. A separate diagnostic manifest uses absolute mounted paths; the
original manifest is untouched. Raw failures and exit codes are retained.
Exact text/token equality is the predeclared output criterion. Do not relax it
retrospectively to hide a discrepancy. Numerical tensor checks and host/container
comparison need their own archived evidence; this script does not claim them.

The edge Compose service now invokes the existing fixed-policy benchmark. Set
PLACEMENT, MANIFEST_PATH (container path) and a unique RUN_ID. For resume, override
the command with the same arguments plus --resume. This is not the later S3
checkpointed cloud campaign scheduler. No full campaign until Gate R0 passes.

## Operational limits

SIGTERM finishes the active fixed-policy inference, fsyncs its result and stops
before another attempt, returning status 143 when incomplete. Docker's grace
period is 180 seconds; forced termination after that can lose an active attempt.
Local journaling is not an S3 commit. SIGKILL, power-loss/torn-write recovery and
S3 snapshot integrity remain the subsequent checkpoint phase.

Activity files are touched at trial boundaries on edge and actual inference
requests on cloud; health checks do not count as work. The watchdog reads the
same artifact volume on the host. A long all-edge phase can leave cloud idle:
the future multi-placement campaign must coordinate an explicit bounded job
lease on both hosts. Do not disable the idle guard indefinitely to mask this.
The watchdog stops cloud-vlm.service and then powers down, preserving EBS.
Verify stop (rather than terminate) behavior before use. Uptime limits reset on
reboot and do not cap cumulative campaign spend.

Telemetry publishes idle seconds, filesystem occupancy and GPU utilization once
per minute. CloudWatch agent tails bootstrap and rotated Docker JSON logs. Test
actual delivery and IAM restrictions; emitting metrics is not evidence of receipt.
Missing telemetry alarms will also trigger when nodes are intentionally stopped;
disable their alarm actions during documented downtime, re-enable before jobs.
Checkpoint heartbeat/upload-failure alarms are reserved for checkpoint integration.

The host shaping helper derives the device from the peer route, permits only
N1-N9 and the frozen peer addresses, and classifies TCP 5050-5052 into netem.
Default traffic bypasses impairment. It refuses unrelated custom root qdiscs.
Install a reviewed root-owned copy; do not grant sudo to a user-writable script.
Capture --dry-run output and baseline qdiscs before applying. Reset removes this
helper's root qdisc and returns to the kernel default; it does not reconstruct
arbitrary previous custom configurations. Recheck the unshaped baseline after
reset. Root privilege remains on the host, never in model containers.

## Native WSL authentication bridge

The WSL AWS CLI now authenticates with profile cloud-infrastructure. The pinned
Terraform provider obtains temporary credentials through aws_credential_process.sh,
which clears the generated bridge config and inherited static credentials before
resolving the source profile. terraform_cloud.sh generates the ignored bridge
configuration automatically. No credential JSON is written to the repository.
Override CLOUD_AWS_SOURCE_PROFILE only when deliberately changing profiles.
The previous Windows credential-helper path is superseded.

Example diagnostic invocation after image/mount/GPU validation and cloud server
startup (set the required environment paths first):

```bash
export PLACEMENT=edge RUN_ID=d0-diagnostic
export MANIFEST_PATH=/app/data/manifests/physical_vqa_v1.jsonl
docker compose -f container/compose.edge.yaml run --rm --entrypoint python benchmark \
  scripts/cloud_container_smoke.py --output-root /artifacts/d0-UNIQUE-REVISION
```

The built image records OS packages, pip installation URLs/artifact hashes and
installed Python distributions under /usr/local/share. Preserve these together
with the ECR digest. Rebuilding from mutable Debian indexes need not be bitwise
identical; the final campaign reuses one immutable image, rather than rebuilding
independently on each node. Local image build has not yet been executed.

## Latest local checks

- Full CPU regression suite after runtime changes: 210 passed, one skipped due to
  sandbox loopback restrictions. Subsequent image-provenance changes passed all
  nine targeted benchmark/environment tests, including rejection of image changes
  during resume and recording image provenance per attempt.
- Ruff passes repository-wide; mypy passes all 64 application modules.
- Five Terraform mock tests pass: no initial GPUs, rejected unapproved launch,
  no initial hourly endpoints, exactly two approved nodes/seven endpoints, and
  both real cloud-init templates decode as valid YAML.
- Shell syntax and baseline hash verification pass. Host mount, watchdog, metric
  delivery and shaping still require AWS execution; mock tests do not prove them.
