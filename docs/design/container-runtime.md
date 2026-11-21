# Container runtime contract

How the application runs on the AWS pair: one image, the services built from it, and what each may do. It applies to both AWS campaigns, G2 v2+ and G5. The procedure that uses it is the [AWS campaign runbook](../runbooks/aws-campaign.md); the infrastructure decisions behind it are listed there under [Design decisions](../runbooks/aws-campaign.md#design-decisions).

This page merges the two earlier contract versions. Version 1 (5 November 2026) was written for the abandoned `cloud-vqa-v1` campaign. Version 2 (17 November 2026) added the orchestrator and SSM control for G2 v2+. Every rule of version 1 that version 2 kept is included here.

## Image

- **One image for both roles.** It is built from a **clean commit** by `scripts/cloud/build_cloud_image.py` (or by `scripts/cloud/bootstrap_host.sh` on the edge host), pushed to ECR and deployed **by digest** (`APP_IMAGE=repository@sha256:…`), never by a mutable tag. Both hosts must run the same digest.
- **Base:** a digest-pinned linux/amd64 `python:3.11-slim-bookworm` (`BASE_IMAGE` build argument). The PyTorch wheels supply the CUDA 12.8 user libraries; the host supplies the driver.
- **Provenance checked at build time.** `container/verify_image.py` checks the inputs listed in [`configs/cloud/image_provenance.json`](../../configs/cloud/image_provenance.json) (model config, artifact manifest, dependency lock; regenerate it with `scripts/cloud/write_image_provenance.py`). The build arguments `GIT_SHA`, `BUILD_CREATED`, `MODEL_CONTRACT_SHA` and `LOCK_SHA` must be present and well formed, and the installed package versions must match the lock. The build record is written to `/usr/local/share/adaptive-vlm-build.json`, and the commit is also the `org.opencontainers.image.revision` label.
- **Contents:** `src/`, `configs/`, `scripts/`, `docs/protocols/` (the grids pin the protocol hash), plus `iperf3`, `iproute2` (`tc`) and the AWS CLI.
- **Not in the image:** model weights, manifests and images. They live on each host's data volume (below).
- **Supply-chain limit:** the lock pins versions but not wheel hashes, so the build is not hermetic. The image digest is what makes a campaign reproducible.

## Source identity

There is no `.git` in the image. Run manifests take the commit from the build record (`git.source: image`, `dirty: false`) and the image from `APP_IMAGE`. Campaign preflight requires `APP_IMAGE` to be a digest and requires the cloud's `adaptive-vlm-inference` container, inspected with `docker inspect` over SSM, to run the **same** image reference.

## Host layout

Each host mounts its encrypted data volume at `/srv/adaptive-vlm` (`infra/runtime/mount-data.sh`), and `/etc/adaptive-vlm.env` names the paths and the digest (no secrets):

| Variable | Host path | Mounted in the container as |
| --- | --- | --- |
| `APP_IMAGE` | — | the image digest |
| `MODEL_DIR` | `/srv/adaptive-vlm/models` | `/models`, **read-only** |
| `DATA_DIR` | `/srv/adaptive-vlm/data` (manifests and images, same layout as `data/`) | `/app/data`, read-only |
| `ARTIFACT_DIR` | `/srv/adaptive-vlm/artifacts` | `/artifacts` |

Only `/models/modules` is overlaid with the writable `artifacts/runtime-cache/modules` (remote-code modules). The CUDA, Triton, Torch and XDG caches also live under `artifacts/runtime-cache`. The volume helper creates these directories as UID/GID 10001 and is idempotent.

**Startup verification.** Before the services start, the image's `container/verify_cache.py` checks the weights, the pinned tokenizer and every remote-code file against the artifact manifest without importing the model, and `scripts/cloud/verify_cloud_baseline.py` checks the build inputs and the frozen manifests, images and splits. Both run with networking disabled. The server then loads with `--cache-dir /models --offline`.

## Services

| Host | Service | Compose file | User | Capabilities | GPU |
| --- | --- | --- | --- | --- | --- |
| cloud | `inference`, container `adaptive-vlm-inference`: `adaptive-server --report-resource-state --load-heartbeat …` on port 5050 | `compose.cloud.yaml` | 10001 | none | yes |
| cloud | `iperf` (iperf3 server, port 5051) and `echo` (`tcp_rtt_probe.py server`, port 5052) | `compose.cloud.yaml` | 10001 | none | no |
| edge | `benchmark`: smoke and diagnostic runs | `compose.edge.yaml` | 10001 | none | yes |
| edge | **`orchestrator`**: `scripts/campaign/run_g2v2_campaign.py run --settings …` | `compose.orchestrator.yaml` | **0** | **`NET_ADMIN`, `DAC_OVERRIDE` only** | yes |

Common to every service:
- read-only root filesystem, all capabilities dropped (except the orchestrator's two), `no-new-privileges`, no Docker socket;
- host networking, so the shaped interface is the host's own;
- JSON-file logs with rotation;
- `SIGTERM` with a 180 s grace period. The benchmark persists its active attempt before it exits.

**The orchestrator is the one exception to "no `NET_ADMIN`".** It shapes the edge interface with `tc` (classified netem on experiment traffic only), and `DAC_OVERRIDE` lets it write into the 10001-owned artifact tree. It keeps the read-only root, `no-new-privileges` and a tmpfs `/tmp`. The interleaved benchmark and the in-process edge load run as its children and exit with it. It lives in its own compose file so that starting it never needs the benchmark service's variables.

`container/cloud-vlm.service` and `container/edge-vlm.service` are systemd units that wrap the compose files. The host watchdog stops them.

## Cloud control from the orchestrator

All cloud control runs through SSM `AWS-RunShellScript` on the cloud **host**, as root. The edge role may send it only to the instance tagged `Role=cloud`:
- `tc` for the cloud side of the shaping (no `sudo` needed);
- `docker inspect` and `docker exec adaptive-vlm-inference python …`;
- the cloud load: `docker exec adaptive-vlm-inference python -m adaptive_vlm.benchmark.cloud_load start|status|stop|reap`. The worker runs detached inside the inference container, as user 10001, on the cloud GPU. It writes its heartbeat to the container's tmpfs `/tmp/adaptive-vlm-cloud-load/heartbeat.json`, which the server reads (`--load-heartbeat`) and reports in every response;
- the watchdog heartbeat `touch`.

Commands are sent per sub-block or per heartbeat interval, never per request. Root in an SSM shell cannot signal processes inside a container (AppArmor confinement), so every kill goes through `docker exec`.

## Shaping

Shaping is classified on the **host** interface: a prio qdisc with flower filters on the peer IP and ports 5050–5052 feeds a netem child, so SSM, S3 and ECR traffic bypass it. Host networking alone does not prove that the right interface is shaped or that the rate and RTT were achieved, so every sub-block measures them before and after (iperf3 and echo, from the final image).

## Files on the data volume

| Path (host / container) | Purpose |
| --- | --- |
| `/srv/adaptive-vlm/artifacts/job.heartbeat` / `/artifacts/job.heartbeat` | Watchdog heartbeat. The orchestrator touches it on the edge and through SSM on the cloud; without it the watchdog stops an idle instance after 15 minutes |
| `/artifacts/settings/*.json` | Campaign settings, made from `configs/campaigns/*_aws.example.json` |
| `/artifacts/g2v2plus/campaign/`, `/artifacts/g5/campaign/` | Campaign outputs; every valid sub-block is copied to S3 with a SHA-256 receipt |
| `/artifacts/g2v2/`, `/artifacts/g5/reference/` | Same-device reference bundles |
