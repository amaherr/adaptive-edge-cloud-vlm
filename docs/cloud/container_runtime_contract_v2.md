# Container runtime contract v2 (G2 v2+ on AWS)

This replaces [v1](container_runtime_contract_v1.md) for the G2 v2+ campaign ([AWS roadmap](../g2-v2-plus-aws-roadmap.md), decisions D3–D5). v1's rules stay in force unless they're changed here.

## Image

- One image, built by `scripts/cloud/build_cloud_image.py` from a **clean commit**, pushed to ECR and deployed **by digest** (`APP_IMAGE=repository@sha256:…`) on both hosts.
- The build checks `configs/cloud/image_provenance.json`: the model config, artifact manifest and dependency lock. It records the commit, base image and lock hash in `/usr/local/share/adaptive-vlm-build.json`.
- The image includes `src/`, `configs/`, `scripts/`, `docs/protocols/` (the grid pins the protocol's hash) and the tools `iperf3`, `iproute2` (`tc`) and `awscli`.
- Manifests and images are **not** in the image. They are on the data volume, mounted at `/app/data`, and `scripts/cloud/verify_cloud_baseline.py` checks them.

## Source identity

There is no `.git` in the image. The run manifests take the commit from the build record (`git.source: image`, `dirty: false`) and the image from `APP_IMAGE`. Campaign preflight requires `APP_IMAGE` to be a digest, and requires the cloud's `adaptive-vlm-inference` container (checked with `docker inspect` over SSM) to run the **same** image reference.

## Services

| Host | Service | User | Capabilities | Network | GPU |
| --- | --- | --- | --- | --- | --- |
| cloud | `inference` (container `adaptive-vlm-inference`): `adaptive-server --report-resource-state` | 10001 | none | host | yes |
| cloud | `iperf`, `echo` probes | 10001 | none | host | no |
| edge | `benchmark` (smoke and diagnostics) | 10001 | none | host | yes |
| edge | **`orchestrator`**: `run_g2v2_campaign.py run` | **0** | **`NET_ADMIN`, `DAC_OVERRIDE` only** | host | yes |

The orchestrator is the one exception to v1's "no `NET_ADMIN`":
- it shapes the edge interface with `tc` (classified netem, experiment traffic only);
- it writes into the 10001-owned artifact tree;
- it keeps a read-only root filesystem, `no-new-privileges` and a tmpfs `/tmp`;
- it has no Docker socket.

It starts the interleaved benchmark and the edge load worker as child processes, which exit with it.

## Cloud access from the orchestrator

All cloud control runs through SSM `AWS-RunShellScript` on the cloud **host**, as root:
- `tc` (no `sudo`);
- `docker inspect` and `docker exec adaptive-vlm-inference python …`;
- the cloud-load control (`docker exec adaptive-vlm-inference python -m adaptive_vlm.benchmark.cloud_load start|status|stop|reap`).
  - The C1 worker runs detached inside the inference container, as user 10001 on its GPU.
  - It writes its heartbeat to the container's tmpfs `/tmp/adaptive-vlm-cloud-load/heartbeat.json`.
  - The server reads that file (`--load-heartbeat`) and reports it with every response;
- heartbeat `touch`.

Commands are sent per sub-block or per heartbeat interval, never per request.

## Files on the data volume

| Path | Purpose |
| --- | --- |
| `/srv/adaptive-vlm/artifacts/job.heartbeat` | Watchdog heartbeat; touched by the orchestrator on the edge (as `/artifacts/job.heartbeat`) and through SSM on the cloud |
| `/artifacts/settings/g2v2_aws.json` | Campaign settings, from `configs/campaigns/g2v2_aws.example.json` |
| `/artifacts/g2v2plus/campaign/` | Campaign output; each valid sub-block is synced to S3 with a SHA-256 receipt |
