# Shared container contract v1

Build context is the repository root: `docker build -f container/Dockerfile .`.
Dockerfile-specific ignore rules exclude Git, credentials, caches, images and
results. A reviewed linux/amd64 Python 3.11 base **must** be supplied as
`--build-arg BASE_IMAGE=repository@sha256:...`. It must contain iperf3, setuptools
and wheel; Python's pinned torch wheels supply CUDA 12.8 user-space libraries.
No invented or untested base digest is committed. Archive the selected base
manifest, OS packages and GPU compatibility evidence before freezing the build.

Supply GIT_SHA from `git rev-parse HEAD`, MODEL_CONTRACT_SHA from SHA256 of
`configs/model/moondream2.yaml`, and LOCK_SHA from the existing model lockfile.
Build fails for missing/malformed provenance or lock/installed-version mismatch.
The existing lock is version pinned but lacks wheel hashes: it is not a fully
hermetic supply-chain lock. Archive wheel artifacts/hashes for the final campaign.
Package build tooling and OS libraries are supplied by the digest-pinned base.

Push the image to the Terraform ECR repository using short-lived builder identity;
instance roles can pull but cannot push. Set APP_IMAGE to its ECR digest, never
a mutable tag. Use identical APP_IMAGE on both nodes. Compose interpolation
requires APP_IMAGE but the operator must verify the digest syntax against ECR.
Set MODEL_DIR, DATA_DIR, ARTIFACT_DIR to persistent absolute paths. DATA_DIR must
contain the existing `manifests/` and `raw/` structure; mount path semantics are
preserved under `/app/data`. Models and tokenizer artifacts must be verified and
seeded offline in MODEL_DIR; server uses `--cache-dir /models --offline`.

Model containers run UID 10001, drop capabilities, use a read-only root filesystem
and host networking. No Docker socket or NET_ADMIN. Only GPU inference services
reserve GPU access. Logs rotate at 10 MB × 5. Inference health checks prove TCP
readiness only; they are not model correctness checks. The shared image runs
inference, iperf3 and echo through separate commands. Probe services restart on
failure; independently verify both ports before profiling.

`container/cloud-vlm.service` manages cloud Compose; install it under systemd and
set its EnvironmentFile with nonsecret paths/digest only. The host watchdog stops
this unit. The edge service remains a diagnostic CLI until the resumable cloud
runner is implemented; do not schedule a long experiment through it yet.
Containers receive SIGTERM with a 120-second grace period, but durable application
checkpoint handling must be added with the later checkpoint manager.

Preflight requires nvidia-smi on host and container, CUDA BF16 matrix operation,
verified model load, full edge inference, isolated vision inference and all three
placement correctness paths. Record allocated/reserved/free memory, drivers,
instance/AZ, base/app digest and hashes. Recreate containers and reboot hosts to
verify cache/data retention. Test watchdog shutdown before long jobs. Do not mark
Gate D0 complete from CPU-only configuration checks.

Keep host traffic shaping peer/port-classified to 5050–5052. S3/SSM endpoint
traffic must bypass shaping. Host networking alone does not prove the selected
interface, tc behavior or achieved bandwidth/RTT; measure from the final image.
