# Shared container contract v1

Build context is the repository root: `docker build -f container/Dockerfile .`.
Dockerfile-specific ignore rules exclude Git, credentials, caches, images and
results. A reviewed linux/amd64 Python 3.11 base **must** be supplied as
`--build-arg BASE_IMAGE=repository@sha256:...`. The Dockerfile defaults to a resolved official Python 3.11 amd64 digest, installs
iperf3 and pinned build tools, and records OS packages. PyTorch wheels supply
CUDA 12.8 user libraries. The image build and GPU compatibility remain unverified.
Archive package/wheel artifacts and the final manifest before campaign freeze.

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
this unit. The edge service invokes the existing fixed-policy benchmark with explicit
manifest, placement and run ID. See cloud_acceptance_v1.md for lifecycle limits.
The S3-resumable multi-placement cloud scheduler remains a later phase.
Containers receive SIGTERM with a 180-second grace period. Fixed-policy SIGTERM handling now persists the active
attempt before stopping; S3 checkpoint handling remains a later phase.

Preflight requires nvidia-smi on host and container, CUDA BF16 matrix operation,
verified model load, full edge inference, isolated vision inference and all three
placement correctness paths. Record allocated/reserved/free memory, drivers,
instance/AZ, base/app digest and hashes. Recreate containers and reboot hosts to
verify cache/data retention. Test watchdog shutdown before long jobs. Do not mark
Gate D0 complete from CPU-only configuration checks.

Keep host traffic shaping peer/port-classified to 5050–5052. S3/SSM endpoint
traffic must bypass shaping. Host networking alone does not prove the selected
interface, tc behavior or achieved bandwidth/RTT; measure from the final image.

## Startup cache verification

Model inputs under /models are now mounted read-only. Only /models/modules
is overlaid with the writable persistent artifacts/runtime-cache/modules directory.
CUDA, Triton, Torch and XDG caches also live under artifacts/runtime-cache.
The volume helper creates those directories as UID/GID 10001, including the
nested mount target, and is idempotent on an already mounted data volume.

Before service startup, the host's standard-library validation script runs the
read-only container/verify_cache.py diagnostic inside the same application image.
It verifies weights, pinned tokenizer and all remote-code files without importing
or executing the model. The image's baseline verifier then checks all 100 frozen
dataset images and configuration hashes. Network is disabled for both checks;
no application packages are installed on the host. Host diagnostic source and
application-image revisions are separately recorded.

The actual shared image passed these checks through read-only model/data mounts
on 5 November 2026: 20 model artifacts and 100 images. This is cache/provenance
evidence, not GPU correctness or AWS persistence evidence. Startup hashing is
outside measured attempts; systemd allows ten minutes for validation/model load.
