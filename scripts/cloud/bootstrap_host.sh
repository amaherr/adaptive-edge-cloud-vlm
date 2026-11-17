#!/bin/bash
# Phase C4-C6 host bootstrap, sent through SSM (AWS-RunShellScript, as root).
#
#   bootstrap_host.sh edge  <commit> <bucket> <created-iso> <ecr-repository-url>
#   bootstrap_host.sh cloud <commit> <bucket> <image-ref repository@sha256:...>
#
# Both roles: keep the watchdog heartbeat fresh, seed /srv/adaptive-vlm/data
# from s3://<bucket>/cloud-v1/data/dataset (checked against SHA256SUMS), and
# unpack the committed source archive (checked against its .sha256) into
# /opt/adaptive-edge-cloud-vlm. The edge then builds the image from that
# source exactly as scripts/cloud/build_cloud_image.py does and pushes it to
# ECR; the cloud pulls it by digest. Each host fills its model cache through
# the adapter's pinned download (exact revision, SHA-256 checked) and checks
# it again with container/verify_cache.py. Prints one JSON line at the end.
set -euo pipefail
role=${1:?role}
commit=${2:?commit}
bucket=${3:?bucket}
[[ "$role" =~ ^(edge|cloud)$ ]] || { echo "unknown role $role" >&2; exit 2; }
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be a full SHA" >&2; exit 2; }
root=/srv/adaptive-vlm
data=s3://$bucket/cloud-v1/data
source=/opt/adaptive-edge-cloud-vlm
mountpoint -q "$root" || { echo "data volume is not mounted at $root" >&2; exit 1; }
beat() { touch "$root/artifacts/job.heartbeat"; }
beat
( while true; do beat; sleep 60; done ) &
keeper=$!
trap 'kill $keeper 2>/dev/null || true' EXIT

aws s3 sync "$data/dataset/" "$root/data/" --only-show-errors --region us-east-1
(cd "$root/data" && sha256sum --check --quiet --strict SHA256SUMS)
chown -R 10001:10001 "$root/data"

work=$(mktemp -d)
aws s3 cp "$data/source/$commit.tar.gz" "$work/" --only-show-errors --region us-east-1
aws s3 cp "$data/source/$commit.tar.gz.sha256" "$work/" --only-show-errors --region us-east-1
(cd "$work" && sha256sum --check --strict "$commit.tar.gz.sha256")
rm -rf "$source.new"
mkdir -p "$source.new"
tar -xzf "$work/$commit.tar.gz" -C "$source.new" --strip-components=1
rm -rf "$source"
mv "$source.new" "$source"
echo "$commit" > "$source/COMMIT"
rm -rf "$work"

if [ "$role" = edge ]; then
  created=${4:?created timestamp}
  repository=${5:?ECR repository URL}
  local_tag="adaptive-vlm:$commit"
  cd "$source"
  docker build --platform linux/amd64 -f container/Dockerfile --tag "$local_tag" \
    --build-arg "GIT_SHA=$commit" --build-arg "BUILD_CREATED=$created" \
    --build-arg "MODEL_CONTRACT_SHA=$(sha256sum configs/model/moondream2.yaml | cut -d' ' -f1)" \
    --build-arg "LOCK_SHA=$(sha256sum requirements/model-cu128.lock | cut -d' ' -f1)" \
    . > "$root/artifacts/image-build-$commit.log" 2>&1 || {
      tail -40 "$root/artifacts/image-build-$commit.log" >&2; exit 1; }
  registry=${repository%%/*}
  export DOCKER_CONFIG
  DOCKER_CONFIG=$(mktemp -d)
  aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin "$registry" > /dev/null
  docker tag "$local_tag" "$repository:$commit"
  docker push "$repository:$commit" > /dev/null
  # The registry digest Docker recorded for this push (no ECR read permission needed).
  image=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' \
    "$repository:$commit" | grep -m1 "^$repository@sha256:")
  rm -rf "$DOCKER_CONFIG"
  unset DOCKER_CONFIG
else
  image=${4:?image reference}
  registry=${image%%/*}
  export DOCKER_CONFIG
  DOCKER_CONFIG=$(mktemp -d)
  aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin "$registry" > /dev/null
  docker pull "$image" > /dev/null
  rm -rf "$DOCKER_CONFIG"
  unset DOCKER_CONFIG
fi
[[ "$image" =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ ]] || { echo "bad image $image" >&2; exit 1; }

# Model cache: the adapter's own pinned download, then an independent check.
docker run --rm --network host --user 10001:10001 \
  -e HF_HOME=/models -e HF_HUB_CACHE=/models/hub -e HF_MODULES_CACHE=/models/modules \
  -v "$root/models:/models" -v "$source/container/verify_cache.py:/tmp/verify_cache.py:ro" \
  --entrypoint python "$image" -c '
import subprocess, sys
import yaml
from huggingface_hub import hf_hub_download
from adaptive_vlm.config import load_experiment_config
from adaptive_vlm.models.moondream.adapter import MoondreamAdapter
config = load_experiment_config("configs/experiments/aws_vqa_v2.yaml")
MoondreamAdapter.from_pretrained(
    config.model, cache_dir="/models",
    artifact_manifest="configs/model/moondream2_artifact.yaml",
    vision_device="cpu", text_device="cpu", offline=False,
)
# The manifest pins hashes for every remote-code file at this revision,
# including ones loading never imports; fetch them all so the check is complete.
manifest = yaml.safe_load(open("configs/model/moondream2_artifact.yaml"))
for name in manifest["remote_code_sha256"]:
    hf_hub_download(manifest["repository"], name, revision=manifest["revision"],
                    cache_dir="/models/hub")
subprocess.run([sys.executable, "/tmp/verify_cache.py",
                "--artifact", "configs/model/moondream2_artifact.yaml",
                "--hub", "/models/hub"], check=True)
' > "$root/artifacts/model-cache-$commit.log" 2>&1 || {
  tail -40 "$root/artifacts/model-cache-$commit.log" >&2; exit 1; }

printf '{"role":"%s","commit":"%s","image":"%s","dataset_files":%d,"model_cache_mib":%d}\n' \
  "$role" "$commit" "$image" "$(wc -l < "$root/data/SHA256SUMS")" \
  "$(du -sm "$root/models" | cut -f1)"
