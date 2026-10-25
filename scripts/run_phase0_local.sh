#!/usr/bin/env bash
set -euo pipefail

# Reproduce the three-policy, one-sweep CPU correctness gate on one host.
# Run from the repository root with a fresh output directory argument.

if [[ $# -ne 1 ]]; then
  echo "Usage: scripts/run_phase0_local.sh NEW_RUN_DIR" >&2
  exit 2
fi

run_dir=$1
if [[ -e "$run_dir" ]]; then
  echo "Output directory already exists: $run_dir" >&2
  exit 2
fi
if [[ ! -x .venv-model/bin/adaptive-vlm || ! -x .venv/bin/adaptive-vlm ]]; then
  echo "Install the development and pinned model environments first" >&2
  exit 2
fi

mkdir -p "$run_dir"
run_dir=$(realpath "$run_dir")
server_pid=""

stop_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_server EXIT

wait_for_server() {
  local log_file=$1
  local attempt
  for ((attempt = 0; attempt < 180; attempt++)); do
    if grep -q '"ready": true' "$log_file"; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      cat "$log_file" >&2
      echo "Server exited before readiness" >&2
      return 1
    fi
    sleep 1
  done
  cat "$log_file" >&2
  echo "Server did not become ready within 180 seconds" >&2
  return 1
}

config=configs/experiments/chunk6_local_campaign.yaml
artifact=configs/model/moondream2_artifact.yaml
model_cli=.venv-model/bin/adaptive-vlm
dev_cli=.venv/bin/adaptive-vlm

git rev-parse HEAD > "$run_dir/git_commit.txt"
sha256sum \
  data/manifests/correctness.jsonl \
  "$config" \
  configs/model/moondream2.yaml \
  "$artifact" \
  requirements/model-cu128.lock > "$run_dir/inputs.sha256"
"$dev_cli" environment --output "$run_dir/environment.json" \
  > "$run_dir/environment.stdout.json"

"$model_cli" benchmark \
  --config "$config" --mode edge --run-dir "$run_dir/edge" \
  --artifact-manifest "$artifact" --cache-dir .model-cache --offline \
  --vision-device cpu --text-device cpu --warmups 1 --repetitions 1 \
  > "$run_dir/edge.log" 2>&1

"$model_cli" cloud-server \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --artifact-manifest "$artifact" --cache-dir .model-cache --offline \
  --device cpu > "$run_dir/cloud-server.log" 2>&1 &
server_pid=$!
wait_for_server "$run_dir/cloud-server.log"
"$model_cli" benchmark \
  --config "$config" --mode cloud --run-dir "$run_dir/cloud" \
  --warmups 1 --repetitions 1 > "$run_dir/cloud.log" 2>&1
stop_server

"$model_cli" distributed-server \
  --config configs/experiments/chunk5_local_distributed.yaml \
  --artifact-manifest "$artifact" --cache-dir .model-cache --offline \
  --device cpu > "$run_dir/distributed-server.log" 2>&1 &
server_pid=$!
wait_for_server "$run_dir/distributed-server.log"
"$model_cli" benchmark \
  --config "$config" --mode distributed --run-dir "$run_dir/distributed" \
  --artifact-manifest "$artifact" --cache-dir .model-cache --offline \
  --vision-device cpu --pipeline synchronous --warmups 1 --repetitions 1 \
  > "$run_dir/distributed.log" 2>&1
stop_server

"$dev_cli" compare-runs \
  "$run_dir/edge" "$run_dir/cloud" "$run_dir/distributed" \
  --output "$run_dir/comparison.json" > "$run_dir/comparison.stdout.json"

.venv/bin/python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
report = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
assert report["complete_and_equal"], report
assert report["expected_measured_pairs"] == 12, report
for mode in ("edge", "cloud", "distributed"):
    summary = json.loads(
        (run_dir / mode / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["measured_attempts"] == 12, summary
    assert summary["status_counts"] == {"ok": 12}, summary
print("G0-local correctness: 12/12 matched pairs; all three modes 12/12 OK")
PY

echo "Results: $run_dir"
