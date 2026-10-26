# Phase 1 handoff — executable configuration space

Completed: 2026-10-26 (Africa/Cairo). **G1 passed for the local CPU/loopback deployment.** A separate cloud PC is still unavailable, so physical-host concurrency and memory behavior remain unverified. This handoff follows the [implementation plan](dynamic_scheduler_implementation_plan.md) and the [Phase 0 baseline](phase0_handoff.md).

## What changed

- Added stable configuration IDs and typed `SystemState`, `TaskState`, capability, and decision records in `src/adaptive_vlm/scheduler/`. A dispatcher routes one request to exactly one already loaded `ExecutionPolicy` and logs the full availability snapshot. Phase 1 selection is **manual**; predicted or automatic selection belongs to later phases.
- Added a capability request/response to the bounded TCP protocol. The response identifies the model contract and supported request paths. An adaptive client excludes a remote configuration if the endpoint is unreachable, its fingerprint differs, or its required path is absent.
- Added `adaptive-server`, which loads one pinned full Moondream2 model and accepts raw-image and projected-tensor requests on the same endpoint. Existing `cloud-server`, `distributed-server`, and fixed-policy commands remain available.
- Added `adaptive-infer` for one manually selected configuration. Its `--edge-components full` mode loads one full edge adapter and shares it between edge inference and edge vision encoding, so switching those paths does not reload weights. `vision` registers cloud and split only; `none` registers cloud only. `prompt_first` is advertised by the server but is not a scheduler candidate until its overlap cost is profiled.
- Added `scripts/verify_phase1.py`, which keeps one client model and one dual-path server resident while executing all three configurations over either one request or a frozen manifest. It saves each decision, result, elapsed time, client RSS reading, model fingerprint, environment, and equality verdict.
- Made memory-budget checks require a supplied **peak** requirement. The post-load RSS was about 1.07 GiB in this run, while request-time RSS reached 4.52 GiB; using the startup value as a safety bound would be wrong. Without a measured requirement, an explicit budget rejects the candidate with `edge_memory_requirement_unknown`.

Implementation commits: `1cff2e0` (contracts, protocol, server, CLI and tests), `e8cfa78` (resident smoke runner), and `c0a4d20` (peak-memory preflight and manifest-wide verification). The real campaign's environment record reports clean commit `c0a4d20b8efdf819e02af3656369a4d90d1023bb`.

## Gate evidence

The local dual-path server advertised `raw_image`, `projected_synchronous`, and `projected_prompt_first` with model contract fingerprint `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`. The client used CPU vision and CPU text with the pinned Moondream2 artifact, the unchanged 32-token greedy contract, and the [same 12-request manifest](../data/manifests/correctness.jsonl) as Phase 0 (SHA-256 `c5bc9df4a010dcefbcbb7fd60c9dd1e47dc0e0cf798acff579a1a004006ce8d1`). The server and client each loaded once and stayed resident for the campaign.

| G1-local check | Result |
| --- | --- |
| Registered configurations | `edge/full`, `cloud/raw_image`, `distributed/projected/synchronous` available together |
| Frozen requests | 12/12 completed through each configuration; 36 result rows |
| Output equivalence | 12/12 request triples have identical token IDs and decoded text |
| Baseline equivalence | 12/12 match the Phase 0 edge token sequence |
| Routing and exclusion | Tests cover exact policy dispatch, missing remote path, wrong remote fingerprint, unreachable endpoint, and insufficient/unknown memory |
| Static checks | Ruff and strict mypy passed |
| Full test suite | 131 passed, 0 skipped with loopback access; 82.31% coverage |

The [campaign report](../artifacts/phase1/2026-10-26/campaign-1/phase1_smoke.json) records `complete_and_equal=true` and `matches_baseline_edge=true`. Its `rows` contain 12 entries for each configuration, including all decisions and output token IDs. The one-request [smoke report](../artifacts/phase1/2026-10-26/smoke-1/phase1_smoke.json) also passed. The local [results archive](../artifacts/phase1/2026-10-26/campaign-1.tar.gz) has SHA-256 `6523f60d9e23c8016ad075ae5d550bb27c0c71e1523663422e663472c97abf3b`. All `artifacts/` paths are ignored by Git; copy the archive to durable storage.

### Memory and limits of this evidence

The client measured about 1,072 MiB RSS immediately after loading and a maximum of 4,522 MiB after requests. Server-side request metrics reported a maximum of about 4,869 MiB RSS. Both processes shared a WSL host with 7.637 GiB reported RAM, so their combined residency can cause contention or swapping. The distributed choice in `full` mode executes only vision on the edge but retains the full edge model in memory for immediate edge-only availability. Use `--edge-components vision` when the full edge model cannot remain resident; then `edge/full` is explicitly unavailable.

The observed request times are sequential localhost CPU correctness observations. They do not establish the best policy, network crossovers, or physical-host performance. The memory numbers are observations for this workload, not a validated universal peak: Phase 2 must profile per-task peaks and add headroom before using them as production constraints. Manual bandwidth and RTT values are recorded by Phase 1 but do not influence the chosen configuration yet.

## How to run it

From the repository root, with the pinned `.venv-model` and verified `.model-cache`, start the server in terminal A:

```bash
.venv-model/bin/adaptive-vlm adaptive-server \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
```

Wait for its readiness JSON to list both `raw_image` and `projected_synchronous`. In terminal B, select one configuration manually:

```bash
.venv-model/bin/adaptive-vlm adaptive-infer \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --configuration-id distributed/projected/synchronous \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --edge-components full \
  --vision-device cpu --text-device cpu \
  --image tests/fixtures/moondream_smoke.ppm \
  --prompt "Describe this image." --output-dir artifacts/manual-adaptive
```

The command prints `decision` and `request` JSON. With `--output-dir`, it writes `run_manifest.json`, `decision.json`, and `request.json` under a fresh run-ID directory. `--configuration-id` may also be `edge/full` or `cloud/raw_image`. For cloud-only use, `--edge-components none` avoids loading model weights on the client; for cloud plus projected split, use `vision`. When passing `--edge-memory-budget-mib`, also pass a conservative profile-derived `--edge-required-peak-mib`; the budget applies to the memory footprint of the currently resident client process.

To repeat the full G1-local check without reloading between requests, keep terminal A running and execute:

```bash
.venv-model/bin/python scripts/verify_phase1.py \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --manifest data/manifests/correctness.jsonl \
  --baseline-edge-requests \
    artifacts/phase0/2026-10-25/local-campaign-1/edge/requests.jsonl \
  --output-dir artifacts/phase1/NEW_CAMPAIGN
```

Use a new output path. Omit `--baseline-edge-requests` if the local Phase 0 artifacts are unavailable; the script will still compare all three Phase 1 outputs with each other. Inspect the saved report and test logs with:

```bash
.venv/bin/python -m json.tool \
  artifacts/phase1/2026-10-26/campaign-1/phase1_smoke.json
tail -5 artifacts/phase1/2026-10-26/campaign-1/pytest.log
```

## Next handoff

Phase 2 should define timing boundaries and a profile schema, then measure compute, serialization, transfer, server work, and conservative memory peaks per configuration/task/network bucket. The automatic predictor and oracle are later gates. The separate physical cloud host is still needed for physical deployment claims.
