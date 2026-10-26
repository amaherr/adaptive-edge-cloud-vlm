# Phase 5 handoff — adaptive execution and mixed-run analysis

Date: 2026-10-26 (Africa/Cairo). Phase 5 implements end-to-end mixed configuration execution from the [scheduler plan](dynamic_scheduler_implementation_plan.md). The **local G5 integration gate passed** on the pinned Moondream2 CPU/loopback fixture. This is a diagnostic run: local G3 ranking failed, physical G4 is open, and the separate cloud machine is unavailable. Production-style automatic selection remains gated.

## What changed

- Added `adaptive-benchmark` with exact pre-inference image and prompt features, a frozen manual/replay network and health state, the Phase 4 selector, and resident edge/cloud/split dispatch. The plan and each terminal trial row pin the manifest, config, model contract, calibration, replay hash, selection trace, observed result, and all timing. The run can resume only under the identical plan.
- Added one bounded retry for a TCP connection failure that occurs before request bytes are sent. It rechecks capabilities, excludes the failed configuration, and logs both attempts and added time. Other failures are explicit terminal rows without an automatic retry. Existing fixed-policy and manual inference paths remain intact.
- Added `analyze-adaptive` to match each selected request with its same-mode fixed control on the exact schedule. It audits output tokens, placement, wire bytes, provenance, decisions, retries, and terminal-row coverage. The [execution contract](phase5_execution_contract.md) defines the full input and failure behavior.
- Froze [the local replay](../configs/adaptive/phase5_local_replay.json) before the real run. It deliberately assigns four request IDs to each of the three available configurations through declared health exclusions. These are replayed conditions, not measured failures.

Implementation checkpoint: `68a81ac` contains the committed execution path and replay used by the real run. The run manifest records full source commit `68a81acf53b37c85aaf61a38aa893598b5c2852f` and `git_dirty=false`. The subsequent analysis/doc update is checked by the final suite.

## Local Moondream diagnostic

On one WSL host, the dual-path CPU server and full edge CPU adapter ran one excluded warmup and one measured pass over all 12 frozen synthetic requests. The [summary](../artifacts/phase5/2026-10-26/local-1/summary.json), [analysis](../artifacts/phase5/2026-10-26/local-1/analysis.json), and [raw rows](../artifacts/phase5/2026-10-26/local-1/requests.jsonl) show:

| Measure | Result |
| --- | ---: |
| Terminal measured trials | 12/12 successful; zero retries |
| Selected edge / cloud / projected split | 4 / 4 / 4 |
| Same-placement output token mismatches | 0/12 |
| Decision trace / wire-byte / control errors | 0 / 0 / 0 |
| Median observed end-to-end request | 29,776.3 ms |
| Median selection overhead | 2.906 ms |
| Mean application wire bytes: edge / cloud / split | 0 / 1,837.5 / 2,987,753.25 |
| `g5_local_gate_passed` | `true` |

The same request IDs and fixed controls share model contract SHA-256 `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1` and dataset manifest SHA-256 `c5bc9df4a010dcefbcbb7fd60c9dd1e47dc0e0cf798acff579a1a004006ce8d1`. Replay SHA-256: `bfcdf06a5dc283ae596abbda51e1da1096bab097d83a7288fe0f78db865ef7d8`. The source calibration ID is `2917acf00f8df890e474d4e5d9c0f9ca66555631a298d004702539f5a85b208d`.

The fixed controls were run earlier; CPU load and run order were uncontrolled. The replay deliberately forces a mixed choice, so the median latency and wire totals describe this integration campaign only. They are not a scheduler speedup or physical-network crossover estimate. No retry happened in the real run; the bounded retry and no-retry-after-unknown-stage paths passed synthetic tests.

The [archive](../artifacts/phase5/2026-10-26/local-1.tar.gz) contains the adaptive plan, run manifest, raw rows, summary, analysis, frozen replay/calibration/gate reports, all three fixed control runs, and [per-file hashes](../artifacts/phase5/2026-10-26/local-1/file_hashes.json). Archive SHA-256: `4659f9fc8804cc140aafbbadb83e0c52ed389e8825f8d0ea3aa2f9e42c381fb6`. Generated `artifacts/` are ignored by Git; preserve the archive when transferring machines.

## Run and inspect it

With the pinned cache and model environment ready, start the dual-path server in terminal A:

```bash
.venv-model/bin/adaptive-vlm adaptive-server \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
```

In terminal B, choose a **new** run directory:

```bash
.venv-model/bin/adaptive-vlm adaptive-benchmark \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --manifest data/manifests/correctness.jsonl \
  --calibration artifacts/phase3/2026-10-26/clean-1/calibration.json \
  --replay configs/adaptive/phase5_local_replay.json \
  --run-dir artifacts/phase5-replay \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --vision-device cpu --text-device cpu --cloud-device cpu \
  --warmups 1 --repetitions 1 --diagnostic-allow-unpassed-gates
```

On interruption, repeat the command with `--resume`; the config, manifest, calibration, replay and schedule must be identical. `summary.json` shows final outcomes and choices. Each `requests.jsonl` row shows `attempts[*].decision.candidates` with predictions and exclusions, the selected mode, time, bytes and generated tokens. `run_manifest.json` records the Git commit, cleanliness, software and placement.

Analyze against the original same-schedule fixed controls:

```bash
.venv/bin/adaptive-vlm analyze-adaptive \
  artifacts/phase5-replay \
  artifacts/phase2/2026-10-26/local-1/edge \
  artifacts/phase2/2026-10-26/local-1/cloud \
  artifacts/phase2/2026-10-26/local-1/distributed \
  --output artifacts/phase5-replay/analysis.json
```

The analyzer refuses mismatched schedules, model/manifest/config hashes and incomplete trial sets. Use the bundled controls in the archive if the original `artifacts/phase2` directory was not transferred.

## Wiring a separate cloud machine

1. Verify the cloud host and dual-path server as in the [deployment guide](../README.md#using-a-separate-cloud-pc). Run fresh fixed edge, cloud, and distributed controls on one frozen physical manifest and identical schedule. Establish real memory and capability evidence before declaring candidate availability.
2. Finish physical G2 profiling and freeze a new G3 training/request/network holdout **before** fitting. Recalibrate on the physical host pair and satisfy G3 ranking and error targets. Evaluate Phase 4 on matched physical holdouts and satisfy G4 safety/regret evidence. Do not reuse the failed local calibration or its inspected requests as a fresh test.
3. Copy the local replay JSON under a new campaign name. Replace manifest/profile/calibration/model/host hashes, task bucket, exact per-candidate placement profiles, and network bucket/source/bandwidth/RTT. The loader checks the placement profiles against runtime device/address arguments and calibration hashes. Use `trial_network_states` for a predeclared condition schedule; use `unhealthy_by_request` only for declared manual fault scenarios. Keep `quality_floor=null` until a validated predecision quality estimate exists. Freeze and commit this file before the adaptive campaign.
4. Start the physical dual-path server, then run `adaptive-benchmark` with the new config/manifest/calibration/replay and matching devices. Once both full gates pass, supply `--predictor-evaluation NEW_G3/evaluation.json --selector-evaluation NEW_G4/evaluation.json`; the diagnostic flag is then unnecessary. Preserve all raw trials and use `analyze-adaptive` with the fresh three fixed controls. Any placement mismatch is disclosed instead of counted as a same-device equality success.

The current CLI requires a resident **full edge model** to count exact prompt tokens before selection. A cloud-only edge that cannot load it needs a separately validated predecision tokenizer/feature path before this mode can run; the Phase 5 command will refuse that unsupported placement.

## Verification and gate status

Synthetic tests exercise mixed 12-request selection, resume without duplicate trials, a single safe pre-send retry, no retry for unknown-stage failure, zero feasible candidates, token and wire matching, and decision-trace tamper detection. The full suite passed with `163 passed`, zero skipped, including loopback integration. Ruff and strict mypy passed. **Local G5 passed; physical G5, G2, G3, and G4 remain open.** Phase 6 live sensing has not been enabled.
