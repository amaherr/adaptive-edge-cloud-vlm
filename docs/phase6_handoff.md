# Phase 6 handoff — bounded live probes and network trace replay

Date: 2026-10-26 (Africa/Cairo). Phase 6 follows the [scheduler plan](dynamic_scheduler_implementation_plan.md) and [Phase 5 handoff](phase5_handoff.md). The **local synthetic trace and live-monitor integration checks passed**. The physical G6 gate remains open because the separate cloud host and controlled physical network are unavailable. The Phase 3 local ranking miss and open G4 gate still prohibit validated automatic performance selection.

## What changed

- Added `replay-network-trace` to run frozen bandwidth/RTT/health/memory samples through the real predictor and selector. The deterministic fixture crosses from edge to projected split, stays stable through a noisy fast segment without hysteresis, and falls back safely on stale/failed probes or unknown memory.
- Added `LiveMonitor` with bounded application health and capability probes, timestamped age/cache/timeout provenance, and fail-closed remote eligibility. It computes a bounded passive **application send-rate** estimate after remote inference and leaves downlink goodput unknown. Probe RTT is kept separate from the predictor's declared path RTT. Controlled calibrations now pin and check their network-state source.
- Extended `adaptive-benchmark` with `--monitor-mode live`, per-attempt monitoring records, monitoring/selection overhead, and passive update count. Added `analyze-monitor` to audit sample completeness, unsafe choices, switch count, and overhead impact. Existing replay/default and fixed-policy commands retain their behavior. The [monitor contract](phase6_monitor_contract.md) defines measurements and failure handling.

Implementation checkpoint `ab97c91` contains the committed Phase 6 monitor, trace inputs, three-request live manifest/replay and tests used by the real run. The live run manifest records full source commit `ab97c9171eb317ce0da26af18543c9fccffb422f` with `git_dirty=false`. The final audit/source-provenance refinement and documentation are verified in the follow-up commit.

## Synthetic controlled trace

The [trace evaluation](../artifacts/phase6/2026-10-26/synthetic-trace-1/evaluation.json) and [decision rows](../artifacts/phase6/2026-10-26/synthetic-trace-1/decisions.jsonl) contain nine deterministic samples. At 1 Mbit/s uplink, edge/full is selected; at 100 Mbit/s, projected split is selected. The declared fast segment varies from 85 to 100 Mbit/s after warmup with **zero switches**. A stale sample, failed probe, and edge-memory-budget exclusion yield the expected safe outcomes, including an explicit no-feasible decision when remote health is lost and edge memory is insufficient. The report has zero expected-choice mismatches and zero unsafe selections. It counts three configuration transitions overall and zero per-switch model loads because all candidates are assumed resident. `synthetic_trace_gate_passed=true`.

This fixture was designed to exercise the decision boundary and is not trained from physical observations. The [synthetic archive](../artifacts/phase6/2026-10-26/synthetic-trace-1.tar.gz) contains its calibration, trace, decisions, evaluation and [file hashes](../artifacts/phase6/2026-10-26/synthetic-trace-1/file_hashes.json). Archive SHA-256: `5a7df280576c3c009ed116070a89036ad260de3cc568e3dc18a3c09508b2f268`.

## Local Moondream live-monitor diagnostic

The [three-request frozen manifest](../data/manifests/phase6_local_probe.jsonl) and [replay](../configs/adaptive/phase6_local_live_replay.json) select one edge, one cloud, and one projected-split request on one WSL CPU/loopback host. A separate dual-path server was running; its full-model load and the client's resident edge-model load both finished before the request schedule. The replay's health exclusions are declared laboratory inputs rather than real observed outages. The [raw run](../artifacts/phase6/2026-10-26/local-live-1/requests.jsonl), [summary](../artifacts/phase6/2026-10-26/local-live-1/summary.json) and [monitor audit](../artifacts/phase6/2026-10-26/local-live-1/monitor_analysis.json) show:

| Measure | Local result |
| --- | ---: |
| Measured requests / successful | 3 / 3 |
| Edge / cloud / projected split choices | 1 / 1 / 1 |
| Timestamped decision samples / probe errors | 3 / 0 |
| Unsafe selections / audit failures | 0 / 0 |
| Median / maximum monitor sampling cost | 1.505 / 34.441 ms |
| Median monitoring share of observed latency | 0.00466% |
| Successful passive send-rate updates | 1 |
| Switches across successful measured trials | 2 |
| `local_live_monitor_gate_passed` | `true` |

All three probes were fresh (`age_ms=0`); their health-probe round trips were approximately 0.792, 30.45 and 1.063 ms. The longest probe occurred before the cloud request under local CPU load. The one projected-tensor transfer yielded a final bounded application send-rate estimate of about **2.20 Gbit/s**. That value reflects local socket buffering and is **not physical uplink goodput**; no later request consumed it. The live selector kept the original declared loopback state for prediction. No probe failure or safe retry occurred during this live run, so those behaviors are established by synthetic tests rather than measured fault recovery. The local run's three choices came from replay exclusions, not a measured bandwidth crossover.

The [live archive](../artifacts/phase6/2026-10-26/local-live-1.tar.gz) includes the complete adaptive run, monitor audit, manifest, replay, calibration and prior G3/G4 gate reports with [per-file hashes](../artifacts/phase6/2026-10-26/local-live-1/file_hashes.json). Archive SHA-256: `c7a663d37befc8533d4d88fac59a0dd121109b6fc518239f6975a4c5a12ca0a3`. Generated `artifacts/` are ignored by Git; preserve both archives when moving machines.

## Run and view it now

Replay the deterministic trace without loading Moondream:

```bash
.venv/bin/adaptive-vlm replay-network-trace \
  --calibration configs/network_traces/phase6_synthetic_calibration.json \
  --trace configs/network_traces/phase6_synthetic_trace.json \
  --output-dir artifacts/phase6-trace-replay
```

For the real local probe, start `adaptive-server` in terminal A:

```bash
.venv-model/bin/adaptive-vlm adaptive-server \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
```

In terminal B, choose a new output directory:

```bash
.venv-model/bin/adaptive-vlm adaptive-benchmark \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --manifest data/manifests/phase6_local_probe.jsonl \
  --calibration artifacts/phase3/2026-10-26/clean-1/calibration.json \
  --replay configs/adaptive/phase6_local_live_replay.json \
  --run-dir artifacts/phase6-live-replay \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --vision-device cpu --text-device cpu --cloud-device cpu \
  --warmups 0 --repetitions 1 --diagnostic-allow-unpassed-gates \
  --monitor-mode live --probe-interval-ms 0 \
  --probe-max-age-ms 5000 --probe-timeout-ms 2000 --passive-alpha 0.25

.venv/bin/adaptive-vlm analyze-monitor artifacts/phase6-live-replay \
  --output artifacts/phase6-live-replay/monitor_analysis.json
```

Inspect `requests.jsonl` → `attempts[*].monitoring` for the probe timestamp, RTT, age, ready paths, error and passive estimate beside each decision; `monitor_analysis.json` reports safety/coverage and monitor overhead. Use the same command with `--resume` only after an interrupted run and identical plan inputs. The diagnostic flag remains required for the current local calibration.

## Wiring a second machine

1. Complete the verified hardware and dual-path server setup from the [deployment guide](../README.md#using-a-separate-cloud-pc). Rebuild Phase 2 profiles across controlled physical bandwidth/RTT blocks and record exactly how each state was measured. Freeze a new Phase 3 split with request and network-block holdouts. Calibrate on the new host pair; controlled fits now bind their source name, and live states from another source are rejected rather than extrapolated.
2. Replace the synthetic trace with a frozen physical trace using the new calibration/task bucket and in-coverage bandwidth/RTT samples. Declare the expected crossover and a constant-condition noisy segment before collecting held-out outcomes. Run `replay-network-trace` to verify transitions and switch count. Do not use the synthetic 1/100 Mbit/s crossover as a prediction of physical hardware.
3. Freeze a new Phase 5/6 replay with actual manifest, calibration, per-candidate placement profiles and physical network state. Use `trial_network_states` for a declared condition sequence. Keep `unhealthy_by_request` empty during normal operation, or populate it only for an explicitly documented fault scenario. For edge memory limits, supply a measured peak and budget; the monitor will not infer free GPU memory from process RSS. Start the physical server and run `adaptive-benchmark --monitor-mode live` with bounded probe settings, plus passing G3/G4 reports when available.
4. Archive the raw requests, probe records, predictor inputs, calibration, replay, controlled network measurements and environment. Run `analyze-monitor` and the Phase 5 `analyze-adaptive` against new matching fixed controls. Measure actual path goodput independently; the passive socket-send estimate and application health-probe RTT are labeled diagnostics and must not be substituted for that measurement. If real noisy traces switch undesirably, predeclare and evaluate a hysteresis rule in a new campaign, counting switching and load costs.

## Gate status and verification

Synthetic fixtures cover the controlled crossover, stable noisy segment, stale/failed probes, no-feasible memory exclusion, cache age, bounded timeout, passive rate clamping/smoothing and resume restoration, and live decision record/audit integration. The full suite passed with `168 passed`, zero skipped, including loopback integration. Ruff and strict mypy passed; both archive hashes and their internal file hashes were verified. **The local synthetic and live G6 checks passed. Physical G6, G2–G4, and validated automatic selection remain open.**
