# Phase 6 monitoring and trace contract

Phase 6 adds a deterministic network trace replay and an optional live monitor to `adaptive-benchmark`. The current Moondream calibration is still limited to uncontrolled loopback and failed G3 ranking. Live monitoring is therefore a **diagnostic integration path**; it does not authorize automatic performance selection or replace the physical network measurements required by G2–G4.

## Trace replay

`replay-network-trace` reads a versioned trace JSON and content-verified calibration, then runs the actual `Predictor.predict` and `select` functions without inference. Each sample names a network bucket/source, uplink/downlink values, RTT, probe health/age/error, optional edge memory budget, a segment, and an expected selected ID. The output preserves every candidate prediction and exclusion reason. Its report counts switches and checks the expected crossover, zero switches after warmup within the declared noisy segment, and no unsafe remote choice under stale/failed health.

The committed [synthetic calibration](../configs/network_traces/phase6_synthetic_calibration.json) is **constructed solely for deterministic tests**. It assigns a 500 ms edge baseline, a 700 ms cloud baseline, and a 100 ms projected split plus transfer; the [trace](../configs/network_traces/phase6_synthetic_trace.json) moves from 1 to 100 Mbit/s uplink with 10 ms declared RTT. The projected payload crosses the edge latency as bandwidth rises. No synthetic latency is presented as a measured physical-host result. The predictor now pins the controlled network state's `source` when calibrated from profiles and rejects a different source at prediction time.

No hysteresis was added: the predeclared constant fast segment (85–100 Mbit/s) has zero switches after warmup. A future physical campaign should add hysteresis only if its own repeated constant-condition trace shows instability, with the threshold and switching/load costs declared before evaluating its holdout.

## Live monitor

`adaptive-benchmark --monitor-mode live` samples the server's application-level `HEALTH_REQUEST` and capability response before each decision. `--probe-interval-ms` controls cache reuse; a cached sample is used only while its age is within both that interval and `--probe-max-age-ms`. `--probe-timeout-ms` bounds **each** health/capability exchange. Failure, not-ready status, fingerprint mismatch, or a stale sample makes remote configurations unavailable; the selector may choose a feasible edge candidate or return an explicit no-feasible result. The choice stays fixed during inference. A safe pre-send retry takes a fresh monitor sample and records both decisions.

Every attempt records the UTC sample timestamp, age, source, health-probe round trip, capability-probe time, ready paths, error, cached/fresh flag, passive estimate and sample count, declared memory evidence, and sampling overhead. `monitor_overhead_ms` is a subset of `selection_overhead_ms`, which is included in the trial's observed end-to-end latency. The run plan and manifest pin the sampling interval, max age, timeout, and passive smoothing settings. `analyze-monitor` audits terminal coverage, per-attempt provenance, remote and memory safety, monitoring overhead fraction, probe errors, switches, and model-load cost. The resident policies have zero **per-switch** model load cost; initial model loads occur before the request schedule and are recorded separately by the CLI.

After a successful remote request, `LiveMonitor.observe_send` computes a bounded exponential moving average of `8,000 × request_wire_bytes / network_send_ms` in bit/s, requiring at least 4,096 sent bytes by default. The default smoothing coefficient is 0.25; estimates are clamped between 0.1 Mbit/s and 10 Gbit/s. This is **application socket-send rate**, affected by buffering and CPU load. It is not a validated path goodput measurement and is never fed to the Phase 3 predictor. No downlink goodput is inferred from response receive time because that interval includes server inference; the decision record states this null reason. Health-probe round trip is likewise an application RTT, distinct from controlled path RTT in the replay state.

On `--resume`, successful saved passive-update rows are replayed through the estimator before new decisions so its smoothing history is preserved. Probe freshness is sampled anew; an old health result is never restored as a live endpoint claim.

An edge memory budget requires a measured peak argument; unknown or excessive edge/split memory excludes those candidates. Process RSS is logged only as a diagnostic resident-process measure and is never treated as free GPU memory. Cloud queue/load and free GPU memory are not reported because the current protocol has no verified sensors for them. A live monitor failure cannot make a remote endpoint eligible, and tests cover the no-feasible result when edge memory is unknown under a budget.

## Commands and artifacts

```bash
.venv/bin/adaptive-vlm replay-network-trace \
  --calibration configs/network_traces/phase6_synthetic_calibration.json \
  --trace configs/network_traces/phase6_synthetic_trace.json \
  --output-dir artifacts/phase6-trace-replay

.venv/bin/adaptive-vlm analyze-monitor artifacts/phase6-live-run \
  --output artifacts/phase6-live-run/monitor_analysis.json
```

For the full local server/client commands and the physical-host migration steps, see the [Phase 6 handoff](phase6_handoff.md). The trace output contains `evaluation.json` and `decisions.jsonl`; the live run contains its Phase 5 plan/rows/summary plus `monitor_analysis.json`.
