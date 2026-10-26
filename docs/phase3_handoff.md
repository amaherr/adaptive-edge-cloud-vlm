# Phase 3 handoff — component predictor and frozen local holdout

Date: 2026-10-26 (Africa/Cairo). Phase 3 follows the [implementation plan](dynamic_scheduler_implementation_plan.md) and [Phase 2 profile handoff](phase2_handoff.md). The predictor and local evaluation pipeline are implemented. **G3 has not passed:** the predeclared local request holdout misses the ranking target, and there is no controlled network-condition holdout while the separate cloud PC is unavailable.

## What changed

- Added typed predictor inputs, a provenance-bound calibration, named latency components, wire-byte estimates, an uncertainty interval, and explicit invalid/coverage reasons in `scheduler/predictor.py`. Matching the host pair, model contract, placement profile, task bucket, generation contract, feature ranges and network regime is required before prediction.
- Added `calibrate-predictor`, which fits only successful measured training rows from a Phase 2 profile and evaluates untouched request IDs and network blocks. Warmups and failures cannot fit coefficients. The command compares against a per-configuration constant training-median predictor and reports total-latency median/p90 APE, signed bias, interval coverage and paired ranking accuracy by task/network bucket.
- Frozen the [local split](../configs/predictor/phase3_local_holdout.json) in commit `cb14032` **before** fitting: requests `correctness-001` through `correctness-010` train; `correctness-011` and `correctness-012` are held out. The split pins the Phase 2 profile SHA-256 and 15% median / 30% p90 APE targets. No network block can be held out from the one uncontrolled loopback bucket.
- Added tests for byte-to-bit units, monotonic transfer cost, deterministic predictions, missing and out-of-range state, model/host/placement mismatch, CPU memory exclusion, controlled-network fitting, profile provenance, and output-length leakage. The [predictor contract](phase3_predictor_schema.md) defines the component equation and coverage behavior.

Implementation commits: `cb14032` (frozen split), `b089cab` (predictor, fitter, evaluator and CLI), and `8b1c03d` (calibration validation). The resulting calibration identifier is `2917acf00f8df890e474d4e5d9c0f9ca66555631a298d004702539f5a85b208d`.

## Held-out local evidence

The [evaluation](../artifacts/phase3/2026-10-26/clean-1/evaluation.json) has two held-out requests per configuration, six prediction rows, no failures and no out-of-coverage inputs. The evaluator passes `expected_output_tokens=null` for every held-out task and substitutes the **training** median of seven output tokens. It never uses a held-out generated length to predict that request.

| Configuration | Median APE | p90 APE | Mean signed bias | Constant-median APE | Held-out interval coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Edge/full | 3.64% | 4.92% | +404 ms | 3.61% | 2/2 |
| Cloud/raw image | 2.87% | 3.14% | −116 ms | 2.92% | 2/2 |
| Distributed/projected synchronous | 4.89% | 6.35% | −597 ms | 5.07% | 2/2 |

All candidate total-latency APEs meet the numeric 15%/30% thresholds on this very small local holdout. **Paired ranking misses the predeclared target:** the component predictor selected the observed fastest configuration in 0/2 cases, while the constant-median baseline did so in 2/2. The predictor placed cloud about 6 ms ahead of edge for each request; edge was the observed winner both times. These are one-sweep CPU/loopback observations and do not establish a stable policy ranking. The intervals use training residuals and 2/2 held-out coverage is too little for an uncertainty claim.

The [calibration](../artifacts/phase3/2026-10-26/clean-1/calibration.json), [per-trial predictions](../artifacts/phase3/2026-10-26/clean-1/predictions.jsonl), and [environment record](../artifacts/phase3/2026-10-26/clean-1/environment.json) preserve the fitted components, byte estimates, fallback output estimate, observed latency, constant baseline, coverage decision, and Git/software provenance. The environment records clean commit `e6bbd17ab50911637e5eb2553350f2635184932c`. The [archive](../artifacts/phase3/2026-10-26/clean-1.tar.gz) contains these files, the frozen split and [per-file hashes](../artifacts/phase3/2026-10-26/clean-1/file_hashes.json). Archive SHA-256: `dc4dec11f286bba21aeca379f42b960275ca2a8018ccdb0267fb7ea621e7b8eb`. Source profile SHA-256: `2a009d880b9c861ee5eedd8d1a60d982d59e91fbd359397d22f0274d1356e054`.

Ruff and strict mypy passed; the full suite passed with `141 passed`, zero skipped using loopback access.

## How to run and inspect it

From the repository root, use a fresh output directory:

```bash
.venv/bin/adaptive-vlm calibrate-predictor \
  --profile artifacts/phase2/2026-10-26/local-1/profile/profile.jsonl \
  --split configs/predictor/phase3_local_holdout.json \
  --output-dir artifacts/phase3-replay
```

Open `artifacts/phase3-replay/evaluation.json` for the gate flags and grouped error/ranking tables. Inspect `calibration.json` for fitted components and coverage bounds, and `predictions.jsonl` for each held-out task snapshot, component estimate and observed result. `Calibration.load(path)` and `Predictor(calibration).predict(spec, state, task)` expose the pure runtime API. Out-of-coverage predictions return `valid=false` with a reason and are not execution decisions.

## Gate scope and next work

The local predictor supports only the measured 4×4 synthetic fixture bucket on the same WSL host and uncontrolled loopback. It uses the measured loopback transport residual; physical uplink/downlink and RTT coefficients remain uncalibrated. CPU memory readings are after-request snapshots, so an edge memory budget makes edge/split prediction invalid with `edge_memory_requirement_unknown`. Live clients still need a pre-inference exact prompt-token feature and a justified output-length estimate.

Full G3 needs a separate cloud host, controlled bandwidth/RTT blocks, a frozen request and network holdout, and enough training coverage after excluding both. The local ranking miss blocks using this fit as an automatic performance selector. If the model or decision rule is revised, declare a **new** evaluation split/campaign before measuring it; do not treat the two already inspected holdout requests as a fresh test. Keep the constant-median policy as a reference and report both ranking and latency error on the new holdout. Phase 4 selector logic can be implemented and tested synthetically, but performance claims and automatic activation must wait for a passing gate and safe memory/executor evidence.
