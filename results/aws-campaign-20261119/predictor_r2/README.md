# Predictor V2 r2 (G3)

The rules come from the [r2 specification](../../../docs/predictor-v2/r2-specification.md), frozen in `1d01508` before any fit. The implementation is `6469129`. Every number below comes from the **training partition** (T01–T12) or from validation **inputs**. No validation outcome has been read.

## Decision: G3 passed, `r2_component` selected and frozen

The single validation pass ran on 20 November 2026 from commit `265e9ea` (clean tree): targets unsealed at 10:36:05 UTC, decision at 10:36:07 UTC.
- **Decision:** [decision.json](validation/decision.json): candidate `r2_component`, selected `r2_component`, G3 `passed`.
- **Freeze marker:** [model_freeze_marker.json](validation/model_freeze_marker.json):
  - calibration SHA-256 `16c20438…f05a`;
  - decision SHA-256 `d3b55134…784d`.
- **Full report:** [validation_report.json](validation/validation_report.json), SHA-256 `c405f30d…5f09`.

Validation covers V01–V05: 630 request-cells per configuration (600 short, 30 long), every attempt successful, no unit missing an observed value.

| `r2_component` | Median APE | P90 APE | MAE | Bias | Interval coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `edge/full` | 3.48% | 6.04% | 38.9 ms | +24.2 ms | 99.8% |
| `cloud/raw_image` | 2.25% | 3.97% | 20.2 ms | +6.4 ms | 96.5% |
| bf16 split | 0.42% | 1.35% | 10.8 ms | +0.6 ms | 99.7% |
| `prompt_first` split | 0.44% | 1.61% | 15.7 ms | −4.4 ms | 96.2% |
| int8 split | 0.43% | 1.50% | 9.3 ms | +1.0 ms | 99.7% |
| **Gate** | **≤ 15%** | **≤ 30%** | | | (reported) |

| Model | Gate | Worst median / P90 APE | Top-1 agreement | Mean regret (95% CI) | Max regret |
| --- | --- | --- | ---: | --- | ---: |
| **`r2_component`** | **pass** | 3.48% / 6.04% | **619/630 (98.3%)** | **0.80 ms (0.35–1.29)** | 82 ms |
| `r2_additive` | pass (not needed) | 4.46% / 11.84% | 614/630 (97.5%) | 1.46 ms (0.85–2.09) | 92 ms |
| `constant_median` | fail | 32.3% / 115.1% | 503/630 (79.8%) | 42.3 ms (33.8–51.6) | 767 ms |

- **Every gate condition holds for the component model:**
  - each configuration is far inside the error limits;
  - coverage is 100%;
  - it agrees on 116 more request-cells than the constant baseline (cluster-bootstrap 95% CI 87–147);
  - there are 0 unsafe feasibility choices.
- **Training and validation errors match.** The grouped-CV P90 APE was at most 8.0%; validation's is at most 6.0%. There is no sign of overfitting to the training conditions.
- **Ranking:**
  - The observed fastest configuration was cloud in 503 request-cells and the edge in 127.
  - The component model chose cloud for 508 and the edge for 122, and agreed in 619.
  - The 11 misses are near-ties: 5 in V05-L1-C2, 3 in V03-L0-C2, and one each in V05-L0, V02-L0-C2 and V01-L1-C2. They cost 4–82 ms.
  - Agreement over the four lossless configurations is the same: 98.3%.
- **Breakdowns** (all in the report):
  - Every load state and condition stays within 6.7% P90 for every configuration, except V05 (80 Mb/s, 150 ms RTT). There the cloud's P90 is 13.1% and `prompt_first`'s 7.7%, both still under the gate.
  - The long task is the most predictable: at most 1.4% P90.
  - The 120 extrapolated predictions (V05-L1 remote, RTT 0.06 ms above training) have a median APE of 0.8% and a maximum of 20.4%.
- **One systematic error:** the edge is over-predicted by about 24 ms (+3%) on average. It's small, and the same in every load state.

r1 failed G3 on the physical pair because of cloud error tails (P90 30–33%) and edge memory coverage. r2 resolves both on AWS:
- **Cloud tails:** the cloud's P90 APE is 4.0%, now that cloud load is modelled explicitly and the network term models slow start;
- **Memory coverage:** the L4-6Q edge holds the model resident, so every configuration has full coverage.

---

The sections below were written before validation and are unchanged.

**Status at freeze:** the candidate was frozen and its validation predictions sealed; the validation pass had not yet run.

## Fit (training only)

`run_predictor_r2.py fit`: 3,060 request-cells per configuration, 15,300 units in all.
- **Source:** `training.jsonl`, SHA-256 `59fec5b1…1aa0`.
- **Unit table:** `training_units.jsonl`, SHA-256 `adb9816c…1fe3`, kept in `artifacts/`.

The `r2_component` calibration ([calibration_r2_component.json](candidate/calibration_r2_component.json)):

| Term | Fit |
| --- | --- |
| Edge stage, split configurations | 16.5 + 58.7·*k* ms at L0; 56.5 + 72.6·*k* ms at L1 (*k* = crops) |
| Edge stage, `edge/full` | short: 495 + 59.4·*k* (L0), 784 + 74.3·*k* (L1); long: 2,898 + 59.3·*k* (L0), 4,408 + 74.1·*k* (L1) |
| Server, `cloud/raw_image` | short: 107 + 18.7·*k* (C0), 616 + 30.4·*k* (C2); long: 599 + 18.9·*k* (C0) |
| Server, split configurations | short: 91–92 ms (C0), 397–398 ms (C2); long: 588–589 ms (C0) |
| Network | a + b·Φ, with Φ = 2·RTT + a slow-start transfer from a W₀ = 64 KiB initial window (best of the eight predeclared windows) |
| Network scales | b = 1.04 (cloud), 1.004 (int8), 1.002 (bf16 and `prompt_first`); intercepts 3–14 ms |
| Request bytes | cloud: image + 391 B; bf16 split: 2,986,476 B; int8 split: 1,494,968 B |

No coefficient needed clamping. The network scale is about 1 for every configuration: the analytic slow-start term describes the transfer almost exactly, and the fit only adds a few milliseconds of fixed overhead.

## Training review

From [training_review.json](candidate/training_review.json). The figures are median APE / P90 APE (%) per configuration:

| Model | edge | cloud | bf16 split | `prompt_first` | int8 | Top-1 agreement | Mean regret |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| **Grouped 5-fold CV (by request)** | | | | | | | |
| `r2_component` | 4.0 / 8.0 | 2.1 / 5.6 | 0.6 / 2.2 | 0.6 / 2.9 | 0.7 / 2.6 | 98.0% | 1.0 ms |
| `r2_additive` | 3.8 / 11.7 | 4.0 / 12.1 | 2.1 / 7.3 | 2.0 / 7.4 | 2.2 / 6.7 | 97.3% | 1.9 ms |
| `constant_median` | 22.5 / 46.8 | 30.5 / 145.2 | 27.4 / 80.2 | 27.7 / 80.1 | 24.3 / 67.5 | 83.4% | 38.4 ms |
| **Leave one condition out (12 folds)** | | | | | | | |
| `r2_component` | 3.7 / 7.2 | 2.1 / 5.2 | 0.6 / 2.2 | 0.6 / 3.0 | 0.8 / 2.6 | 98.2% | 0.9 ms |
| `r2_additive` | 3.7 / 10.9 | 4.2 / 13.9 | 3.1 / 11.0 | 3.0 / 10.8 | 3.2 / 9.7 | 97.5% | 2.2 ms |

- **Candidate:** `r2_component`. Its grouped-CV P90 APE is at most 8.0%, well under the 30% rule of section 6.
- **Leave one condition out:** the component model's error barely changes when a whole network condition is held out, so the network term carries over to links it was not fitted on.
- **The constant median** always predicts `cloud/raw_image`. That is the ranking bar: 83.4% agreement on training.
- **Intervals:** the 95th-percentile grouped-CV APE per configuration, which validation interval coverage will use:
  - `r2_component`: edge 12.2%, cloud 8.0%, bf16 split 3.2%, `prompt_first` 5.4%, int8 3.7%.

## Sealed validation predictions

`run_predictor_r2.py predict`, from `validation_inputs.jsonl` only. The [manifest](candidate/prediction_manifest.json) has:
- **Predictions:** 9,450, which is 3 models × 630 request-cells × 5 configurations.
- **Prediction file:** [validation_predictions.jsonl](candidate/validation_predictions.jsonl), SHA-256 `f2f14188…8f72`. Every `observed` field is null.
- **Code:** commit `6469129`, clean tree.
- **Labels:** every prediction is valid.
  - 120 per model are `physics_extrapolation`: the four remote configurations at V05-L1, whose measured RTT (150.672 ms) is 0.06 ms above the largest training RTT (150.608 ms).
  - Everything else is `interpolation`.
- **Sealed targets:** `validation_targets.jsonl` (SHA-256 `d8baf559…9759`) is mode 000 and has not been read.

The partition audit is copied to [candidate/partition_audit.json](candidate/partition_audit.json).

## The validation pass (run once)

```bash
.venv/bin/python scripts/prediction/run_predictor_r2.py validate \
  --partitions ~/g2v2plus-r2/partitions \
  --fit artifacts/g2v2plus-aws-campaign-derived/predictor-r2/fit \
  --predictions artifacts/g2v2plus-aws-campaign-derived/predictor-r2/predictions \
  --output results/aws-campaign-20261119/predictor_r2/validation
```

What this command does:
1. It unseals the targets, checks their hash and the hashes of the predictions and calibrations, and scores all three models.
2. It applies the section 8 gate and choice rule, and writes `decision.json`. It writes `model_freeze_marker.json` only if a model passes.
3. It refuses to run a second time. A decision now exists, so it can't run again.
