# Predictor V2 r2 (G3)

The rules come from the [r2 specification](../../../docs/predictor-v2/r2-specification.md), frozen in `1d01508` before any fit. The implementation is `6469129`. Every number below comes from the **training partition** (T01–T12) or from validation **inputs**. No validation outcome has been read.

**Status:** the candidate is frozen and its validation predictions are sealed. The single validation pass has **not** been run.

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

## Next: the single validation pass

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
3. It refuses to run a second time.
