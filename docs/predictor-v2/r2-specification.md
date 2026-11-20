# Predictor V2 r2: predeclared specification (G3 on the G2 v2+ AWS campaign)

**Frozen 20 November 2026, before any r2 fit was run.** This document fixes the r2 inputs, models, fitting procedure, training review, validation protocol, G3 gate, model-choice rule and fallback. Nothing here may change after the single validation pass. Any later attempt is a new, separately frozen revision (r3) on data that r2 has not scored.

Background:
- [Predictor V2 specification](specification.md): the r1 rules this revision inherits;
- [r1 validation report](validation-report.md): G3 failed on cloud error tails and edge memory coverage, on the physical pair;
- [G2 v2+ baseline report](../../results/aws-campaign-20261119/baseline/README.md): training cells only;
- [amendment A3](../protocols/g2-v2-amendment-a3.md): aggregate exposure of V/E cells, and the rules that follow from it.

## 1. Data and isolation

**Source.** The G2 v2+ campaign on AWS (`aws-vqa-v2plus`), all 196 sub-blocks valid.

**Partitions.**
- **How they were made:** `derive_g2v2_tables.py profiles --splits training validation` then `partitions`. Both ran on 20 November 2026 from commit `0948aa8`, whose `src/` equals the frozen campaign commit `d5e116c` except the two hash-exempt analysis modules, so reference verification passed.
- **Where:** the private WSL directory `~/g2v2plus-r2/` (mode 700).
- **Hashes:**

  | File | Content | SHA-256 |
  | --- | --- | --- |
  | `partitions/training.jsonl` | 30,600 rows (T01–T12, 108 sub-blocks, every attempt `ok`) | `59fec5b16c544b1e6c97b0aeba48f5b2118663ab4a72dffeffa599135f861aa0` |
  | `partitions/validation_inputs.jsonl` | 6,300 input snapshots, no outcomes | `e916bad4d3264e2c4d82e22461c279ac0eeed71e0b38a9539d50ae84795ce6c8` |
  | `sealed/predictor-v2-sealed-r5p6u31e/validation_targets.jsonl` | the validation outcomes, mode 000 | `d8baf5598be353cafdfd0aa58b1b950f90403362c85578dbf1a19fde7d519759` |

- **Validation profiles:** the 44 directories `profiles/V*` hold outcomes and are mode 000.
- **Evaluation:** E01–E05 were neither profiled nor opened (`evaluation_profiles_opened: []`), as before.

**Isolation record.**
- While sealing, one command printed the first 80 bytes of `profiles/V01-L0-r0/profile/profile.jsonl`: the campaign ID, cloud-load level and the start of a config hash. These are identifiers, with no latency, status or output.
- Apart from that, no validation outcome has been read.
- Every design choice below comes from the training partition (T01–T12), the phase D and E calibration, or documents frozen before 19 November (A3 rule 1).

**Source hash after r2.** The r2 code adds modules under `src/adaptive_vlm/`, which changes the reference verifier's source hash. Later derivations from the campaign must therefore run from a worktree of `0948aa8`. That includes the evaluation profiles for G4 and any re-derivation of these partitions.

## 2. Unit of analysis

- **The unit is a request-cell:** one request ID in one cell (condition × edge load × cloud load × task).
- **Its target** is the median of the successful repetitions' `observed_end_to_end_ms`, two in this campaign. Stage targets are medians in the same way.
- **The data:**
  - training has 48 short cells × 60 IDs plus 6 long cells × 30 IDs, which is 3,060 units per configuration;
  - validation has 20 short cells × 30 IDs plus 2 long cells (V04 at L0 and L1) × 15 IDs, which is 630 units per configuration and 3,150 predictions.
- **Why:** the repetitions are not independent experimental units (r1 rule E3), so all fits, errors and rankings use request-cells.

## 3. Inputs

Every input is known before the request starts:

| Input | Source | Used by |
| --- | --- | --- |
| Configuration (five) | candidate | all |
| Task (`short`, `long`) | task bucket | all |
| Crop count *k* | image width and height through the pinned tiling rule (below) | component, additive |
| Image bytes | task features | component, additive (cloud request size) |
| Edge-load level (L0, L1) | input snapshot | component, additive |
| Cloud-load level (C0, C2) | input snapshot | component, additive |
| Uplink goodput (bit/s), echo RTT (ms) | pre-sub-block probe | component, additive |

**Crop count.** *k* = tiles + 1 (the global crop). The tiles come from the pinned model's `select_tiling(height − 112, width − 112, 266, 12)` (Moondream `image_crops.py`: crop 378 px, patch 14, overlap margin 4 patches, at most 12 crops), reimplemented exactly. Training has *k* ∈ {2, 9, 10}.

- **Why crops:** on the training partition, the split path's edge vision time is about 57 ms per crop. At L0 the medians are 134, 546 and 596 ms for *k* = 2, 9 and 10. r1's training-only diagnosis had already named crop-aware vision as the next feature.

**Declared but unused:** prompt tokens (15–24 in the short task, constant in the long task), downlink goodput (responses are about 2.2 KB), the pre-request edge NVML snapshot and the previous response's cloud state.
- On training, the load levels capture the load effects, and these fields add nothing a two-level input doesn't already give.
- Sensing load without the level labels is a G5 question (phase 5), not a G3 one.

## 4. Models

Three models are fitted on the training partition only. All durations are in ms.

### 4.1 `r2_component` (primary)

Prediction = **E** (edge stage) + **S** (server) + **N** (network) + **R** (client residual), so the components sum exactly to the total.

| Component | Training target (profile `fields`) | Form |
| --- | --- | --- |
| **E**, edge stage | `edge_total_ms` | `edge/full`: a + b·*k*, per (task, edge load). Split configurations: a + b·*k*, per (configuration, edge load), pooled over tasks, because the vision stage doesn't see the prompt. Cloud: 0. |
| **S**, server | `server_processing_ms` | Cloud: a + b·*k*, per (task, cloud load). Split configurations: the median, per (configuration, task, cloud load). Edge: 0. |
| **N**, network | `transport_residual_ms` = exchange − server, which includes the connection setup | a_c + b_c · Φ(S_c, B, RTT; W₀), per remote configuration c. Edge: 0. |
| **R**, client residual | `observed_end_to_end_ms` − `edge_total_ms` − `network_exchange_ms` | the median, per (configuration, edge load) |

**The network term Φ.** One RTT for the TCP handshake, plus the slow-start transfer of S_c bytes, plus one RTT for the request's last byte and the response:
- The transfer starts with a window of W₀ bytes that doubles each round.
- While the window is below the bandwidth-delay product B·RTT, a round costs max(RTT, round bytes ÷ B), and the last round costs only its bytes ÷ B.
- Once the window reaches the bandwidth-delay product, the remaining bytes take bytes ÷ B.
- B is the measured uplink goodput in bytes per ms. The exact algorithm is `slow_start_ms` in the implementation.
- **Request bytes S_c:**
  - cloud: image bytes plus the training median of (`request_wire_bytes` − image bytes);
  - split configurations: the training median of `request_wire_bytes`, which is constant per configuration.
- **W₀** is one value shared by all configurations, chosen on training from {16, 32, 48, 64, 96, 128, 192, 256} KiB. The choice minimises the sum over remote configurations of the squared relative error of N on the training request-cells, with a_c and b_c refitted for every candidate.
- **Fitting a_c and b_c:** ordinary least squares, with b_c constrained to ≥ 0. If the unconstrained b_c is negative, b_c = 0 and a_c is the mean.

**Fitting the edge and server terms.**
- **a + b·*k* terms:** ordinary least squares on the request-cell stage medians. If the fitted b is negative, b = 0 and a is the median.
- **Median terms:** medians of the request-cell stage medians.

**Rationale (training only).** The baseline report found:
- the cloud load acts on the server stage only;
- the edge load acts on the edge stages only;
- transfer time at long RTT follows slow-start dynamics, not bytes ÷ rate.

The training partition confirms the stage medians: edge stages don't change with the cloud-load level, and server stages don't change with the edge-load level. Φ with W₀ = 64 KiB tracks the training transport medians within a few per cent from 5 to 150 ms of RTT, where bytes ÷ rate + 2·RTT misses by up to a factor of 2.4.
- **Load model:** the load structure is the one in the phase E estimate, now fitted per level (A3 rule 2).
- **Decode:** answer length isn't an input. Training answers are 1 token at the median for the short task and always 32 tokens for the long task, so the per-task terms absorb the decode time. The observed length is never used.

### 4.2 `r2_additive` (simple baseline)

This is ordinary least squares on request-cell `observed_end_to_end_ms`, per (configuration, task):

T = θ₀ + θ₁·1[L1] + θ₂·1[C2] + θ₃·*k* + θ₄·RTT + θ₅·(8 · S_c ÷ B)

- **`edge/full`** has no network terms (θ₄ = θ₅ = 0).
- **Terms without variation in a group are dropped:** for example, θ₂ for the long task, which ran only at C0.
- **No constraints.**

This is the linear analytic model that the phase E estimate used (profile + RTT + payload ÷ rate + load terms).

### 4.3 `constant_median` (ranking baseline)

For each (configuration, task), the median of the training request-cell latencies. It ignores the network and the loads. It is never selectable; it only sets the ranking bar.

## 5. Coverage and feasibility

**Prediction labels.** Each prediction gets one of three labels:
- **`unsupported`** (invalid; it counts against coverage):
  - a missing or non-finite input;
  - an unknown configuration or task;
  - a (task, load level) combination absent from training, such as the long task at C2.
- **`physics_extrapolation`** (valid and scored, but reported separately):
  - a measured uplink or RTT outside the training range of the pre-sub-block probes;
  - or *k* outside [2, 10].
- **`interpolation`** (valid): everything else.

**Feasibility.** All five configurations are feasible for every supported input.
- The L4-6Q edge holds the model resident (A2b rule 2), and there were no OOMs and no failed attempts in training.
- int8 is admitted (A2a rule 4).
- An "unsafe feasibility choice" is a validation attempt that failed (`status` not `ok`) for a configuration declared feasible.

## 6. Training review (before validation)

Run on the training partition only, and reported with the frozen candidate.

- **Grouped 5-fold CV** by request ID (seed 1707). Folds are assigned by sorting the IDs, shuffling them with `random.Random(1707)`, then dealing them round-robin. All of a request's cells go in the same fold.
  - Per model and configuration: out-of-fold median APE, P90 APE, MAE and bias.
- **Leave-one-condition-out:** 12 folds, T01–T12, each condition with all its load states and tasks. This tests the network term on a link it hasn't seen.
- **Interval:** for each configuration, the 95th percentile of the out-of-fold absolute percentage error from grouped CV gives a symmetric interval, prediction × (1 ± q). Its coverage on validation is reported, not gated.
- **Training rule:** if the fitted primary model has a negative network scale b_c or an unconstrained negative crop slope that had to be set to 0 (an unphysical sign), it is reported. If its grouped-CV P90 APE exceeds 30% for any configuration, `r2_additive` replaces it as the candidate before validation. There is no other change or retuning.

## 7. Validation (single pass)

**Order of steps:**
1. The fitted calibrations are written and hashed, the training review is completed, and the candidate is committed with its sealed predictions:
   - the predictions of all three models for the 630 validation units × 5 configurations;
   - made from `validation_inputs.jsonl` only;
   - written to a file whose SHA-256 is recorded in the commit.
2. Only then are the validation targets unsealed, by `chmod` 400 on the sealed file. They are verified against the hash in section 1 and read once.
3. The decision is written. A second validation run is refused if a decision file exists.

**Observed values.**
- **Latency:** the observed latency of a unit is the median of its successful repetitions. A unit with no successful repetition for a configuration has no observed value for it; this is reported, and it is not scored for that configuration.
- **Fastest configuration:** the one with the lowest observed latency among the configurations that succeeded. Exact ties break by configuration order: `edge/full`, `cloud/raw_image`, synchronous, `prompt_first`, int8.

**Metrics, per model and configuration**, over valid predictions with an observed value:
- **Error:**
  - median APE and P90 APE, with linear-interpolation percentiles (`prediction.v1.calibration.percentile`);
  - MAE and signed bias (prediction − observed);
  - interval coverage.
- **Coverage:** valid predictions ÷ units. Labels are counted separately.
- **Breakdowns:** the same metrics by condition (V01–V05), load state, task and coverage label.

**Ranking, per model** (on request-cells; the predicted choice is the configuration with the lowest prediction, with the same tie-break):
- **top-1 agreement** with the observed fastest configuration, over all five configurations;
- the same over the four lossless configurations (reported, not gated);
- **mean and median regret:** the observed latency of the predicted choice minus the observed fastest;
- a cluster bootstrap over request IDs (seed 1707, 2,000 resamples) for the agreement difference against `constant_median` and for each model's mean regret.

## 8. G3 gate and model choice

**A model passes G3 if all of these hold:**
1. For each of the five configurations: median APE ≤ 15% and P90 APE ≤ 30%, with short and long units pooled.
2. Coverage is 100%: every validation unit has a valid prediction for every configuration.
3. Top-1 agreement is at least the `constant_median` agreement on the same units.
4. There are 0 unsafe feasibility choices.

**Model choice (fixed now):**
- If `r2_component` (or `r2_additive`, if section 6's training rule made it the candidate) passes, it is selected.
- Otherwise, if `r2_additive` passes, it is selected as the predeclared fallback.
- Otherwise G3 fails and no model is selected. The constant median is never selected.

**After the decision:**
- On a pass, `model_freeze_marker.json` records:
  - the selected calibration's SHA-256;
  - the decision's SHA-256;
  - the source commit.
- No model is refitted after validation in any case. G4 uses the frozen selected model, or, if G3 fails, the offline oracle and simple policies (roadmap 3.4).

## 9. Outputs

The results go to `results/aws-campaign-20261119/predictor_r2/`:
- calibrations (JSON, with training-row hashes);
- the training review;
- the sealed predictions and their hash;
- the validation report;
- the decision;
- the freeze marker (pass only);
- `SHA256SUMS`.

Validation latency aggregates enter the committed record only after the decision (A3 rule 5 covers training; the validation pass is the protocol's one opening of V).
