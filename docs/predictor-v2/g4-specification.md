# G4: the r2 selector against the measured oracle (predeclared)

**Frozen 20 November 2026, before any evaluation cell (E01–E05) was profiled or opened.**
- **Machine-readable copy:** `results/aws-campaign-20261119/g4/g4_specification.json`. It pins by SHA-256 every input and the selection code; its own SHA-256 is recorded in the freeze commit and is required by the evaluation command.
- **Evaluation:** `scripts/prediction/run_g4_r2.py evaluate`. It refuses to run if the specification, the code or the calibration differ from what was frozen, or if an evaluation already exists.

**Background:**
- G3 passed and froze `r2_component` (calibration `16c20438…f05a`, [r2 record](../../results/aws-campaign-20261119/predictor_r2/README.md)).
- The physical roadmap set the G4 target ("median regret ≤ 10%, ≥ 10 complete paired blocks, network holdout included"), and the [selector and oracle design](../design/selector-and-oracle.md) describes the same gate. This document applies them to the G2 v2+ campaign. As [amendment A3](../protocols/g2-v2-amendment-a3.md) requires, no choice below uses evaluation data.

## Selector

**Decision function.** The existing `scheduler.selector.select`, extended with an explicit candidate set and unchanged otherwise:
- **Exclusions:** unavailable, unhealthy and invalid predictions are excluded with a reason.
- **Choice:** the lowest predicted total wins, and the stable configuration order breaks exact ties.
- **No outcomes:** no measured outcome enters a decision.

**Prediction.** The frozen `r2_component` prediction for each configuration, from the request's pre-execution inputs (section 3 of the [r2 specification](r2-specification.md)).

**Candidate sets:**
- **Primary:** all five configurations. int8 is admitted (A2a rule 4), and quality is identical across configurations: official VQA accuracy is the same on training and validation IDs, with 0 questions differing.
- **Secondary (reported):** the four lossless configurations.

**Constraints.** None:
- no SLA;
- no quality floor, because of the parity above;
- no memory budget, because the edge model is resident and no attempt in the campaign ran out of memory.

All five configurations are available and healthy for every evaluation request, as they were throughout the campaign.

## Units, oracle and blocks

- **Unit:** a request-cell, as in G3. The evaluation cells hold 20 short cells × 30 IDs and 2 long cells (E04 at L0 and L1) × 15 IDs, so 630 request-cells.
  - Its observed latency per configuration is the median of the successful repetitions.
  - A request-cell is **complete** only if every configuration succeeded in every repetition.
- **Measured oracle:** the configuration with the lowest observed latency in the request-cell, with the same tie order. It is used only afterwards, to score decisions.
- **Block:** a cell, in which all five configurations ran on the same request IDs. A block is complete if all its request-cells are.
- **Network holdout:** E01–E05 are network conditions that neither fitting (T01–T12) nor selection choices (V01–V05) used. The evaluator checks that the scored conditions are all in E and none is in T or V.

## Baselines

- **Always-edge, always-cloud and always-int8.**
- **Best static:** the configuration with the lowest median request-cell latency over the **validation** cells, with configuration order breaking ties. It is `cloud/raw_image` (median 830 ms, against 1,052 ms for the edge and 1,354 ms for int8), chosen before any evaluation data was opened (`best_static.json`).

## Metrics

For the selector, the lossless selector and every baseline, over complete request-cells:
- **Top-1 agreement** with the oracle (the lossless selector is scored against the lossless oracle).
- **Regret:** the absolute regret (ms) and the relative regret (% of the oracle latency) of the configuration chosen:
  - the median, P90, mean and maximum;
  - the mean with a cluster-bootstrap 95% CI over request IDs (seed 1707, 2,000 resamples).
- **Paired comparison:** the selector's latency minus each baseline's, as a mean with a bootstrap CI.
- **Breakdowns:** by load state, by condition and by cell.
- **Quality:** official VQA accuracy on the evaluation IDs, for every configuration, with parity checked (roadmap 4.4). The answers are scored by `scripts/vqa/score_campaign_vqa.py --splits evaluation --unseal-evaluation`, as for training and validation.
- **Section 9 oracle:** the cell-level oracle for the evaluation cells (medians, bootstrap CIs, ties), from `derive_g2v2_tables.py oracle --splits evaluation --unseal-evaluation`. It is reported with the selector's per-cell choices.

## G4 gate

The primary selector (five configurations) passes G4 if all of these hold:
1. **Median relative regret ≤ 10%** over the complete request-cells.
2. **At least 10 complete blocks.**
3. **Network holdout:** every scored condition is in E01–E05.
4. **Safety:** 0 decisions with no feasible candidate, and 0 selections of a configuration that failed an attempt.
5. **Consistency:** every decision equals the minimum over its own logged feasible predictions (`predicted_oracle`).

**What a pass and a fail mean:**
- **A pass** shows that the frozen predictor and selector come close to the measured oracle on unseen networks and requests. It does not enable live automatic selection by itself; that is phases 5 and 6.
- **A failure** is reported as such. There is no refitting and no retuning after the evaluation pass.

## Order of operations

1. Commit the selector change, the G4 code and its tests, this document and `best_static.json`.
2. Run `run_g4_r2.py freeze` to write `g4_specification.json` (it records that commit and the code hashes), and commit it. This is the freeze.
3. Derive the evaluation profiles with `derive_g2v2_tables.py profiles --splits evaluation` from a worktree of `0948aa8`. The r2 code changed the source hash that reference verification checks, and that worktree's `src/` equals the frozen campaign commit except the two exempt analysis modules.
4. Run `run_g4_r2.py evaluate` once.
5. Then run the evaluation-ID VQA scoring and the section 9 oracle for the evaluation cells.
