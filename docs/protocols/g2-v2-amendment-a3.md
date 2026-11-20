# G2 v2 protocol, amendment A3: aggregate exposure of the validation and evaluation cells

**Dated 20 November 2026.** Written after the G2 v2+ campaign finished and before any derived table, Predictor V2 r2 fit or selector design.

This amendment is a separate file on purpose. The [protocol](g2-v2-protocol.md) is pinned by SHA-256 (`a972ecdc595ab664c9914987c5aeaf71aba50daffcc86e02df2598315bf0e1d5`) in the [campaign grid](../../configs/profiles/aws_network_grid_v2plus.json) and in every campaign preflight. Editing it would break the verification of the finished run. The protocol text stays exactly as frozen.

## 1. What happened

Section 9 ("Isolation") keeps two kinds of cells closed:
- the **validation cells** (V01–V05), opened once for G3;
- the **evaluation cells** (E01–E05), sealed until the selector is frozen for G4.

On 19–20 November, while the campaign record ([results/aws-campaign-20261119](../../results/aws-campaign-20261119/README.md)) was being written, aggregates over **all 98 cells**, including V and E, were computed and read. The source was the per-sub-block `validation_report.json` medians.

**Exposed (cell-level aggregates only):**
- for every cell and configuration, the mean of the two repetitions' median end-to-end latencies;
- the configuration with the lowest such mean for every cell (the "descriptive winner");
- the edge's winning margins at L0 / C2, and the cloud's margins at the L0 / C2 cells it won;
- across all short cells: the ranges of the cloud-load effect (C0 → C2 on `cloud/raw_image`, +593 to +632 ms) and the edge-load effect (L0 → L1 on `edge/full`, ×1.39 to ×1.44);
- the regret of always choosing one configuration, over all 98 cells;
- an observed threshold: at L0 / C2, the edge wins exactly when the C0 cloud median is above about 390 ms.

**Not exposed:**
- no per-request rows were analysed;
- no p90, bootstrap CI, tie decision, accuracy score or predictor input table was computed for V or E;
- `derive_g2v2_tables.py --unseal-evaluation` was never run, and the sealed-split mechanism is intact.

## 2. Why the damage is limited

**The structure was predicted before the run.**
- The phase E winner estimate ([winners.json](../../results/aws-phase-e-20261118/winners.json)) was committed in `6e9b88a` on 18 November, before any reference, pilot or campaign run.
- It already modelled latency as:
  - an unshaped per-configuration profile;
  - plus a fitted network term in RTT and payload over rate;
  - plus the calibrated load effects: L1 as a factor on the edge, C2 as a penalty on the cloud.
- It predicted the winner configuration of **all 98 campaign cells correctly (98/98)**, including the 13 edge wins at L0 / C2.
- The additive load structure and the crossover are therefore prior knowledge from training-free calibration data, not something learnt from V or E.

**What G3 and G4 score was never computed for V or E.** G3 scores per-request prediction error and ranking. G4 scores regret on complete paired blocks against the section 9 oracle, with CIs and ties.

## 3. Rules from now on

1. **No design choice may use V or E data.** That covers every Predictor V2 r2 and selector choice: inputs, model form, features, hyperparameters, fallback rules, quality constraints, thresholds and hysteresis. Each must be justified by one of:
   - training cells (T01–T12);
   - the phase D and E calibration data;
   - documents frozen before 19 November.

   Each choice is declared in the r2 specification before the single validation pass.
2. **The additive load model.** r2 may use it only when it is justified and fitted on training cells. The specification cites the phase E estimate as its prior source, and this amendment as disclosure.
3. **No tuning against the exposed aggregates.** Neither the 390 ms threshold nor any other exposed number may be used as a selector constant or a tuning target.
4. **Gates and seals unchanged.**
   - The G3 and G4 gates, the bootstrap seed 1707 and the resample count stay as declared.
   - V per-request rows are opened once, for the G3 validation pass.
   - E rows stay sealed (no `--unseal-evaluation`) until the selector and its oracle specification are frozen by hash.
5. **Committed records hold training-cell latency aggregates only.**
   - The V and E rows of `subblock_medians.csv` and `descriptive_winners.json` move to the ignored `artifacts/aws-campaign-20261119-sealed/`. Their SHA-256 hashes are recorded in the results README.
   - Hash-only files (`runs.csv`, `SHA256SUMS`) stay complete.
6. **Disclosure in the thesis.** This exposure is reported under threats to validity, together with the 98/98 prior-estimate agreement as mitigation.
