# Thesis completion roadmap

> Written 20 November 2026, after the G2 v2+ campaign on AWS finished (196/196 sub-blocks valid, [record](../results/aws-campaign-20261119/README.md)). It replaces the "later phases" of the [physical roadmap](roadmap.md) (phases Q–AR) and section 8 of the [G2 v2+ plan](g2-v2-plus-plan.md). The gates and criteria stay as they were predeclared.

The thesis claim we are working towards:

> Under changing network, edge-load and cloud-load conditions, the system observes the current state, predicts the latency of each feasible configuration of one VLM, and selects a configuration per request. It comes close to the measured oracle and beats the fixed-placement baselines where adapting pays off, with no loss in answer quality.

What the G2 v2+ data shows (training cells; [baseline report](../results/aws-campaign-20261119/baseline/README.md)):
- Cloud wins most cells. The edge wins only under heavy cloud load on slower or higher-latency links. Split inference wins none.
- The two loads add independent, roughly fixed costs, so a simple predictor is plausible.
- The gain from adapting is concentrated in specific conditions rather than spread on average.

That makes G4 and G5 (selection under changing conditions) the core of the thesis, and the split-inference negative result a reportable finding.

## Overview

| Phase | What | Where | Depends on | Effort | Cost |
| --- | --- | --- | --- | --- | --- |
| **0** | Close out the run: data isolation, commit, AWS cost | local | — | ½ day | — |
| **1** | Fix the analysis code; derive the G2 v2+ tables | local | 0 | 1–2 days | — |
| **2** | G2 v2+ baseline report (training cells), quality, comparison with physical G2 | local | 1 | 2–3 days | — |
| **3** | Predictor V2 r2 and **G3** (validation cells opened once) | local | 2 | 4–6 days | — |
| **4** | Selector V2 and **G4** offline (evaluation cells unsealed) | local | 3 | 2–3 days | — |
| **5** | Adaptive runtime for the five configurations; dynamic protocol (amendment A4) | local | 4 (or 3 if G4 is offline-only) | 4–6 days | — |
| **6** | **G5** dynamic evaluation on AWS | AWS | 5 | 3–4 days | ~$50 |
| **7** | Final paired analysis (Phase 7), communication, overheads, stability | local | 6 | 3–4 days | — |
| **8** | Archive, reproducibility package, AWS teardown | local + AWS | 7 | 1 day | — |
| **9** | Thesis writing | — | 2 onward, in parallel | 3–5 weeks | — |

Critical path: 0 → 1 → 3 → 4 → 5 → 6 → 7. Phase 2 and the writing of chapters 1–4 can run alongside phases 3–5.

Budget: about $60–65 spent of $180. G5 needs about $50, which leaves about $65 contingency. The stopped pair costs about $0.35 a day.

---

## Phase 0: close out the run

**Status (20 November 2026):**
- **Done:** 0.1 ([amendment A3](protocols/g2-v2-amendment-a3.md), kept in a separate file so the pinned protocol hash stays valid; the pre-run phase E estimate had predicted all 98 winners), 0.2, 0.3 (pushed) and 0.5.
- **0.4 decided:** keep the stopped pair until G5 (~$0.35 a day). The compute-down plan was not applied, and the ECR images are kept.

0.1 **Record the split exposure (amendment A3).** On 19–20 November, per-cell medians and descriptive winners for **all** 98 cells were computed and viewed. That includes the validation (V01–V05) and evaluation (E01–E05) cells, which the protocol keeps unopened until G3 and sealed until G4 (section 9, "Isolation").
- Write down exactly what was seen: cell medians and winners, not per-request rows.
- Declare that no predictor or selector design choice may use V or E data, and that the predictor's model form (phase 3) is fixed from **training cells only**.
- If phase 3 adopts the additive "edge × load factor, cloud + load penalty" form, the amendment must say so honestly: it was suggested by a view of all cells, but it is justified and fitted on training cells.
- Report this in the thesis's threats-to-validity section.

0.2 **Reseal the committed record.**
- Keep only training rows in `subblock_medians.csv` and `descriptive_winners.json`.
- Move the V and E rows to `artifacts/` (ignored).
- Keep `runs.csv` and `SHA256SUMS` complete, since they contain hashes only.
- Rewrite the README's "first look" section to training cells only.

0.3 **Commit** the record and the roadmap. Push when you say "Commit and merge"; this includes the unpushed phase E/F commits `6e9b88a`…`883d132`.

0.4 **AWS cost decision.**
- Option one: keep the stopped pair until G5 (~$0.35 a day).
- Option two: apply the reviewed compute-down plan (15 resources destroyed: instances, system disks, alarms and the budget stop action), bringing it to ~$0.17 a day. The trade-off is a ~20 min rebuild for G5, with g6.xlarge capacity risk in us-east-1d.
- Either way, the data volumes, bucket and ECR stay.
- Optionally delete the four superseded ECR images and keep `4049c069` (frozen) only.

0.5 **Update the [documentation index](README.md#where-things-stand):** G2 v2+ is done on AWS, and physical G2 v2 has been superseded.

## Phase 1: analysis code and derived tables

**Status (20 November 2026):** see [derived checks](../results/aws-campaign-20261119/derived/README.md).
- **1.1 done.** Scope fixed, plus two related defects: empty groups and timed-only rows.
- **1.2 done.** 196/196 sub-blocks pass, with 0 reference mismatches and 0 cross-GPU disagreements.
- **1.4 done.** All five configurations score identically (training 75.33%, validation 79.67%, lowercased), so parity holds.
- **1.3 done.** The derivation ran in WSL with temporary `/app` and `/artifacts` links.
  - 108 training profiles pass.
  - Oracle: cloud wins 48 of 54 cells, the edge 6; 2 are ties.
  - The task-feature bundles (120 short and 60 long requests), which feed phase 3, are derived too.

**Gate passed:** correctness is clean, the training oracle is derived, and parity holds.

1.1 **Fix `evaluate_rows` scope** (`benchmark/heterogeneous_correctness.py`).
- Scope coverage, quality equality and repeatability to the IDs and repetitions a sub-block actually measured.
- Remove the `defaultdict` side effect in the `insufficient` comprehension.
- Add tests: a single-repetition subset passes, and a genuine mismatch still fails.

1.2 **Recompute row-level correctness** for all 196 sub-blocks: reference mismatches and cross-GPU disagreements against the short and long bundles. This compares outputs only, not latency, so it doesn't open any split. The result must be zero mismatches, as in the pilot.

1.3 **Run `derive_g2v2_tables.py` on the AWS campaign** with the dataset contract `aws_g2v2plus_dataset_contract.json`.
- Produce per-sub-block profiles and the section 9 oracle: median, p90, bootstrap 95% CI with seed 1707 and 2,000 resamples, measured winner, margin and ties.
- Run it for **training cells only** (the evaluation stays sealed; no `--unseal-evaluation`).
- Check that the script handles five configurations, the cloud-load levels and the long task; extend it if not.

1.4 **Official VQA accuracy** per configuration on the training and validation IDs (`run_official_vqa_eval.py`).
- Score both lowercased (primary) and unchanged answers.
- Check placement parity: at most 2 pp between any two configurations on the same IDs.
- int8 is covered by its admission evidence, but it must also pass parity here.
- The evaluation-split accuracy waits until phase 4.

**Gate:** row-level correctness is clean, the training oracle table is derived, and parity holds. If parity fails, the selector must treat quality as configuration-dependent (section 9).

## Phase 2: G2 v2+ baseline report

**Status (20 November 2026): done.** See the [baseline report](../results/aws-campaign-20261119/baseline/README.md).
- **2.1:** the record stays in `results/aws-campaign-20261119/`, now with an S3 receipt (12,442 objects verified by MD5).
- **2.2:** crossover map, load effects with CIs, the long task, split anatomy, int8 gain, the `prompt_first` null result and static policies.
  - The loads are independent: cloud load adds +606 ms to the cloud, edge load multiplies the edge by ×1.415, and the interaction is at most 18 ms.
  - The split is dominated by the cloud in every load state.
  - Always-cloud regret is 2.5%.
- **2.3:** the hardware ratio and what carries over from physical G2. Whether splitting pays depends on the edge-vision to cloud-vision speed ratio.
- **2.4:** G2 v2+ passed, with the section 8 gate evidence for all 196 sub-blocks.

2.1 **Results record** `results/aws-g2v2plus-<date>/` with `SHA256SUMS` and S3 receipts (roadmap phase G).

2.2 **Training-cell findings:**
- **Crossover map:** winner against link rate and RTT for each load state.
- **Load effects with CIs:** the cloud-load penalty and the edge-load factor, and whether they are independent.
- **Long task.**
- **Split inference:** why it loses (payload, two-stage overhead), the int8 gain against link rate, and the `prompt_first` null result.
- **Static policies:** the cost of always choosing one configuration, as motivation for adapting.

2.3 **Comparison with physical G2:**
- the hardware ratio (RTX 3050 laptop against L4-6Q, and the cloud PC against L4);
- which conclusions carry over, and why G2 v2+ is a new baseline rather than a replication.

2.4 **Write it up as G2 v2+ passed**, with the validity gates as evidence (protocol section 8).

## Phase 3: Predictor V2 r2 and G3

**Status (20 November 2026): G3 passed.** `r2_component` is selected and frozen. See the [r2 record](../results/aws-campaign-20261119/predictor_r2/README.md).
- **3.1:** the [r2 specification](predictor-v2/r2-specification.md) was committed (`1d01508`) before any fit.
- **3.2:** partitions came from `0948aa8`, with the validation targets sealed. The models were fitted on T01–T12 only. Training review: grouped-CV P90 APE at most 8.0%, 98.0% top-1 agreement.
- **3.3:** the single validation pass on V01–V05 (630 request-cells × 5 configurations):
  - median APE 0.4–3.5% and P90 APE 1.4–6.0% per configuration;
  - top-1 agreement 98.3% against 79.8% for the constant baseline;
  - mean regret 0.8 ms;
  - 0 unsafe choices.
- **Source hash:** the r2 code changes the reference verifier's source hash. G4's evaluation profiles must therefore be derived from a worktree of `0948aa8`.

3.1 **Predeclare r2 before fitting** (spec revision plus amendment):
- **Inputs:** the schema-3 edge state, `cloud_load_level`, `previous_cloud_state` (B8), the measured network (pre-sub-block probe), task (short or long, crop count, prompt tokens), and configuration (five).
- **Models:**
  - the primary component model;
  - a simple additive or analytical baseline (roadmap phase U);
  - the constant-median baseline for ranking.
- **Lessons from r1:**
  - r1 failed on cloud error tails, so model the cloud-load penalty explicitly and report tails;
  - r1 also failed on edge memory coverage, which is resolved on AWS: the L4-6Q edge is resident, with no swap.
- **Gate:** unchanged.
  - median APE ≤ 15% and P90 APE ≤ 30% per configuration;
  - top-1 feasible agreement at least the constant-median baseline;
  - coverage and safety as in the [specification](predictor-v2/specification.md);
  - model choice rule fixed in advance (seed 1707, 2,000 resamples).
- **Fallback:** if the primary model fails, decide in advance whether the simpler model may be selected, and on what rule. Never refit after validation.

3.2 **Update the data preparation** (`prepare_predictor_v2_data.py`) for the AWS layout, the five configurations, both loads and the long task. Fit on **T01–T12 only**.

3.3 **Single validation pass** on V01–V05. Write the validation report and the G3 decision. Issue `model_freeze_marker.json` only on a pass.

3.4 **If G3 fails:** report it honestly. A failed G3 still allows an offline G4 with the oracle and simple policies, but the thesis claim weakens to "a simple rule-based selector" and must be stated that way.

## Phase 4: Selector V2 and G4 (offline, evaluation cells)

**Status (20 November 2026): G4 passed.** See the [G4 record](../results/aws-campaign-20261119/g4/README.md).
- **4.1:** `scheduler.selector.select` now takes an explicit candidate set (five configurations, or the four lossless ones), fed by the frozen `r2_component` predictions. There are no constraints: quality parity holds and the edge model is resident.
- **4.2:** the best static configuration on the validation cells is `cloud/raw_image`.
- **4.3:** the [G4 specification](predictor-v2/g4-specification.md) was frozen by hash (`b2242d7f…`, commit `8af54cc`) before the evaluation profiles were derived from the `0948aa8` worktree, and the evaluation ran once.
- **4.4:**
  - median relative regret 0.00% (mean 0.06%);
  - top-1 agreement 98.1%;
  - mean regret 0.6 ms;
  - 42.6 ms (5.2%) faster than always-cloud = best static, with a CI that excludes zero;
  - 22 of 22 complete blocks, and the cell-level winner matched in 22 of 22 cells;
  - 0 unsafe or no-feasible decisions.
  - Evaluation-ID VQA accuracy is 70.67% for every configuration.

4.1 **Update the selector** to the five-configuration list (`ALL_CONFIGURATIONS`).
- Feasibility: memory budgets and int8 admission.
- Quality constraint, if phase 1.4 requires one.
- Load-aware inputs.

4.2 **Best static baseline** (roadmap phase AC): choose the best single configuration on **validation cells only**.

4.3 **Freeze** the selector and its oracle specification (hashes) **before** unsealing. Then run `derive_g2v2_tables.py --unseal-evaluation` and `evaluate-selector` on E01–E05.

4.4 **G4 metrics:**
- top-1 agreement;
- absolute and relative regret against the measured oracle, with the paired bootstrap;
- the same against always-edge, always-cloud, always-int8 and best static;
- quality on the evaluation IDs.

**Gate:** median regret ≤ 10%, at least 10 complete paired blocks, and a network holdout ([selector and oracle](design/selector-and-oracle.md)).

## Phase 5: adaptive runtime and dynamic protocol

**Status (20 November 2026): done locally.** Image build and reference rebuild are carried into phase 6.1, because they need AWS.
- **5.1 Runtime:**
  - The live adaptive policy (`prediction/r2/live.py`) senses its inputs, predicts with the frozen `r2_component`, selects with the five-configuration selector and runs the chosen configuration.
  - It runs as a sixth interleaved placement, so the fixed configurations give a per-request oracle on the same requests.
  - **Sensing** is chosen by you and lives in `scheduler/sensing.py`:
    - cloud load from the server's GPU power, which the health response now carries, with a threshold of 58 W from training;
    - edge load from a timing probe on the edge GPU, with its threshold set by a pilot rule;
    - the network from the segment's periodic probe.
  - **Traces:** campaign grids gain ordered trace segments, which switch netem, edge load and cloud load between segments.
  - **Analysis:** `scripts/campaign/analyze_g5.py`. The older `thesis_analysis.py` is left for phase 7.
- **5.2 Switching cost:** none measurable. The paired latency after the same versus a different configuration differs by −0.12% to +0.9%, and stage preparation takes 0 ms. So there is no hysteresis and no dwell; decisions are per request.
- **5.3:** [amendment A4](protocols/g2-v2-amendment-a4.md) fixes the sensing rules, the four traces × five segments × four repetitions (80 segments, 14,400 measured attempts, about 9 h), the metrics, the pilot go/no-go and the G5 gate.
- **5.4:** the [loopback rehearsal with fault injection](../results/g5-rehearsal-20261120/README.md) passed all checks. The full suite passes (471 passed, 1 skipped), as do Ruff and mypy.
- **5.5:** committed. Left for 6.1:
  - build and push the image;
  - rebuild the reference bundles, since `src` changed;
  - run the edge-probe calibration and the pilot;
  - pin the hashes.

5.1 **Bring the Phase 5–7 runtime up to the v2+ configuration space:**
- `adaptive-benchmark`, the dispatcher and the live monitor support five configurations;
- the monitor gains cloud-load and edge-load sensing (server state, edge GPU utilization);
- `thesis_analysis.py` still assumes three modes and needs updating;
- trace replay drives netem changes plus edge and cloud load switching on and off.

5.2 **Switching stability:** hysteresis and minimum dwell (roadmap phase AA), predeclared. Measure the cost of changing configuration, such as server warm state.

5.3 **Dynamic protocol (amendment A4):**
- **Traces:** network steps and ramps within the training ranges, plus load episodes (C2 and L1 on and off), on evaluation request IDs only.
- **Policies:** adaptive, each fixed configuration, and best static. The oracle reference comes from interleaved fixed runs on the same trace segments.
- **Metrics:** p50/p95 latency, SLA success, bytes, failure rate, VQA accuracy, switching frequency, and selector and monitor overhead.
- **Size, repetitions and session plan**, sized for about 24 h of pair time.
- **Pilot go/no-go rules.**

5.4 **Local rehearsal** on loopback with fault injection, as for G2 v2. Run the full test suite, Ruff and mypy.

5.5 **Freeze:**
- commit;
- build and push the image;
- rebuild the reference bundles if `src` changed (~1 h on AWS).

## Phase 6: G5 on AWS

6.1 **Bring the pair up:**
- start the stopped instances, or recreate them if 0.4 removed them;
- deploy the new image digest on both hosts;
- run the references and the preflight.

6.2 **Pilot**, then the go/no-go decision, then the protocol hash pin.

6.3 **Run** in sessions with the existing monitor and session scripts. Have someone sign in to AWS mid-session for runs longer than 5 h.

6.4 **Mirror S3 and the data volumes locally** and verify by key, size and hash, as on 19 November.

## Phase 7: final analysis

7.1 **Phase 7 paired analysis** (`thesis-analysis`, updated): adaptive against each baseline and the oracle on the same trace segments, with paired bootstrap CIs.

7.2 **Communication:** bytes per request by configuration and policy. Under the int8 split, bytes fall at a latency cost; the cloud sends the raw image.

7.3 **Overheads:** selector and monitor time per request, against the regret saved.

7.4 **Stability:** the number of switches and oscillation under each trace.

7.5 **Quality:** official VQA accuracy for each policy on the evaluation IDs, checked against parity.

7.6 **Final tables and figures** (list in phase 9).

## Phase 8: archive and teardown

8.1 **Archive:**
- the raw data mirrors (G2 v2+ and G5) with SHA256SUMS;
- tag the source commits used, with image digests and protocol hashes;
- keep an off-machine backup of `artifacts/aws-s3-mirror/` and the G5 mirror (external disk or university storage).

8.2 **Reproducibility package:** the commands, configs, protocol and amendments that regenerate every table and figure from the archived data.

8.3 **AWS teardown:**
- destroy the disposable stack;
- confirm that the backups are restorable before removing the data volumes;
- the retained stack (bucket, ECR, budget) has `prevent_destroy`, so decide after thesis acceptance;
- check that the account bill is zero.

## Phase 9: thesis writing (in parallel from phase 2)

| Chapter | Content | Available from |
| --- | --- | --- |
| 1 Introduction | Motivation, research question, contributions | now |
| 2 Background and related work | VLMs, split computing, edge–cloud offloading, latency prediction, adaptive scheduling | now |
| 3 System design | Configurations and split boundary, transport, monitor, predictor, selector | now |
| 4 Methodology | Testbeds (physical and AWS), frozen protocol and amendments, the network grid and its verification, the loads, the dataset and splits, validity gates, statistics | now |
| 5 Baseline results (G2 v2+) | Crossover map, load effects, split-inference analysis, quality parity, comparison with physical G2 | phase 2 |
| 6 Prediction and selection (G3, G4) | Predictor accuracy, ranking, regret against the oracle and static baselines | phases 3–4 |
| 7 Dynamic evaluation (G5) | Adaptive against baselines under traces, overheads, stability, bytes | phase 7 |
| 8 Discussion | When adapting pays off, why split inference loses here, generalization limits, threats to validity (emulated network, vGPU edge, one model, split exposure) | phase 7 |
| 9 Conclusion and future work | Reduced vision budget, queue awareness, energy, other VLMs | end |
| Appendices | Protocol, amendments, reproducibility, hashes | phase 8 |

Key figures:
1. Architecture.
2. Configuration pipelines.
3. Network grid with measured verification.
4. Crossover map by load state.
5. Load-effect plot.
6. Static-policy regret.
7. Predictor error (APE CDF per configuration).
8. G4 regret against the oracle.
9. G5 latency over a trace with the selected configuration.
10. Bytes against latency.

## Must-have checklist ([roadmap](roadmap.md#must-have-for-a-strong-bachelor-thesis))

| # | Item | Status |
| --- | --- | --- |
| 1–5 | CI, two machines, model equivalence, measured memory, controlled network | **Done**: physical G2 and AWS |
| 6–7 | Real task dataset, profiling | **Done**: G2 v2+, 43,200 requests |
| 8–10 | Fresh predictor calibration, held-out evaluation, G3 | Phase 3 (r1 failed on the physical pair) |
| 11, 14, 15 | G4, measured oracle, regret | Phase 4 |
| 12, 13 | Adaptive execution, fixed baselines | Phases 5–6 |
| 16 | Phase 7 paired evaluation | Phase 7 |
| 17 | Official VQA scoring | Phases 1.4, 4.4 and 7.5 |
| 18 | Statistical uncertainty and repetitions | Done for G2 v2+; continues in phases 3–7 |
| 19 | Reproducible artifacts | Phase 8 |
| 20 | Honest negative results | Split inference (phase 2); G3 if it fails |

**Already covered "strong additions":** edge and cloud load, and task awareness (short and long tasks, crop strata). **Phase 6 adds:** dynamic traces and hysteresis. **Out of scope unless time is left over:** the reduced-vision-budget configuration, queue awareness, energy, other VLMs, and learned or RL schedulers.

## Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| The predictor fails G3 again | Weaker adaptive claim | Predeclared simpler-model fallback; a load-aware additive model fits the observed structure |
| Small average adaptive gain | "Always cloud is enough" critique | Predeclare where adapting matters (overloaded cloud, high RTT); G5 traces exercise exactly those transitions |
| The V/E exposure on 19–20 November | Validity critique | Amendment A3, training-only design choices, and threats-to-validity disclosure |
| AWS capacity (g6.xlarge in us-east-1d) | G5 delay | Keep the stopped pair (0.4); the capacity-reservation retry in `up.sh` |
| AWS sign-in expires mid-session | The monitor can't stop the VMs on time | Sign in again before each 5 h window; idle watchdog and budget stop action |
| Changes to `src` after the G5 freeze | References invalid | Freeze only after phase 5.4; rebuilding references takes ~1 h |
| Budget | Lost runs | ~$65 contingency after G5; check spend before every phase |
