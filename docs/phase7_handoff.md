# Phase 7 handoff — task-aware evaluation pipeline

Date: 2026-10-26 (Africa/Cairo). This handoff follows the [Phase 7 protocol](phase7_evaluation_protocol.md) and [scheduler plan](dynamic_scheduler_implementation_plan.md). **The local analysis pipeline is implemented and verified; full G7 is open.** The separate cloud PC, controlled physical link, and official VQA files are still unavailable. The current calibration's local fastest-choice ranking failed G3, and the Phase 4 performance gate is open. Do not report the local comparison as an adaptive improvement claim.

## Changes made

- Froze the [local specification](../configs/evaluation/phase7_local_diagnostic.json) in commit `b04b156` before running the Phase 7 analyzer. It pins the model/manifest/calibration hashes, predictor-training IDs, static-validation IDs, evaluation IDs, SLA, bootstrap seed, task label and uncontrolled network regime. The prospective [protocol](phase7_evaluation_protocol.md) defines the physical experiment and G7 gate.
- Added `analyze-thesis` to verify complete four-policy schedules, row/run/plan provenance, split disjointness, decision audit, token equivalence, and matching model/config/manifest hashes. It picks best-static using validation SLA success then median latency, reports held-out paired latency/SLA differences and measured-oracle regret, and retains failures in denominators. Hierarchical bootstrap samples network blocks, then request IDs within each block; the report states its counts and seed. It includes successful median/p95 latency, wire bytes, status counts, decision distribution, selection/monitor overhead and prediction error against matched fixed observations. `paired_trials.jsonl` and `report.json` include source hashes. The pipeline checkpoint is commit `1cecb69`.
- Added input-only `task_type` classification to frozen requests and adaptive rows. VQA is `vqa_short_answer`; exact-match prompts are `labeled_short_answer`; other prompts are `unscored_prompt`. A VQA replay must declare its type, and each adaptive campaign must contain one task type. The exact prompt token count and frozen task bucket are already computed before selection; the predictor uses its training-median output-length estimate unless an independently predeclared estimate is supplied. No generated answer is used as a scheduler input.
- Added `prepare-vqa-eval-subset` and `export-vqa-predictions`. The former writes official-schema question/annotation JSON restricted to the imported IDs, which the official loader requires; the latter writes official-format prediction JSON for one repetition, exporting failed requests as empty answers and recording provenance. `scripts/run_official_vqa_eval.py` invokes a pinned upstream evaluator and hashes its source and inputs. The upstream [GT-Vision-Lab/VQA evaluator](https://github.com/GT-Vision-Lab/VQA/tree/master/PythonEvaluationTools) uses Python 2 syntax, so run it in a compatible environment. It has **not** run here because the official files are absent.
- Updated the README's project state, CLI, VQA workflow and Phase 7 reproduction steps. No model or official dataset files were added to Git.

## Local descriptive result

The [report](../artifacts/phase7/2026-10-26/local-diagnostic-3/report.json) and [paired table](../artifacts/phase7/2026-10-26/local-diagnostic-3/paired_trials.jsonl) use the already archived Phase 2 fixed and Phase 5 adaptive CPU/loopback runs. The frozen specification chooses `correctness-001` through `010` for static-policy validation and reports `011`–`012` for evaluation. The predictor had trained on `001`–`010`; its two evaluation IDs were already inspected in prior phases. One measured repetition and uncontrolled policy order make all intervals **descriptive only**.

| Held-out local measure | Adaptive | Edge | Cloud | Projected split |
| --- | ---: | ---: | ---: | ---: |
| Successful / attempted | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| Median request latency | 28.605 s | 27.442 s | 27.957 s | 28.881 s |
| Nearest-rank p95 latency | 29.679 s | 28.425 s | 28.763 s | 30.318 s |
| Mean application wire bytes / attempt | 2,987,709 | 0 | 1,795 | 2,987,716 |
| Success within frozen 60 s SLA | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |

Validation chooses **edge** as best-static. On the two evaluation pairs, adaptive selected projected split twice and averaged **1.162 s slower than edge**; the request-cluster 95% bootstrap interval is 1.070–1.254 s. The same signed value is the measured-oracle regret because edge happened to be fastest in both pairs. Relative to cloud, adaptive was 0.648 s slower on average; relative to fixed projected split, it was 0.276 s faster. Two observations cannot establish an improvement or stable uncertainty bound. Candidate prediction median APE against these separately timed fixed runs was 3.64% edge, 2.87% cloud and 4.89% split. The earlier Phase 5 audit found zero adaptive-versus-selected-control token mismatches; the Phase 7 report also records fixed-policy token comparisons. Official VQA accuracy, measured edge-memory peak, and physical goodput remain `null`. `g7_passed=false`.

The [self-contained archive](../artifacts/phase7/2026-10-26/local-diagnostic-3.tar.gz) contains the specification, all four run plans/manifests/raw JSONL files, calibration and predictor gate context, report, paired rows and [per-file hashes](../artifacts/phase7/2026-10-26/local-diagnostic-3/file_hashes.json). Its 18 hashed payload files were verified, and an analysis rerun using only the bundled inputs reproduced `report.json` and `paired_trials.jsonl` byte for byte. Archive SHA-256: `fb552994d808bcc2c72e9ad407f43164aae599c1ed5ad71defc55124aade2917`. `artifacts/` is Git-ignored; preserve the archive when moving machines.

## Reproduce and inspect

With the original local artifacts available, run this from the repository root; no model weights or server are needed:

```bash
.venv/bin/adaptive-vlm analyze-thesis \
  --specification configs/evaluation/phase7_local_diagnostic.json \
  --adaptive-run artifacts/phase5/2026-10-26/local-1 \
  --fixed-runs artifacts/phase2/2026-10-26/local-1/edge \
    artifacts/phase2/2026-10-26/local-1/cloud \
    artifacts/phase2/2026-10-26/local-1/distributed \
  --output-dir artifacts/phase7-reproduction
```

To reproduce using only the Phase 7 archive, extract it and replace the five paths above with `local-diagnostic-3/inputs/specification.json`, `inputs/adaptive`, `inputs/edge`, `inputs/cloud`, and `inputs/distributed`. The generated `report.json` shows frozen hashes, status and metric tables, validation choice, paired bootstrap results, prediction errors and oracle regret. `paired_trials.jsonl` shows one record per held-out trial. Preserve complete raw rows; the report is a derived view.

## Complete the prospective physical and VQA campaign

1. Verify the second host using the [deployment guide](../README.md#using-a-separate-cloud-pc) and close G0 physical. Freeze new task manifests and the 32-token generation contract. Import the official VQA v2 val2014 subset using the README command. Define a separate short-answer VQA bucket and a distinct descriptive task bucket if the pinned model supports it; profile and calibrate each. Do not reuse the synthetic task fit or mix a 100/300-token contract into the 32-token campaign.
2. Measure controlled uplink/downlink/RTT blocks and memory peaks on the real host pair. Freeze a new Phase 2 profile and a Phase 3 split with disjoint fitting, static-validation and evaluation request IDs **and** held-out network blocks. Refit until the new G3 predictor and G4 selector gates pass; replay the physical trace and validate G6 health/safety. Record shaper settings and independent achieved-network measurements for every policy/block. A replayed state label alone is not proof of a physical network condition.
3. Copy the local Phase 7 JSON specification under a new campaign name and replace all hashes, task label, IDs, SLA, bootstrap settings and block IDs **before running the evaluation**. Use `design: "prospective"` and nonempty, disjoint validation/evaluation network-block lists. Use the same randomized per-request schedule, warmup count, repetition count and model contract for all four policies within each block; rotate the policy execution order between blocks. Run fixed `benchmark` controls and `adaptive-benchmark --monitor-mode live` using the fresh calibration/replay and passing G3/G4 reports. A fresh output directory is required per run; keep failures and all warmups. Supply the actual measured edge-memory peak/budget to the runtime rather than inferring it from RSS.
4. Run `analyze-thesis` on each matching block campaign, preserving the raw runs and the report. The analyzer checks the adaptive rows' declared block IDs against the specification; independent physical-network logs and fixed-policy condition verification still need to be archived and reviewed. The current CLI analyzes one matched run quartet per invocation, so combine per-block results only with a separately frozen aggregation protocol; do not pool a local loopback diagnostic with physical observations. Use the official scorer below for each VQA policy/repetition. G7 remains open until all predeclared cases and external evidence are present.

For a VQA campaign, after the four raw runs complete:

```bash
.venv/bin/adaptive-vlm prepare-vqa-eval-subset \
  --manifest data/manifests/vqa_val2014_seed42.jsonl \
  --questions data/raw/vqa/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations data/raw/vqa/v2_mscoco_val2014_annotations.json \
  --output-dir artifacts/vqa-official-subset

.venv/bin/adaptive-vlm export-vqa-predictions \
  --manifest data/manifests/vqa_val2014_seed42.jsonl \
  --run-dir artifacts/NEW_VQA_POLICY_RUN \
  --repetition 0 --output artifacts/vqa-policy-r0.json

python2 scripts/run_official_vqa_eval.py \
  --vqa-root /path/to/pinned/GT-Vision-Lab-VQA \
  --questions artifacts/vqa-official-subset/questions.json \
  --annotations artifacts/vqa-official-subset/annotations.json \
  --predictions artifacts/vqa-policy-r0.json \
  --output artifacts/vqa-policy-r0-official-eval.json
```

Pin the official evaluator checkout commit and retain the generated source/input hashes and per-question scores. Repeat export/scoring for every policy and repetition. The runner's provisional VQA `quality_score` cannot replace this result.

## Verification and remaining gates

The Phase 7 tests cover validation-only static selection, failure-inclusive SLA accounting, paired-success coverage, training/evaluation leakage rejection, incomplete raw rows, deterministic official-format VQA export with failed answers and matching subset creation. Ruff, strict mypy and the full test suite pass; the complete suite has **171 passed** including loopback integration. The local report and archive hashes are verified. The VQA adapter's CLI was checked but official scoring could not run without the files and compatible evaluator checkout. The physical-host campaign, official VQA scoring, fresh held-out controlled network comparisons, measured edge-memory peak, and G3/G4/G6 physical gates remain open. A complete negative adaptive result on that prospective campaign would still satisfy the thesis evaluation design.
