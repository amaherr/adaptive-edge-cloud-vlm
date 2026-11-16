# G2 v2 baseline protocol

**Status: draft.** Section 3 (prompt contract and selection rule) and section 4 (dataset construction) are **frozen and executed**: they were committed before any calibration answer was generated or scored, and were applied exactly as written. All other sections stay a draft until the off-site rehearsal and edge pilot in the [preparation roadmap](../g2-v2-preparation.md) confirm feasibility. At that point the whole document is frozen, and the campaign grid config pins its SHA-256. Changes after freezing are allowed only as dated amendments at the end of this document, written before the affected data is collected.

## 1. Purpose and scope

G2 v2 is the physical latency baseline the rest of the thesis builds on. Predictor V2 r2 (G3), the selector against the measured oracle (G4), and later adaptive evaluations all use only G2 v2 data. The original G2 ([results/physical-g2-20261110](../../results/physical-g2-20261110/README.md)) stays archived and reported. Its data are not pooled with G2 v2.

Compared with G2, five things change:
1. edge/full runs with swap residency;
2. a short-answer prompt contract;
3. fresh request IDs and network conditions;
4. placements interleaved per request;
5. edge GPU load as a factor, with resource state logged per request.

Everything else stays as in G2:
- the hosts, the model, the precision on each host;
- the transport and the one-TCP-connection-per-request client;
- the netem shaping method;
- the heterogeneous-GPU correctness protocol.

## 2. Fixed system contract

| Item | Value |
| --- | --- |
| Model | Moondream2, revision `9a7d4024050840e001defacec2b00727e89149e6`, artifact SHA-256 `70a7d94c…7403f` ([moondream2.yaml](../../configs/model/moondream2.yaml)) |
| Precision | BF16 on both GPUs (the RTX 2070 Super emulates BF16; reported as a limitation) |
| Decoding | Greedy, `max_new_tokens` 32, batch size 1 |
| Placements | `edge` (full model on the edge GPU, swap residency); `cloud` (raw image sent, full model on the cloud GPU); `distributed` (vision on the edge GPU, projected tensor sent, text on the cloud GPU) |
| Edge weights | One edge process holds one swap-residency adapter, shared by edge/full and the edge half of distributed |
| Transport | One new TCP connection per request; no connection reuse |
| Correctness | [Heterogeneous-GPU correctness v2](../physical-testbed/correctness-criteria.md#heterogeneous-gpu-correctness-v2-approved), with a reference bundle regenerated for the v2 manifest |

## 3. Prompt contract and selection rule (frozen)

The G2 prompt sent the VQA question verbatim. Moondream answered in full sentences, and every placement scored 2.0% on official VQA. G2 v2 appends a fixed short-answer instruction. The template is applied when the manifest is frozen, so the request `prompt` field holds the final text. The adapter, model configuration and model contract hash do not change.

### 3.1 Candidates

| ID | Template (`{question}` is the verbatim VQA question) | Eligible |
| --- | --- | --- |
| `P0` | `{question}` | No: the G2 contract, run only as a reference |
| `P1` | `{question} Answer with a single word or phrase.` | Yes |
| `P2` | `{question} Answer in one or two words.` | Yes |
| `P3` | `{question} Respond with only the answer, in as few words as possible.` | Yes |

### 3.2 Calibration set

- 200 VQA v2 val2014 questions, built with the method in section 4.1, with seed `2611`.
- The set excludes every question and image in the G2 manifest ([physical_vqa_v1.jsonl](../../data/manifests/physical_vqa_v1.jsonl)).
- It has no crop stratification and allows one question per image.
- The G2 v2 dataset (section 4) excludes every calibration question and image, so calibration answers never touch G2 v2 data.

### 3.3 Procedure

1. Run every template on the edge laptop: edge placement, swap residency, one warmup, one measured repetition, same model contract. Decoding is greedy, so one repetition fixes the answers.
2. Export the predictions unchanged, without post-processing, and score them with the pinned official evaluator. That evaluator is GT-Vision-Lab/VQA at commit `a013f0043c1e2cdc995922dfe257f7149aa9af06`, run under PyPy 2.7.18 / 7.3.15, the same setup as G2. A failed request scores as an empty answer.
3. For each template, record:
   - overall accuracy;
   - accuracy by answer type (yes/no, number, other);
   - mean output token count;
   - the number of answers that reach the 32-token cap;
   - the number of failed requests.

### 3.4 Selection rule

1. Let *A** be the highest overall accuracy among eligible templates.
2. A template is *near-best* if its accuracy is at least *A** − 1.0 percentage point.
3. Among near-best templates, choose the one with the fewest 32-token cap hits. Break remaining ties by lowest mean output token count, then by lowest template ID.
4. **Floor.** If *A** is below 50.0%, no template is selected. The protocol must be amended (new candidates, evaluated on a new calibration set) before G2 v2 can proceed.
5. `P0` is reported for comparison and can never be selected.

The selection record (all metrics, the chosen ID, and the hashes of the calibration manifest and predictions) is committed next to the calibration results, before the G2 v2 dataset is imported.

**Executed 15 November 2026.** The rule selected **P1** at 56.7% official accuracy. P2 scored 57.2% and was also near-best; P1 won the tie-break on mean output tokens (1.47 against 1.50). No template hit the 32-token cap. See [results/prompt-calibration-20261115](../../results/prompt-calibration-20261115/README.md).

### 3.5 Long-answer task (frozen 16 November 2026, before any candidate was run)

**Why.** A pre-run estimate (`scripts/campaign/estimate_g2v2_winners.py`, from G2's per-stage timings and the W8 edge pilot) predicts that with P1's ~1.5-token answers the distributed placement never wins any cell. Its advantage, decoding on the cloud while the edge is loaded, needs answers long enough that the decode dominates. Measured edge decode cost is 24.7, 37.3 and 66.5 ms per token at L0, L1 and L2, against about 22.6 ms on the cloud. G2 v2 therefore adds a second task bucket, `vqa-long-answer-32-token-budget`: the same images and questions with a prompt that asks for a long answer. The generation contract is unchanged (greedy, 32-token cap), so the model contract hash is the same.

**Scoring.** The long-answer bucket is timed only. Long answers cannot be scored by official VQA accuracy (P0 scored 1.0%), so its manifest uses `scoring: none`. Correctness still uses the per-GPU same-device references (section 8), built separately for the long-answer manifest.

**Candidates** (`{question}` is the verbatim VQA question):

| ID | Template |
| --- | --- |
| `D1` | `{question} Explain your answer in detail.` |
| `D2` | `{question} Answer in a full sentence, then describe the relevant details of the image.` |
| `D3` | `Describe the image in detail, then answer this question: {question}` |

**Procedure.** Use the section 3.2 calibration set and the section 3.3 run setup: edge placement, swap residency, one warmup, one measured repetition. For each template, record the mean output tokens, the number of answers reaching the 32-token cap, the number of failed requests, and median edge latency.

**Selection rule.**
1. A template is eligible if it has no failed request and a mean of at least 24.0 output tokens.
2. Among eligible templates, choose the one with the most 32-token cap hits. Break ties by higher mean output tokens, then by lowest template ID.
3. **Floor:** if no template is eligible, none is selected, and the protocol must be amended with new candidates before the long-answer task can be added.

**Long-answer request set.**
- Each split's long-answer IDs are a subset of the same split's short-answer IDs: 30 training, 15 validation and 15 evaluation.
- The subset is drawn with seed `3907`, stratified by crop count, so it keeps the 20% two-crop share (6/3/3 two-crop).
- Long-answer request IDs are the short-answer IDs with `-long` appended, so the two buckets never share a key.

**Executed 16 November 2026: no template eligible.** All three ran 200/200 without failures, but none reached the 24-token floor:

| ID | Mean output tokens | 32-token cap hits | Median edge latency |
| --- | ---: | ---: | ---: |
| `D1` | 17.45 | 46 | 1,447 ms |
| `D2` | 16.11 | 20 | 1,384 ms |
| `D3` | 13.05 | 21 | 1,313 ms |

Moondream mostly answers the question in one sentence even when asked for detail. Under rule 3, the protocol is amended as follows.

**Amendment A1 (16 November 2026, before any candidate below was run).** These three candidates replace D1–D3. The calibration set, procedure and selection rule above are unchanged.

| ID | Template |
| --- | --- |
| `E1` | `Describe this image in detail.` (the question is not used: a detailed-description task on the same images) |
| `E2` | `{question} Answer in at least three full sentences, describing everything in the image that relates to the question.` |
| `E3` | `Write a detailed paragraph describing this image, including the answer to this question: {question}` |

If no E template is eligible either, the long-answer task is dropped from G2 v2 and the run proceeds with the short-answer task only. Distributed is then reported as never winning on this hardware pair, with the pre-run estimate as the explanation.

## 4. Dataset construction (frozen)

### 4.1 Candidate order (shared by the calibration set and the dataset)

1. Start from the official val2014 question file (SHA-256 `f34d9c99…44811`) with its annotations (SHA-256 `0564650e…31c4bf`). Sort question IDs ascending, then shuffle them with Python's `random.Random(seed)`.
2. Walk the shuffled list. Skip a question if any of these holds:
   - its question ID or image ID is in an excluded manifest;
   - its image already holds a question earlier in the walk;
   - it does not have exactly ten annotations.
3. The candidate list is the first *N* questions that survive. The images for those candidates are fetched from the official COCO bucket and verified as JPEG.

### 4.2 G2 v2 dataset

- **Size:** 120 questions, seed `3907`.
- **Excluded:** the G2 manifest and the calibration manifest.
- **Candidates:** the first 400 in candidate order.
- **Crop strata.** The pinned Moondream tiling (378-pixel crops; see [stage diagnosis](../../results/physical-g2-20261110/predictor_v2/r1/stage_diagnosis/README.md)) gives exactly two crops when either image side is at most 378 pixels. Otherwise it gives nine or more for COCO-sized images. G2 had 15% two-crop requests, too few to fit crop-aware features. G2 v2 fixes the two-crop stratum at 24 of 120 (20%).
- **Selection:** walk the candidate order, take the first 24 two-crop questions and the first 96 others, and stop. If the 400 candidates can't fill both strata, import fails, and the candidate count may only be raised by amendment.
- **Prompt:** the prompt template chosen in section 3.4.
- **Revision:** `vqa-v2-val2014-physical-v2`.

### 4.3 Request split

| Split | Request IDs | Two-crop | Other | Network conditions |
| --- | ---: | ---: | ---: | --- |
| Training | 60 | 12 | 48 | T01–T12 |
| Validation | 30 | 6 | 24 | V01–V05 |
| Evaluation (protected test) | 30 | 6 | 24 | E01–E05 |

Within each crop stratum, request IDs are shuffled with seed `3907` and assigned to training, validation and evaluation in that order. Request IDs are measured only under their own split's conditions. G2 crossed every ID with every block; G2 v2 does not. The split file is [physical_vqa_split_v2.json](../../configs/evaluation/physical_vqa_split_v2.json).

**Executed 15 November 2026.**
- **Manifest:** [physical_vqa_v2.jsonl](../../data/manifests/physical_vqa_v2/physical_vqa_v2.jsonl), SHA-256 `4f8bfe63f245db948274d7e0717bde3bab388e5ffe65505a3f841a35e3f9b89d`. It holds 120 questions (24 two-crop), filled from the first 126 of the 400 candidates, with prompt P1.
- **Split:** 60/30/30 requests, with 12/6/6 two-crop. SHA-256 `9133a9d3011d4c92d377abd30b29265ee1179cd03d157f1bcc7f81be64143dd1`.
- **Model contract:** unchanged from G2 (`e7ef0180…94aa1`).
- **Overlap:** no question or image is shared with G2 or with the calibration set.

## 5. Network conditions (draft)

The rate applies to egress on both hosts. Added RTT is split evenly, with each host's egress delay set to half of it. Every netem qdisc sets `limit 10000` packets and must report zero drops before and after each sub-block (`tc -s qdisc`). A condition passes the same live-measurement checks as G2: iperf3 receiver rate within ±20% of the target in both directions, and echo RTT within max(5 ms, 20%) of the unshaped baseline plus the added RTT.

| ID | Rate (Mb/s) | Added RTT (ms) | | ID | Rate (Mb/s) | Added RTT (ms) |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| T01 | 5 | 20 | | V01 | 7 | 50 |
| T02 | 15 | 20 | | V02 | 35 | 20 |
| T03 | 25 | 20 | | V03 | 60 | 50 |
| T04 | 45 | 20 | | V04 | 120 | 20 |
| T05 | 80 | 20 | | V05 | 80 | 150 |
| T06 | 150 | 20 | | E01 | 20 | 50 |
| T07 | 100 | 5 | | E02 | 70 | 20 |
| T08 | 200 | 5 | | E03 | 120 | 100 |
| T09 | 100 | 100 | | E04 | 175 | 50 |
| T10 | 45 | 100 | | E05 | 50 | 150 |
| T11 | 150 | 150 | | | | |
| T12 | 25 | 150 | | | | |

Every validation and evaluation condition lies inside the training rate and RTT ranges, and none was used in G2. If the on-site pilot shows the edge uplink can't sustain 200 Mb/s within tolerance, T08 becomes 150 Mb/s / 5 ms and E04 becomes 140 Mb/s / 50 ms. This substitution is fixed now, so it isn't a post-hoc choice.

## 6. Edge GPU load (draft)

| Level | Competitor ([levels_v1.json](../../configs/edge_load/levels_v1.json)) | Edge/full median under P1 | Purpose |
| --- | --- | ---: | --- |
| L0 | None | 1.27 s | Unloaded edge |
| L1 | 512 MiB working set, 10 ms busy / 10 ms idle (achieved duty 0.60) | 1.85 s | Edge near the likely cloud/distributed range |
| L2 | 512 MiB working set, 20 ms busy / 0 ms idle (achieved duty 0.99) | 2.81 s | Edge clearly slower than cloud |

[Calibrated 15–16 November 2026](../../results/edge-load-calibration-20261116/README.md). Both levels reproduced within 1.5% over three separate runs. They load the GPU with compute on a fixed 512 MiB working set. The draft's 1 GiB L2 was dropped: it sat within about 50 MiB of the edge's dedicated-memory limit, made latency insensitive to the duty cycle, and gave a p90 about 55% above the median. That is uncontrolled memory pressure rather than load.

The competitor (`python -m adaptive_vlm.benchmark.edge_load_worker`) runs in its own process. `interleaved-benchmark --edge-load-config … --edge-load-level L1|L2` starts it after the model has loaded. It verifies the heartbeat before every attempt; if the generator is not alive, the command exits with code 76 and the supervisor resumes. It stops the generator at the end. Its level, liveness, heartbeat age and achieved duty are in every record.

**Pilot check.** If the pilot shows the median cloud latency at 35 Mb/s / 20 ms (V02) below 1.6 s or above 2.6 s, L1 does not sit near the cloud/distributed range. The levels are then re-chosen from the calibration curve by a dated amendment, before any G2 v2 data is collected.

All conditions run at L0. L1 and L2 run on six conditions:
- training: T02, T04, T06, T09;
- validation: V03;
- evaluation: E03.

Each split therefore contains cells where the best placement changes with load.

## 7. Execution design (draft)

- **Cell:** one condition at one load level. There are 34 cells: 22 at L0 and 12 at L1 or L2.
- **Sub-blocks:** each cell runs as three sub-blocks, one per repetition (0, 1, 2), so every request ID is measured at three separate times. The 102 sub-blocks run in an order shuffled with seed `5113`, constrained so that the three sub-blocks of a cell are never adjacent.
- **Within a sub-block:**
  - one excluded warmup per placement;
  - then each request ID of the split, in a per-sub-block seeded order;
  - for each request, all three placements, in a per-request seeded order.
- **Edge state preparation:** before each timed request, the adapter is set, untimed, to the stage a run of same-placement requests would leave active: text before edge/full, vision before distributed. The time spent is logged as `stage_prepare_ms`. Edge/full still pays both of its stage swaps inside the timed request.
- **CUDA failures:** a `CUDA error: unknown error` leaves the process's CUDA context unusable. It happened once in about 730 swap-residency requests during calibration. The supervisor must restart the edge process and resume the sub-block, rather than stopping the campaign. Restarted attempts are recorded, and failures still count toward the section 8 limit. Core dumps are disabled for every campaign process (`ulimit -c 1`; WSL's piped crash handler ignores `ulimit -c 0`), so a crash can't fill the Windows system drive.
- **Resource state:** each record carries a snapshot taken outside the timed region (`interleaved-benchmark --resource-state`, `adaptive-server --report-resource-state`):
  - **Edge, read synchronously right before the timer starts:** NVML GPU clocks, temperature, power, utilization, used and free VRAM, performance state and clock-event reasons; host available memory and swap; competitor status (level, alive, heartbeat age, achieved duty).
  - **Edge Windows GPU counters:** dedicated and shared usage on the CUDA adapter. One read takes about 1.4 s, so a background `powershell.exe` streams them once a second, and each record stores the latest sample and its age.
  - **Cloud:** the same NVML and host fields, polled every 0.5 s in the server's background thread and returned with each cloud and distributed response as `server_state_*`, with the sample's age. Reading the cached sample adds no latency.

### Expected attempt count

| Part | (Condition, ID) pairs | × reps × placements | Measured attempts |
| --- | ---: | ---: | ---: |
| L0 | 12×60 + 5×30 + 5×30 = 1,020 | × 3 × 3 | 9,180 |
| L1 | 4×60 + 30 + 30 = 300 | × 3 × 3 | 2,700 |
| L2 | 300 | × 3 × 3 | 2,700 |
| **Total** | | | **14,580** |

There are also 306 excluded warmups (102 sub-blocks × 3 placements).

## 8. Validity gates (draft)

A sub-block is valid only if:
1. it passes the pre and post network measurements and the zero-drop check;
2. the heterogeneous-correctness reference was verified at campaign preflight;
3. every planned attempt has a record, with at most 1% failed attempts and fewer than three consecutive failures;
4. at L0, the edge never spills: Windows *dedicated* usage on the CUDA adapter stays at least 128 MiB below the adapter's maximum (3,960 MiB on the RTX 3050) at every request snapshot. Shared usage is not the spill signal under swap residency, because Windows counts the pinned host master copies (about 4.4 GB) as GPU shared usage. The resident model on this GPU spilled with dedicated usage at 3,960 MiB and about 670 MiB shared; swap ran at 3,271 MiB dedicated;
5. at L1/L2, the competitor was verified running at the first and last request snapshots.

An invalid sub-block is rerun as a whole. A dataset with any invalid sub-block left is not G2 v2.

## 9. Analysis plan (draft)

- **Latency.** For each cell and placement: median, p90 and the bootstrap 95% CI of the median end-to-end latency over all of the split's IDs and repetitions (seed `1707`, 2,000 resamples, as in G2).
- **Measured winner.** For each cell, the placement with the lowest median latency, plus its margin over the runner-up. Cells where the CIs of the top two overlap are reported as ties.
- **Quality.** Official VQA accuracy for each placement on each split, scored as in section 3.3. *Proposed, to be decided before freezing:* the primary score lowercases each answer before export. That is standard VQA submission practice, and it avoids the evaluator's case-sensitive comparison when all references agree; the [calibration](../../results/prompt-calibration-20261115/README.md#diagnostic-the-evaluators-case-handling) found that comparison costs about 22 points on capitalized "Yes"/"No" answers. The unchanged score is reported alongside. Placement quality parity: the accuracy difference between any two placements, on the same IDs, must be at most 2 percentage points. If not, the selector must treat quality as placement-dependent.
- **Isolation.** Training cells only, for anything fitted (Predictor V2 r2, G3). The validation cells are opened once for G3. The evaluation cells stay sealed until the selector is frozen for G4.

## Amendments

None.
