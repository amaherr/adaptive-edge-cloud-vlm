# G2 v2+ campaign: derived checks (phase 1)

Derived from the local mirror of the edge data volume (`artifacts/aws-s3-mirror/cloud-v1/edge/ebs-final/g2v2plus/campaign/`).

**Source state.** The tree matches the campaign's frozen commit `d5e116c` except the two analysis modules that reference verification exempts from its source hash: `benchmark/heterogeneous_correctness.py` and `benchmark/profile.py`. Both received the scope fixes below. The reference bundles verify against this tree.

## Fixes to the correctness check (`evaluate_rows`)

1. **Scope.**
   - **Before:** the check covered all manifest IDs and required two repetitions per ID. A campaign sub-block measures a subset of IDs once (its repetitions are separate sub-blocks), so every sub-block failed: unmeasured IDs counted as quality disagreements and as missing repetitions.
   - **Now:** `scope="measured"` checks the IDs a sub-block measured and needs one successful row per configuration. `profile.py` picks that scope from the run's scheduled trials, never from outcomes. Full-manifest runs with two or more repetitions (references, preflight smoke) keep the original `manifest` scope.
2. **Empty groups.** The coverage comprehension created empty groups for unmeasured IDs, which then read as non-repeatable. It no longer does.
3. **Timed-only requests.** The long task's manifest has `scoring: none` (protocol section 9: timed only). Its rows must carry no score and are left out of score equality. Their outputs are still compared with the references.

Tests: `tests/benchmark/test_heterogeneous_correctness.py`. The full suite passes: 428 tests, 1 skipped.

## Row-level correctness: 196/196 sub-blocks pass

[row_correctness.json](row_correctness.json), produced by `scripts/campaign/check_campaign_correctness.py`. This reads outputs only, not latency, so it covers every split without opening any sealed latency data ([amendment A3](../../../docs/protocols/g2-v2-amendment-a3.md)).

| Check | Result |
| --- | --- |
| Sub-blocks passing | 196/196 (176 short, 20 long) |
| Requests checked (ID × sub-block) | 8,640 |
| Rows differing from their same-device reference | 0 |
| IDs where edge, cloud and the lossless split disagree | 0 |
| Score disagreements (short task) | 0 |
| Measured status | 43,200 `ok` |

Every configuration's output equals its reference:
- `prompt_first` equals the synchronous control;
- int8 equals the int8 control.

## Official VQA accuracy and parity (training and validation IDs)

[official_vqa.json](official_vqa.json), produced by `scripts/vqa/score_campaign_vqa.py`:
- **Evaluator:** the pinned GT-Vision-Lab/VQA evaluator (`a013f00`) under PyPy 2.7.18.
- **Answers:** every measured short-task row gives one answer per ID and configuration, and they are identical across all of that ID's sub-blocks.
- **Evaluation IDs:** not scored. They stay sealed until G4 (A3).

| Split | IDs | Lowercased (primary) | Unchanged |
| --- | ---: | ---: | ---: |
| Training | 60 | 75.33% | 55.33% |
| Validation | 30 | 79.67% | 59.67% |

**These figures are identical for all five configurations**: `edge/full`, `cloud/raw_image`, and the synchronous, `prompt_first` and int8 split configurations. No question's accuracy differs between any pair. Parity, at most 2 pp, holds with 0.0 pp difference, so the selector can treat quality as independent of configuration (section 9). The roughly 20-point gap between unchanged and lowercased is the evaluator's case-sensitive handling of "Yes"/"No" found in the prompt calibration.

## Section 9 oracle (training cells): pending

`scripts/campaign/derive_g2v2_tables.py` could not run locally. The campaign plan and every run plan store container paths (`/app/configs/…`, `/app/data/…`), and the run plans are hash-checked by the validation reports, so they cannot be rewritten. Mapping `/app` (and `/artifacts`) inside WSL needs root, or the derivation has to run inside the frozen image. The [roadmap](../../../docs/thesis-completion-roadmap.md) tracks this as step 1.3.
