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

## Section 9 oracle: training cells

[oracle-training/oracle_cells.csv](oracle-training/oracle_cells.csv) and [oracle_cells.json](oracle-training/oracle_cells.json) come from `scripts/campaign/derive_g2v2_tables.py profiles --splits training` followed by `oracle` (training is the default; there is no `--unseal-evaluation`). The run used the tree described above.

**How it was run.**
- The campaign plan and run plans store container paths (`/app/…`, `/artifacts/…`), and the validation reports hash-check the run plans, so the paths can't be rewritten.
- The derivation ran in WSL (20 November 2026) with two temporary links:
  - `/app` → the repository;
  - `/artifacts` → `artifacts/aws-s3-mirror/cloud-v1/edge/ebs-final`.
- The reference paths came from `artifacts/g2v2plus-aws-campaign-derived/contract_local.json`.
- Both links were removed afterwards.

**Profiles.**
- 108 training sub-blocks were profiled.
- All 108 pass `heterogeneous_correctness`, with scope `measured`, and the controlled-network gate.
- The legacy strict CPU gate is false by design: it doesn't apply to heterogeneous GPUs.

**Oracle:** 54 cells, 6,120 attempts. Bootstrap 95% CI of each median: seed 1707, 2,000 resamples.

| | |
| --- | --- |
| Cell winners | `cloud/raw_image` 48, `edge/full` 6, distributed 0 |
| Lossless-only winners | identical (int8 never wins a cell) |
| Agreement with the descriptive first look | 54/54 cells |
| Ties (top two CIs overlap) | 2: T02-L0-C2 (edge ahead by 10.0 ms) and T12-L1-C2 (cloud ahead by 18.5 ms) |
| Clear edge wins | T01, T09, T10, T11 and T12 at L0/C2, by 164–380 ms |
| Cell margin | median 429 ms (range 10–1,016) |

**Attempt level (6,120 attempts):**
- **Fastest configuration per attempt:**
  - cloud in 83.2% of attempts;
  - edge in 16.5%;
  - int8 in 0.3%;
  - synchronous and `prompt_first` never.
- **Cell winner vs per-request choice:** always using the cell winner loses to the per-attempt fastest in 6.8% of attempts, at a mean cost of 7.2 ms (median 0). Per-request selection within a cell has almost nothing to gain over per-cell selection.
- **Mean regret of always choosing one configuration:**
  - cloud 38 ms;
  - edge 613 ms;
  - int8 753 ms;
  - synchronous 1,183 ms;
  - `prompt_first` 1,188 ms.

**Kept outside git** (`artifacts/g2v2plus-aws-campaign-derived/`):

| File | SHA-256 |
| --- | --- |
| `oracle-training/oracle_attempts.jsonl` (4.4 MB) | `f520771448fbacdcdfed910be201a378ea2a2a68c5b293efbdd2cf7a4cffddc2` |
| `oracle-training/oracle_cells.csv` | `cf31c4dfd003b1ee041326cdd16de4f59dbefe1573abd977c79a314e5f6ae479` |
| `oracle-training/oracle_cells.json` | `d5065fde3f98d2ce75a4ac51bf2fb510b34b21becf2cd6326cf623e9ff60f727` |
| `profiles/profiles_index.json` | `f4e0d023048a7e684e431d92b7d9046fb4e37c5dd5b1d2db98a7331e5348d39a` |

**Task features: not derived yet.**
- The `features` step failed because the WSL environment lacks `tokenizers` (pinned `0.21.4` in `requirements/model-cu128.lock`).
- The features feed the G3 predictor (phase 3), not the oracle.
