# G2 v2+ campaign on AWS (18–19 November 2026)

The full A2b campaign, run under the frozen G2 v2 protocol (`protocol_sha256` `a972ecdc…e1d5`). This folder records the run and checks its integrity. The [derived checks](derived/README.md) add row-level correctness, official VQA accuracy and the training-cell section 9 oracle. The **[baseline report](baseline/README.md)** (phase 2) states the G2 v2+ pass decision with the gate evidence and analyses the training cells. G3 and G4 are still to come.

- **Hosts:** edge `g6f.2xlarge` (L4-6Q vGPU), cloud `g6.xlarge` (L4), us-east-1d.
- **Image:** `sha256:4049c069dc79…` (commit `d5e116c`), the same digest on both hosts in every session.
- **Grid:** [aws_network_grid_v2plus.json](../../configs/profiles/aws_network_grid_v2plus.json):
  - 196 sub-blocks: 88 short cells and 10 long cells, two repetitions each;
  - order seed 7219.
- **Raw data (git-ignored):**
  - `artifacts/aws-s3-mirror/` mirrors the whole artifact bucket (19,028 objects, 694 MB), checked against S3 by key and size.
  - The campaign folder is `cloud-v1/edge/ebs-final/g2v2plus/campaign/`, a full copy of the edge data volume made after the run. It includes the preflights and final status that the per-sub-block checkpoint leaves out.
  - `cloud-v1/edge/g2v2plus/` is the checkpoint that the supervisor uploaded during the run.

## Outcome

| | |
| --- | --- |
| State | `completed` |
| Sub-blocks valid | 196/196, all on the first attempt, with no restarts and no validation reasons |
| Measured attempts | 43,200/43,200, all `ok` |
| Warmups | 980 |
| Warnings and errors | none |

## Sessions

| Session | Preflight (UTC) | Ended (UTC) | Sub-blocks valid at the end | How it ended |
| --- | --- | --- | --- | --- |
| 1 | 18 Nov 19:25 | 19 Nov 03:07 | 68 | Stopped on the session target (SIGTERM, 200 s grace). |
| 2 | 19 Nov 03:23 | 19 Nov 11:00 | 132 | Stopped on the session target. |
| 3 | 19 Nov 11:17 | 19 Nov 19:12 | 196 | Campaign completed. |

Each restart repeated the preflight against the campaign contract and reran the sub-block that had been interrupted, from the start. The monitor polled every 5 minutes and stopped both instances at the end of each session.

## Files

- **[runs.csv](runs.csv):** the SHA-256 of every `requests.jsonl` journal, 196 sub-blocks × 5 configurations = 980 rows. Every hash matches both the sub-block's validation report and the S3 checkpoint copy.
- **[subblock_medians.csv](subblock_medians.csv):** **training cells only** (54 cells, 108 sub-blocks). For each sub-block and configuration: the cell, condition, levels, task, repetition, measured and `ok` counts, and the median latency taken from the validation report.
- **[s3_receipt.json](s3_receipt.json):** the campaign and reference prefixes in the artifact bucket, listed on 20 November 2026. All 12,442 objects match the local mirror by size and MD5.
- **[SHA256SUMS](SHA256SUMS):** every file in the campaign folder (5,921 files, excluding `supervisor.pid`), with paths relative to the campaign folder.
- **[descriptive_winners.json](descriptive_winners.json):** **training cells only.** For each cell, the configuration with the lowest mean of its two repetition medians, with the edge, cloud and fastest-distributed means in ms.

### Sealed (amendment A3)

The validation and evaluation rows were briefly included and then moved to the ignored folder `artifacts/aws-campaign-20261119-sealed/` ([amendment A3](../../docs/protocols/g2-v2-amendment-a3.md)).

| File | SHA-256 |
| --- | --- |
| `subblock_medians_validation_evaluation.csv` | `f3df477a1e3fa94473f3467b1d20fbda786fc821bd91c571212f29669fc22fdf` |
| `descriptive_winners_validation_evaluation.json` | `7d36d568082ba67b9772749f0db17663b2ed88ed017dd6a0827e76782ae6f385` |
| `subblock_medians_all98.csv` | `56388beb5f2e211218fb7bb902e9f7acb023027e3aaff74302dddb5eb5139b5e` |
| `descriptive_winners_all98.json` | `72e9041a5fb14df53c4ccfda0f2a3ce9c4826773ffb341564fd38b1de22fcea7` |

## First look: training cells (descriptive, not the oracle)

These numbers compare medians only. The section 9 oracle in the [derived checks](derived/README.md#section-9-oracle-training-cells) adds bootstrap CIs and ties, and it agrees on all 54 winners. The [derived checks](derived/README.md) cover row-level correctness (196/196 sub-blocks pass) and official VQA accuracy, which is identical for all five configurations. The training cells are T01–T12: 48 short cells and 6 long cells.

- **Repeatability:** the two repetitions' medians differ by 0.26% at the median (95th percentile 1.8%, maximum 3.5%).
- **Winners:** cloud is fastest in 48 of 54 cells, the edge in 6, and a distributed configuration in none.
- **Edge wins:** all six are at L0 / C2 (T01, T02, T09, T10, T11 and T12), by 11–381 ms.
- **Edge load:** L1 sends every one of them back to the cloud.
- **Load effects:**
  - C2 adds 597–632 ms to `cloud/raw_image` (median 613 ms);
  - L1 multiplies `edge/full` by 1.39–1.44 (median 1.42);
  - L1 does not change cloud latency (−6 to +9 ms).
- **Distributed configurations:**
  - int8 is the fastest distributed configuration in all 54 cells, yet it is 126–2,731 ms slower than the cloud (median 659 ms);
  - the synchronous configuration takes 1.04–1.78× as long as int8;
  - `prompt_first` is within 0.4% of synchronous.
- **Long task:** cloud in all 6 cells, at 800–1,090 ms against about 3,430 ms on the edge at L0 and about 5,070 ms at L1.
- **Always choosing one configuration:**
  - always cloud is wrong in 6 cells (mean regret 25 ms, maximum 381 ms);
  - always edge is wrong in 48 cells (mean 808 ms);
  - always int8 is wrong in all 54 (mean 803 ms).
