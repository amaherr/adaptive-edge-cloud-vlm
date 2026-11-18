# G2 v2+ pilot on AWS (18 November 2026)

The go/no-go pilot of amendment A2b rule 9. The data is **discarded**: it used training IDs only and is not campaign data.

- **Hosts:** edge `g6f.2xlarge` (L4-6Q), cloud `g6.xlarge` (L4).
- **Image:** `sha256:75d2f7349425…` (commit `b360d6b`), the same digest on both hosts.
- **Grid:** [aws_pilot_grid_v2plus.json](../../configs/profiles/aws_pilot_grid_v2plus.json), 16 sub-blocks with one repetition.
- **Raw data:** S3 `cloud-v1/edge/g2v2plus-pilot/`, and the ignored `artifacts/g2v2plus-aws-pilot/`.
- **Files:** [runs.csv](runs.csv) lists the SHA-256 of every journal, and [oracle_cells.csv](oracle_cells.csv) is the derived oracle table.

## Go/no-go: go

| Check (A2b rule 9) | Result |
| --- | --- |
| 1. Every sub-block valid | 16/16 on their first attempt, and 750/750 attempts ok. Probes, drops, L0 memory, the cloud-load gate at C0 and C2, and edge load at L1 all passed. |
| 2. C2 changes the winner at L0 | Yes at P02 and P03: `cloud/raw_image` → `edge/full`. |
| 3. L1 changes the winner at C2 | Yes at P02 and P03: `edge/full` → `cloud/raw_image`. |
| 4. Rows match their references | 0 reference mismatches and 0 cross-GPU disagreements in all 16 profiles. This covers `prompt_first` against the synchronous control and int8 against the int8 control. See the note below. |
| 5. Cloud state on cloud rows | Present on every cloud and distributed row. |
| 6. Run length | Overhead was 56–69 s per sub-block (mean 62 s), and request time was 0.99× the E5 estimate. The campaign is about 22 h, roughly 7.2 h per session over three sessions. |

Preflight also passed the reference verification for both tasks and the GPU correctness smoke.

## Measured medians (ms)

These are the short-task cells, over 10 training IDs. "Distributed" is the fastest of the three distributed configurations.

| Condition | L0 / C0 | L0 / C2 | L1 / C0 | L1 / C2 |
| --- | --- | --- | --- | --- |
| P01 100 Mb/s / 5 ms | **cloud 291**, edge 998 | **cloud 888**, edge 1,002 | | |
| P02 100 / 100 | **cloud 577**, edge 1,002 | **edge 997**, cloud 1,170 | **cloud 575**, edge 1,405 | **cloud 1,173**, edge 1,408 |
| P03 150 / 150 | **cloud 725**, edge 1,000 | **edge 1,000**, cloud 1,319 | **cloud 726**, edge 1,402 | **cloud 1,322**, edge 1,410 |
| P04 35 / 20 | **cloud 347**, edge 1,000 | **cloud 942**, edge 1,000 | | |
| P05 200 / 5 | **cloud 290**, edge 1,002 | **cloud 887**, edge 1,004 | | |

The long task (5 IDs) went to the cloud in both cells: at P01 / L0, 822 ms against 3,430 ms on the edge; at P05 / L1, 807 against 5,080. The fastest distributed configuration was int8 in all 16 cells. A distributed configuration won no cell, as the E5 estimate predicted.

**Ties.** The oracle marks a cell a tie when the bootstrap CIs of the top two overlap (section 9). With 10 IDs those CIs are wide, so several clear median gaps still count as ties: P02-L0 (425 ms) and P02-L0-C2 (173 ms) are marked ties, while P03-L0-C2 (319 ms) is not. The campaign has 60/30/30 IDs × 2 repetitions per cell.

**Note on the profile correctness flag.** `heterogeneous_correctness.passed` is false in every per-sub-block profile, for a structural reason:
- `evaluate_rows` checks coverage, quality equality and repeatability against all 120 manifest IDs;
- it also requires two repetitions within the evaluated rows;
- a pilot or campaign sub-block measures a subset of IDs once.

So unmeasured IDs count as quality disagreements and as non-repeatable empty groups. The row-level checks, reference mismatches and cross-GPU disagreements, are the ones that compare outputs, and both are empty. Sub-block validity does not depend on this flag. The flag's scope should be fixed in the analysis code before the derived tables are used (G3).
