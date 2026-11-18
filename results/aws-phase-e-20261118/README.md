# Phase E recalibration on AWS (18 November 2026)

These are calibration runs only, not campaign data. All requests used P1 prompt-calibration questions on the AWS pair: edge `g6f.2xlarge` (L4-6Q vGPU, resident) and cloud `g6.xlarge` (L4). Image `sha256:2c1bbf861a4d…` was built from commit `61c1748`. Raw run directories are in the ignored `artifacts/phase-e/` folder and in S3 under `cloud-v1/edge/phase-e/`. [runs.csv](runs.csv) lists the SHA-256 of every `requests.jsonl`, and [summary.jsonl](summary.jsonl) has one summary line per run.

## E3: cloud load C1 (rule 6, A2a-4)

Each level ran 20 `cloud/raw_image` requests, unshaped, with the edge idle, on questions 120–139. The competitor was the 512 MiB worker in the inference container.

| Level | Busy / idle (ms) | Achieved duty | Median server time | × C0 |
| --- | --- | ---: | ---: | ---: |
| C0 | none | none | 263.3 ms | 1.00 |
| C10 | 2 / 18 | 0.23 | 314.0 ms | 1.19 |
| C15 | 3 / 17 | 0.25 | 316.0 ms | 1.20 |
| C25 | 5 / 15 | 0.37 | 324.6 ms | 1.23 |
| **C50** | **10 / 10** | **0.56** | **385.3 ms** | **1.46** |
| C75 | 15 / 5 | 0.78 | 496.9 ms | 1.89 |
| C100 | 20 / 0 | 1.00 | 862.0 ms | 3.27 |

**C1 = C50.** It is the smallest duty within 1.4–1.6×. It is now in `configs/cloud_load/levels_v1.json`. The worker was alive in all 20 responses at every level.

## Reference at 35 Mb/s / 20 ms (rule 11's anchor)

All five configurations ran at L0 and C0 on questions 140–159. Classified shaping measured 34.7 / 34.8 Mb/s and 20.3 ms of echo RTT, with no failures.

| Configuration | Median | p90 | Request bytes |
| --- | ---: | ---: | ---: |
| cloud/raw_image | **359 ms** | 382 ms | 141 KB |
| distributed/projected/int8 | 1,053 ms | 1,083 ms | 1.49 MB |
| edge/full | 1,019 ms | 1,093 ms | none |
| distributed/projected/synchronous | 1,391 ms | 1,432 ms | 2.99 MB |
| distributed/projected/prompt_first | 1,394 ms | 1,423 ms | 2.99 MB |

## E2: edge load sweep (in-process, A2a-3)

Each setting ran 20 edge/full requests on questions 120–139, with 2 warmups. The working set was 256 MiB. All 180 requests succeeded with no OOM. Peak NVML usage was 6,058 MiB, against 5,782 MiB at L0.

| Setting | Kernel | Busy / idle (ms) | Achieved duty | Median | p90 |
| --- | ---: | --- | ---: | ---: | ---: |
| L0 | none | none | none | 1,020 ms | 1,098 ms |
| E-I40 | 8 MiB | 1 / 40 | 0.26 | 1,212 ms | 1,313 ms |
| E-I20 | 8 MiB | 1 / 20 | 0.46 | 1,311 ms | 1,413 ms |
| E-I10 | 8 MiB | 1 / 10 | 0.64 | 1,422 ms | 1,564 ms |
| E-I5 | 8 MiB | 1 / 5 | 0.73 | 1,740 ms | 1,910 ms |
| E-I2 | 8 MiB | 1 / 2 | 0.89 | 1,734 ms | 1,890 ms |
| E-K4 | 4 MiB | 20 / 0 | 1.00 | 1,852 ms | 2,102 ms |
| E-K8 | 8 MiB | 20 / 0 | 1.00 | 2,365 ms | 2,683 ms |
| E-K16 | 16 MiB | 20 / 0 | 1.00 | 3,006 ms | 3,590 ms |

**Rule 11's targets cannot be met on AWS.** L1 asks for edge ≈ the cloud median at 35/20 (359 ms ± 10%), and L2 for 1.5× that. Even the unloaded edge takes 1,019 ms, 2.8× the cloud. Load only slows the edge further. No L1 or L2 is chosen yet (see the open decision in the roadmap).

## E4: int8 quality (rule 4)

[int8_quality.json](int8_quality.json) records the full results.
- **Short task (P1, 200 questions):** 199 of 200 answers are token-identical; the exception is "6" against "5". The identical-answer share is 99.5%, above the 95% minimum. With only one answer changed, official accuracy can move by at most 0.5 points under either scoring, so the 1-point rule holds by construction. The official evaluator scores are still to be recorded.
- **Long task (E1, 200 images):** 132 of 200 token sequences are identical. Mean position agreement is 0.83, and the mean common prefix is 26 of 32 tokens. Every output reaches the 32-token cap. This is reported, not gated.

int8 is admitted (rule 4) once the official scores are recorded.
