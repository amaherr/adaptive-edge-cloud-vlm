# Phase E recalibration on AWS (18 November 2026)

These are calibration runs only, not campaign data. Requests used prompt-calibration questions (P1 short task; E1 long task in E4 and the E5 profile) on the AWS pair: edge `g6f.2xlarge` (L4-6Q vGPU, resident) and cloud `g6.xlarge` (L4). Image `sha256:2c1bbf861a4d…` was built from commit `61c1748`. Raw run directories are in the ignored `artifacts/phase-e/` folder and in S3 under `cloud-v1/edge/phase-e/`. [runs.csv](runs.csv) lists the SHA-256 of every `requests.jsonl`, and [summary.jsonl](summary.jsonl) has one summary line per run.

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

**Rule 11's targets cannot be met on AWS.** L1 asks for edge ≈ the cloud median at 35/20 (359 ms ± 10%), and L2 for 1.5× that. Even the unloaded edge takes 1,019 ms, 2.8× the cloud. Load only slows the edge further. Amendment A2a-5 replaced these targets before L1 was chosen (next section).

## A2a-5: L1 and C2

**L1 = E-I10** (1 ms busy / 10 ms idle, 256 MiB, 8 MiB kernels), now in `configs/edge_load/aws_levels_v1.json`. In the sweep it gave 1,422 ms, 1.39× L0, the closest setting to 1.4× and inside 1.26–1.54×. Three reproduction runs (questions 120–139) measured 1,409.9, 1,411.5 and 1,408.6 ms, a 0.2% spread and 0.8% below the sweep. In the E5 profile every configuration ran at L1 on both tasks with no OOM (600 requests).

**C2 = C100** (20 ms busy / 0 idle, 512 MiB), now in `configs/cloud_load/levels_v1.json`. `edge/full` and `cloud/raw_image` were interleaved at 35/20, L0, on questions 140–159:

| Candidate | cloud/raw_image | edge/full | Cloud vs edge | Within ±10% |
| --- | ---: | ---: | ---: | --- |
| C75 | 602 ms | 1,004 ms | −40% | no |
| **C100** | **971 ms** | **1,005 ms** | **−3.4%** | **yes** |

## E5: shaped references and network fit

All five configurations ran at L0 / C0 on questions 140–159 at two more conditions. Shaping held within 1% of rate and 0.4 ms of RTT, with no failures.

| Configuration | T06 150 Mb/s / 20 ms | 35 Mb/s / 20 ms | T12 25 Mb/s / 150 ms |
| --- | ---: | ---: | ---: |
| cloud/raw_image | 348 ms | 359 ms | 766 ms |
| distributed/projected/int8 | 816 ms | 1,053 ms | 1,775 ms |
| distributed/projected/synchronous | 891 ms | 1,391 ms | 2,247 ms |
| distributed/projected/prompt_first | 898 ms | 1,394 ms | 2,247 ms |
| edge/full | 1,001 ms | 1,019 ms | 1,000 ms |

[winners.json](winners.json) fits each configuration's overhead over its unshaped median as `beta × transfer time + gamma × added RTT` (`scripts/campaign/estimate_aws_winners.py`). Every residual is within 7.3 ms. `cloud/raw_image` pays about 3.1 RTTs per request; the distributed paths pay about 4.6.

## E5: unshaped profile

Median end-to-end time over 20 requests per configuration, questions 160–179, all requests ok.

| Task / edge / cloud | edge/full | cloud/raw_image | distributed sync | prompt_first | int8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| p1 / L0 / C0 | 1,003 | **279** | 656 | 656 | 658 |
| p1 / L0 / C1 | 1,002 | **407** | 708 | 714 | 709 |
| p1 / L0 / C2 | 999 | **901** | 957 | 959 | 961 |
| p1 / L1 / C0 | 1,410 | **279** | 824 | 825 | 821 |
| p1 / L1 / C1 | 1,418 | **417** | 882 | 880 | 881 |
| p1 / L1 / C2 | 1,403 | **905** | 1,126 | 1,131 | 1,128 |
| e1 / L0 / C0 | 3,434 | **786** | 1,152 | 1,149 | 1,151 |
| e1 / L0 / C1 | 3,442 | **1,183** | 1,466 | 1,462 | 1,465 |
| e1 / L0 / C2 | 3,425 | **1,953** | 2,005 | 2,007 | 2,005 |
| e1 / L1 / C0 | 5,057 | **781** | 1,317 | 1,319 | 1,318 |
| e1 / L1 / C1 | 5,056 | **1,178** | 1,628 | 1,625 | 1,631 |
| e1 / L1 / C2 | 5,047 | **1,875** | 2,110 | 2,111 | 2,114 |

L1 slows `edge/full` by 1.40× (p1) and 1.47× (e1), and the distributed paths by 105–175 ms, since they encode the image on the edge. C2 slows the cloud server 3.3× (p1) and 2.5× (e1).

## Winner estimate over the 22 section 5 conditions

| Task / edge / cloud | Predicted winners |
| --- | --- |
| p1 / L0 / C2 | edge/full 13, cloud/raw_image 9 |
| every other combination (11) | cloud/raw_image 22 |

A distributed configuration never wins, so int8 never changes the winner. Within p1 / L0 / C2 the winner changes in every split: edge/cloud is 6/6 in training, 3/2 in validation and 4/1 in evaluation. The edge wins at low rate or high RTT. Eight of the 22 margins are under 35 ms, small against the 40–130 ms gap between a configuration's median and p90. Adding L1 to C2 turns all 13 edge wins back to the cloud. Four of those (T11, T12, V05, E05, all 150 ms) stay within 32 ms. The long task goes to the cloud everywhere. Its smallest margin is 109 ms, over int8 at C2 / T08, and `edge/full` trails by about 1 s or more.

## E4: int8 quality (rule 4)

[int8_quality.json](int8_quality.json) records the full results.
- **Short task (P1, 200 questions):** 199 of 200 answers are token-identical; the exception is "6" against "5". The identical-answer share is 99.5%. Official VQA accuracy, lossless against int8, is 56.2% vs 56.7% unchanged and 79.2% vs 79.7% lowercased. **int8 is admitted** ([decision.json](../int8-quality-20261118/decision.json), scored with the pinned evaluator under PyPy 2.7.18 / 7.3.15).
- **Long task (E1, 200 images):** 132 of 200 token sequences are identical. Mean position agreement is 0.83, and the mean common prefix is 26 of 32 tokens. Every output reaches the 32-token cap. This is reported, not gated.

