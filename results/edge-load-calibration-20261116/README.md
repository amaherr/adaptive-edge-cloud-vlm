# Edge-load calibration and resource-state check (15–16 November 2026)

**Chosen levels ([levels_v1.json](../../configs/edge_load/levels_v1.json)):** both are compute contention on a 512 MiB working set.

| Level | Working set | Busy / idle per cycle | Achieved duty | Edge/full median (3 runs) | p90 |
| --- | ---: | --- | ---: | --- | --- |
| L0 | none | none | none | 1,272 ms | 1,315 ms |
| **L1** | 512 MiB | 10 ms / 10 ms | 0.60 | **1,842 / 1,863 / 1,868 ms** | 1,900–1,934 ms |
| **L2** | 512 MiB | 20 ms / 0 ms | 0.99 | **2,805 / 2,805 / 2,807 ms** | 2,878–2,889 ms |

These supersede the draft levels in the G2 v2 protocol, which were 512 MiB for L1 and 1 GiB for L2. Section 6 now records them.

## Setup

- Edge laptop (RTX 3050, 4 GiB); edge placement only; swap residency.
- 20 prompt-calibration questions with prompt P1, not G2 v2 data; two warmups and one measured repetition per run.
- Every run used `interleaved-benchmark --resource-state --edge-load-config … --edge-load-level …`.
- The generator is `python -m adaptive_vlm.benchmark.edge_load_worker`. It runs in-place additions over its buffer for `busy_ms`, synchronizes, then sleeps for `idle_ms`. The controller starts it after the model has loaded and checks its heartbeat before every request. It stops the generator at the end and records its final heartbeat in `sessions.jsonl`.

All runs are listed in [runs.csv](runs.csv), with the SHA-256 of each `requests.jsonl`. The raw run directories are kept in the ignored `artifacts/edge-load-calibration/` and `artifacts/resource-state-check/` folders.

## Sweep

| Working set | Busy / idle (ms) | Achieved duty | Median | p90 | Max Windows dedicated |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 MiB | 5 / 15 | 0.45 | 1,701 ms | 1,750 ms | 3,862 MiB |
| 512 MiB | 10 / 10 | 0.60 | 1,832 ms | 1,880 ms | 3,862 MiB |
| 512 MiB | 15 / 5 | 0.77 | 2,129 ms | 2,214 ms | 3,862 MiB |
| 512 MiB | 18 / 2 | 0.93 | 2,592 ms | 2,660 ms | 3,862 MiB |
| 512 MiB | 20 / 0 | 0.99 | 2,840 ms | 2,932 ms | 3,862 MiB |
| 1 GiB | 5 / 15 | 0.77 | 2,963 ms | 4,655 ms | 3,902 MiB |
| 1 GiB | 10 / 10 | 0.82 | 3,032 ms | 4,751 ms | 3,902 MiB |
| 1 GiB | 15 / 5 | 0.92 | 2,707 ms | 4,189 ms | 3,912 MiB |

Achieved duty is higher than the nominal busy share because each kernel runs longer while the GPU is shared.

**Why 512 MiB for both levels.** With a 512 MiB working set, the edge latency rises smoothly with the duty cycle and stays tight: p90 is within 4% of the median. With 1 GiB, the edge sits within about 50 MiB of the 3,960 MiB dedicated limit. Latency is then about 2.7–3.0 s regardless of duty, with a long tail (p90 4.2–4.8 s). That is memory pressure rather than a controlled load, so it would add an uncontrolled factor to the baseline.

**Why L1 = 1.85 s and L2 = 2.8 s.** The short-answer prompt cut edge/full from about 1.5 s to 1.27 s, and it should shorten cloud decoding the same way. Cloud under P1 has not been measured yet (that needs the cloud PC), but it is expected below G2's 2.3–2.9 s.
- L1 puts the edge near the likely cloud/distributed range, so the winner can go either way.
- L2 makes the edge clearly slower than cloud.

The on-site pilot checks this (see the protocol, section 6).

Under L1 and L2, Windows dedicated usage peaks at 3,862 MiB, and shared usage sits about 76 MiB above the pinned-weights baseline. The edge is close to its memory limit but its latency stays stable. The protocol's no-spill gate applies only at L0.

## Resource-state check: a deliberate spill

Six questions, edge only, resource state on:

| Edge residency | Median | Windows dedicated | Windows shared | NVML used |
| --- | ---: | ---: | ---: | ---: |
| resident (spills) | 12,363 ms | 3,960 MiB (the maximum) | 668 MiB | 4,089 MiB |
| swap | 1,292 ms | 3,271 MiB | 4,374 MiB | 3,393 MiB |

The spill is clearly visible, but in *dedicated* usage reaching the adapter's maximum, not in shared usage. Under swap residency, Windows counts the pinned host master copies of the weights (about 4.4 GB) as shared GPU memory, so shared usage is high even though nothing spills. The protocol's L0 gate therefore checks dedicated usage. Other things visible in the snapshots:
- SM clocks vary from request to request (about 1,450–2,000 MHz within one run).
- Power is 38–44 W under load.
- Peak temperature is 58–62 °C.
