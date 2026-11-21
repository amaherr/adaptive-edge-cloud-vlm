# G5 on AWS: the live adaptive policy on dynamic traces (20–21 November 2026)

**G5 passed.** The live adaptive system ran on the AWS pair. It sensed the network, edge load and cloud load itself, predicted latency with the frozen r2 model and chose a configuration per request, while the conditions changed from segment to segment.
- **It is 118 ms (11.5%) faster on average than the best static configuration (always cloud),** with a paired 95% CI of 95–143 ms.
- **It is within 22 ms of the per-request hindsight oracle** (median relative regret 2.2%).
- **Answer quality is unchanged,** and no adaptive request failed.

The rules come from [amendment A4](../../docs/protocols/g2-v2-amendment-a4.md), with [A4a](../../docs/protocols/g2-v2-amendment-a4a.md) (edge-probe shape) and [A4b](../../docs/protocols/g2-v2-amendment-a4b.md) (background cloud sensing). Every input was pinned before the campaign in [freeze/g5_freeze.json](freeze/g5_freeze.json).

## Setup

- **Hosts:** edge `g6f.2xlarge` (L4-6Q vGPU), cloud `g6.xlarge` (L4), us-east-1d.
- **Image:** `sha256:2cd3bd95…` (commit `6cf005f`) on both hosts. Its reference bundles were rebuilt for it and verified: 120 requests per GPU, 0 mismatches.
- **System under test:**
  - the frozen `r2_component` predictor (G3);
  - the five-configuration selector (G4);
  - live sensing:
    - cloud load from the server's GPU power in a background sampler (C2 at 58 W or more);
    - edge load from a 128 × 128 × 80 synchronised timing probe (L1 at 20.65 ms or more);
    - the network from each segment's periodic probe.
- **Traces:** four traces of five segments over the evaluation conditions E01–E05, four repetitions, 80 segments in all. Each segment ran 30 evaluation questions, with the five fixed configurations and the adaptive policy interleaved per request.
  - `net`: the network changes under cloud load;
  - `cloud`: cloud-load episodes;
  - `edge`: edge-load episodes under cloud load;
  - `mixed`: both loads change.
- **Sessions:**
  - session 1: 20 November, 19:20 UTC – 21 November, 00:52 UTC, ending with 56 segments valid;
  - session 2: 21 November, 01:07–03:26 UTC, completing the campaign.
  - The segment interrupted at the end of session 1 reran from the start.

### Pilot history

These runs are not campaign data.

| Step | Result |
| --- | --- |
| A4 edge-probe calibration (2048 × 5 burst) | **No-go:** it couldn't separate L0 from L1 on the time-sliced vGPU → amendment A4a |
| A4a sweep, then calibration (128 × 80, synchronise after each) | **Go:** L0 p95 16.86 ms < threshold 20.65 ms < L1 p05 25.38 ms |
| Pilot 1 | **No-go on overhead:** a health probe per decision cost 126 ms → amendment A4b |
| Pilot 2 (background cloud sensing) | **Go:** valid 2 of 2, cloud sensing 100%, edge sensing 98.3%, median overhead 23.8 ms, answers 60 of 60 |

Files: [freeze/](freeze/), [pilot/](pilot/g5_summary.json), [pilot2/](pilot2/g5_summary.json).

## Validity

| Check | Result |
| --- | --- |
| Segments valid (A4 section 3 gates) | **80 of 80**, all on the first attempt, 0 restarts |
| Measured attempts | **14,400 of 14,400** `ok`: 2,400 for each fixed configuration and 2,400 adaptive |
| Warmups | 480 |
| Adaptive requests failed | **0** |
| Every adaptive answer equals its chosen configuration's | **2,400 of 2,400** |

## G5 gate (A4 section 6)

| Criterion | Result | |
| --- | --- | --- |
| Faster than the best static configuration (always cloud) | adaptive − cloud = **−118.1 ms** (95% CI −142.8 to −94.8) | **pass** |
| Median relative regret against the per-request oracle ≤ 10% | **2.2%** (P90 4.4%) | **pass** |
| Every segment valid, no failed adaptive request | 80 of 80; 0 failed | **pass** |
| Quality within 2 pp of every fixed configuration | **0 pp**: all 30 questions got one identical answer across every policy and segment, equal to the answers scored officially in G4 (70.67%, lowercased) | **pass** |

## Results

Over 2,400 measured requests:

| Policy | Mean | p50 | p95 | ≤ 1.5 s SLA | Adaptive − policy (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| **Adaptive** | **908.3 ms** | 1,009.9 | 1,412.4 | **98.5%** | |
| Oracle (hindsight) | 886.3 | 995.3 | 1,373.2 | 98.7% | |
| `cloud/raw_image` (best static) | 1,026.4 | 1,096.8 | 1,527.5 | 91.7% | −118.1 (−142.8 to −94.8) |
| `edge/full` | 1,036.3 | 1,003.8 | 1,514.9 | 94.0% | −128.0 (−145.8 to −109.6) |
| int8 split | 1,573.8 | 1,587.0 | 2,138.9 | 45.6% | −665.5 |
| bf16 split | 1,877.4 | 1,926.0 | 2,377.8 | 24.0% | −969.1 |
| `prompt_first` split | 1,888.3 | 1,975.8 | 2,376.2 | 24.0% | −980.0 |

- **Close to the oracle.** The adaptive policy picks the fastest configuration in 95.7% of requests. Its mean regret is 22.0 ms (CI 20.8–23.3).
- **Sensing:**
  - cloud load is sensed correctly in 100% of requests, edge load in 97.8%;
  - the median overhead (sensing plus selection) is 15.9 ms, p95 36.6 ms.
- **Most of the regret is overhead.** The label counterfactual reruns the frozen selector offline with the true load levels on the same requests; it would lose only **1.0 ms** to the oracle, and it agrees with the live choice in 98.8% of requests. Nearly all of the 22 ms comes from the sensing overhead and a few edge misreadings, not from prediction.
- **Choices:** the edge 1,330 times, the cloud 1,070 times, never a split. There were 317 configuration changes within segments, because the decision is per request (small images go to the edge).

**By trace segment** (mean ms over the four repetitions):

| Trace | State | Adaptive | Always cloud | Oracle | Adaptive's choices |
| --- | --- | ---: | ---: | ---: | --- |
| `cloud` | E01 L0/C0 (s0, s2, s4) | 475–477 | 460 | 460 | all cloud |
| `cloud` | E01 L0/C2 (s1, s3) | 970–976 | 1,058–1,059 | 952–953 | 93–97% edge |
| `edge` | E05 L0/C2 (s0, s2, s4) | 975–977 | 1,379–1,383 | 956 | 98–99% edge |
| `edge` | E05 L1/C2 (s1, s3) | 1,335–1,342 | 1,382–1,383 | 1,303–1,308 | 56% cloud, 44% edge |
| `mixed` | E03 L0/C0 → L1/C2 → L1/C0 | 609 / 1,219 / 633 | 595 / 1,202 / 597 | 595 / 1,184 / 597 | cloud; 79% cloud; cloud |
| `mixed` | E03 L0/C2 (s1, s4) | 976–978 | 1,191–1,193 | 956 | 97–98% edge |
| `net` | E02 L0/C2 (s0, s4) | 922–923 | 936–937 | 907–908 | 80% cloud |
| `net` | E01 / E04 / E05 L0/C2 | 978 / 970 / 978 | 1,066 / 1,029 / 1,378 | 954 / 948 / 957 | mostly edge |

- **With the cloud unloaded (C0)** the policy correctly keeps the cloud. It pays its sensing overhead, about 16 ms (3%), behind always-cloud.
- **When the cloud is loaded** it moves to the edge and gains up to about 400 ms per request.
- **Under both loads** it splits requests between the two by image size.
- **The gain is concentrated where adapting matters,** as the baseline report predicted.

## Files

| Path | Content |
| --- | --- |
| [campaign/g5_summary.json](campaign/g5_summary.json) | The full analysis (`scripts/campaign/analyze_g5.py`): latency per policy, paired differences, regret, sensing, overhead, label counterfactual, per segment |
| [campaign/g5_requests.jsonl](campaign/g5_requests.jsonl) | One record per measured request: adaptive choice, sensed state, the five fixed latencies, answers |
| [campaign/SHA256SUMS](campaign/SHA256SUMS) | Every file of the mirrored campaign checkpoint (2,721 files) |
| [freeze/](freeze/) | Freeze record, calibrated sensing file, probe calibration and sweep |

**Raw data (git-ignored).** The data is in `artifacts/g5-aws/campaign/`, a mirror of `s3://…/cloud-v1/edge/g5-campaign/`. All 2,722 objects (77 MB) match S3 by key and size.

**The edge data volume** holds the final supervisor status (`completed`, 80 of 80 valid, 14,400 of 14,400) and the preflights. It was copied to S3 and archived in roadmap phase 8 ([archive record](../archive-20261121/README.md)).

## Limits

- **Network sensing:** the network input comes from the periodic probe taken at the start of each segment. Within a segment the link is constant, so the traces test step changes between segments, not drift within one.
- **Task:** only the short-answer task ran. The long task under cloud load has no training data.
- **Background polling:** the cloud sampler adds a health exchange every second on the shaped link, of a few hundred bytes (disclosed in A4b).
- **One pair, one campaign:** the results describe this hardware pair and this campaign.
