# G2 v2 protocol, amendment A4: G5, the live adaptive policy on dynamic traces

**Dated 20 November 2026.** Written after G4 passed and before any G5 measurement. Like [A3](g2-v2-amendment-a3.md), this is a separate file, so the pinned [protocol](g2-v2-protocol.md) and its hash stay unchanged.

**Two stages:**
- **Now:** the rules, traces, metrics, gate and pilot are fixed here.
- **After the pilot:** the edge-probe threshold is filled in by the rule in section 5, and the G5 grid, this file and `configs/g5/sensing.json` are pinned by hash before the campaign starts.

## 1. Purpose

G4 showed, offline, that the frozen r2 predictor plus the selector comes within 0.6 ms of the per-request oracle on unseen networks, **given the true load levels**. G5 runs the same predictor and selector **live**:
- the system senses the network, the edge load and the cloud load itself;
- the conditions change during the run;
- sensing, prediction and selection costs are part of the measured latency.

## 2. System under test

**Predictor.** `r2_component`, frozen by G3:
- calibration SHA-256 `16c20438…f05a`;
- copied unchanged to `configs/g5/calibration_r2_component.json`.

**Selector.** `scheduler.selector.select` over all five configurations, exactly as in G4.

**Sensing**, fresh before every decision (`scheduler/sensing.py`):

| Input | Source | Rule |
| --- | --- | --- |
| Cloud load (C0 or C2) | The server's own GPU telemetry, which the health response now carries (its cached background sample) | C2 iff GPU power ≥ **58 W** |
| Edge load (L0 or L1) | A short timing probe on the edge GPU: five 2048 × 2048 bf16 matrix multiplies, synchronised | L1 iff the probe takes at least the pilot threshold (section 5) |
| Uplink goodput, RTT | The latest periodic active probe: the pre-segment iperf3 and echo measurement, taken by the runtime before each segment | used as measured |

**Where the 58 W comes from.** It is fixed on the training partition. It is the midpoint between the 95th percentile of cloud GPU power at C0 (50.6 W) and the 5th percentile at C2 (65.2 W), measured on the previous response's server state.

**Why the edge needs a probe.** The L4-6Q vGPU's own utilisation counter does not separate L0 from L1 on training: the median single reading is 0% at L0 and 6% at L1. The memory difference the load generator causes is an artifact of its own allocation, so it is not used either.

**Pre-execution inputs:**
- crop count from the image size (pinned tiling rule);
- image bytes;
- task.

They come from `configs/g5/task_features-short.json`, a copy of the committed feature bundle (SHA-256 `e7a18def…c81d`).

**No hysteresis, no minimum dwell (roadmap 5.2).** Switching configuration costs nothing measurable on this hardware. In the G2 v2+ training data the interleaved order randomises the preceding configuration, and the paired latency of the same request-cell after the same versus a different configuration differs by:
- −0.12% to +0.04% at the median for the cloud and the three split configurations;
- +0.9% for `edge/full`, where repeating the configuration is slower, not faster.

Stage preparation on the resident edge takes 0 ms. Decisions are therefore per request: a hysteresis band would only block legitimate per-request choices, such as sending small images to the edge.

**Live failure handling.** If the selector finds no feasible configuration, the request fails and is recorded; it is never silently routed. If the cloud power reading is missing, the cloud is taken as C0 and the source is recorded.

## 3. Execution

**The adaptive placement.** The adaptive policy is one more interleaved placement (`adaptive`).
- Every measured request of a segment runs the five fixed configurations and the adaptive policy back to back, in a seeded random order.
- The adaptive policy's observed latency is timed like the others. It includes sensing, prediction, selection and the chosen configuration's inference.
- Each placement has one excluded warmup per segment.

**Segments and traces.**
- A **segment** is one sub-block in one state: a network condition, edge load and cloud load. It uses all 30 evaluation request IDs of the short task, with one repetition.
- A **trace** is an ordered sequence of five segments; the state changes at every segment boundary.

The grid is `configs/profiles/aws_g5_traces.json`. Segment states are given as condition and load, edge / cloud:

| Trace | What changes | Segments, in order |
| --- | --- | --- |
| `net` | the network, under heavy cloud load | E02 L0/C2 → E01 L0/C2 → E04 L0/C2 → E05 L0/C2 → E02 L0/C2 |
| `cloud` | cloud-load episodes | E01 L0/C0 → L0/C2 → L0/C0 → L0/C2 → L0/C0 |
| `edge` | edge-load episodes, under heavy cloud load | E05 L0/C2 → L1/C2 → L0/C2 → L1/C2 → L0/C2 |
| `mixed` | both loads | E03 L0/C0 → L0/C2 → L1/C2 → L1/C0 → L0/C2 |

The evaluation conditions are E01 (20 Mb/s, 50 ms), E02 (70, 20), E03 (120, 100), E04 (175, 50) and E05 (50, 150), all inside the training ranges.

- **Repetitions and order:** four repetitions. Within each repetition, the four traces run in an order shuffled with seed **8123**; segments keep their order.
- **Size:** 80 segments; 14,400 measured attempts (80 × 30 × 6) and 480 warmups.

**Validity.** Every segment passes the section 8 gates as adapted on AWS:
- probes within tolerance and zero netem drops;
- references verified at preflight;
- at most 1% failed attempts, for each placement including the adaptive one;
- edge memory at L0 within the limit;
- the competitor alive at L1;
- the cloud-load gate (A2a rule 7).

The cloud-load gate ignores adaptive requests that ran on the edge, because they carry no server state. An invalid segment is rerun as a whole.

**Time and cost.** The G2 v2+ campaign ran at about 1.9 s per attempt including probes, so the campaign takes about 9 hours. It runs in two sessions of at most 6 hours, with the existing monitor and session scripts. At about $1.8 an hour for the pair, the campaign costs about $16, and about $25 including the image build, references and pilot.

## 4. Metrics and analysis

The analysis is `scripts/campaign/analyze_g5.py`. For every measured request it computes:
- **Oracle:** the fastest of that request's five fixed attempts (hindsight).
- **Adaptive regret:** the adaptive latency minus the oracle latency.
- **Comparison with each fixed configuration:** the adaptive latency minus that configuration's latency.
- **Label counterfactual:** the frozen selector rerun offline with the true load labels and the same network probe, scored by the observed latency of the fixed attempt it would have chosen. It separates sensing error from prediction error.

**Reported:**
- **Latency:** mean, p50 and p95 for the adaptive policy, every fixed configuration and the oracle.
- **SLA success**, with the SLA declared now as **1,500 ms**, just above the unloaded edge's short-task latency (training medians 1.0–1.06 s) and below its L1 latency.
- **Regret:** median and P90 relative regret, mean regret with a cluster-bootstrap 95% CI (request IDs; seed 1707, 2,000 resamples), and top-1 agreement.
- **Sensing accuracy:** edge and cloud, against the true segment labels.
- **Overhead:** median and p95 of sensing plus selection.
- **Switching:** configuration changes between consecutive adaptive requests within a segment, and the choice distribution.
- **Answers:** whether each adaptive answer equals the chosen configuration's answer.
- **Per-segment summaries.**
- **Quality and transfer:** official VQA accuracy of the adaptive answers (parity at most 2 pp against the fixed configurations), bytes per request, and failure rates.

## 5. Pilot and go/no-go

The pilot runs after the new image is built and the references are rebuilt. The `src` change since `d5e116c` invalidates the old reference bundles' source hash, so they must be rebuilt.

**1. Edge probe calibration** (`scripts/campaign/calibrate_edge_probe.py`, on the edge, inside the image):
- 60 probes at L0 and 60 under the calibrated L1 competitor, 0.25 s apart.
- Threshold = √(median L0 × median L1). **Go** iff p95(L0) < threshold < p05(L1).
- On a go, the script writes `configs/g5/sensing.json` from the template with the threshold.

**2. Pilot trace:** one repetition of a two-segment trace, E01 L0/C0 → E01 L1/C2, with the 30 evaluation IDs. **Go** iff:
- both segments are valid;
- cloud sensing accuracy is at least 99% and edge sensing accuracy at least 95% on the adaptive requests;
- the median adaptive overhead is at most 50 ms;
- every adaptive answer equals the chosen configuration's answer.

**3. On a no-go:** a dated amendment before any campaign data is collected.

Pilot data is not campaign data and is not scored for G5.

## 6. G5 gate

G5 passes if all of these hold over the whole campaign:
1. **Faster than the best static configuration.** The best static configuration was chosen on validation (`cloud/raw_image`). The paired mean of adaptive minus always-cloud latency is negative, with the upper end of its cluster-bootstrap 95% CI below 0.
2. **Close to the oracle:** the median relative regret against the per-request oracle is at most 10%.
3. **Reliable:** every segment is valid, and no adaptive request failed.
4. **Quality:** the adaptive answers' official VQA accuracy is within 2 pp of every fixed configuration's.

Every metric in section 4 is reported, pass or fail. There is no refitting or retuning after the campaign starts.

## 7. Disclosure

**How the traces were chosen.** The traces use evaluation conditions and request IDs whose per-cell winners G4 has already reported. They were chosen so that the oracle's choice changes along each trace; that is the point of a dynamic test.

**Why G5 is still a fresh test.** It measures new live latencies with a predictor, selector and sensing rules that were fixed before any G5 data existed, and nothing is fitted on G5 data.

**One threshold comes from the pilot:** the edge-probe threshold. The rule that sets it is fixed here.
