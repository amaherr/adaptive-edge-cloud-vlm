# Adaptive runtime (G5)

The live adaptive policy evaluated in G5: before each request it senses the system state, predicts the latency of every configuration with the frozen `r2_component` model, picks the fastest, and runs it. This page describes the code. The rules it implements are fixed in [amendment A4](../protocols/g2-v2-amendment-a4.md), with [A4a](../protocols/g2-v2-amendment-a4a.md) (edge-probe shape) and [A4b](../protocols/g2-v2-amendment-a4b.md) (background cloud sensing); the results are in the [G5 record](../../results/g5-aws-20261120/README.md).

It replaces the prototype path of [adaptive execution](adaptive-execution.md) and the [live monitor](live-monitor-and-trace-replay.md) for the AWS campaigns. Those still describe the local `adaptive-benchmark` command, which uses the schema-1 predictor and the three original configurations.

## The five configurations

`scheduler/configuration.py` defines `ALL_CONFIGURATIONS`, in this stable order (it breaks exact ties):

| ID | Edge | Cloud | Crosses the link |
| --- | --- | --- | --- |
| `edge/full` | Whole model | — | Nothing |
| `cloud/raw_image` | Reads the file | Whole model | Encoded image and prompt |
| `distributed/projected/synchronous` | Vision encoder and projector | Language model | Projected BF16 tensor `[729, 2048]` (2.99 MB) and prompt |
| `distributed/projected/prompt_first` | Same | Same; tokenises the prompt while the edge encodes | Prompt first, then the tensor |
| `distributed/projected/int8` | Same, then quantises each row to int8 with an fp16 scale | Dequantises to bf16, then the language model | int8 tensor plus one scale per row (0.5005× the bytes) |

The first four are lossless and produce identical tokens on the same GPU; int8 was admitted by the quality rule of amendment A2a ([phase E](../../results/aws-phase-e-20261118/README.md)). The cloud runs one `adaptive-server` that accepts both raw images and projected tensors, so switching configuration never reloads a model. The prototype's three-configuration set (`CONFIGURATIONS`) is still the default for the local commands.

## Decision path

`prediction/r2/live.py` `AdaptiveRunner.infer(request)` does, per request:

1. **Sense** (`scheduler/sensing.py` `Sensor.sense`):
   - **edge load:** run the edge timing probe (`torch_edge_probe`: 80 bf16 matrix multiplies of 128 × 128, synchronising after each, per A4a). L1 when it takes at least the calibrated threshold (20.65 ms), otherwise L0. The probe shares the GPU with any competing load, so contention lengthens it;
   - **cloud load:** read the newest cloud state from `BackgroundCloudState` (A4b). C2 when the cloud GPU's power draw is at least 58 W, otherwise C0. Without a power reading the cloud counts as C0 and the source says so;
   - **network:** the uplink goodput and RTT measured by the supervisor's probe just before the segment (`network_pre.json`), never the experiment's nominal values.
2. **Build the inputs** (`prediction/r2/features.py` `UnitInputs`): the sensed loads and network, plus the pre-execution task features from [`configs/g5/task_features-short.json`](../../configs/g5/task_features-short.json): image width, height and bytes (which give the crop count) and the task. No generated output enters a decision.
3. **Predict and select** (`prediction/r2/g4.py` `decide`): predict all five configurations with `r2_component` ([r2 specification](../predictor-v2/r2-specification.md)), then `scheduler.selector.select(..., configurations=ALL_CONFIGURATIONS)` picks the lowest predicted latency among the feasible ones. If none is feasible the request fails with `NoFeasibleConfiguration`; this never happened.
4. **Execute** the chosen configuration through the same policy object the fixed placement uses, so its answer is identical to that configuration's.
5. **Record** in the request's metrics: every sensed value (`adaptive_sensed_*`), every prediction (`adaptive_predicted_ms_<configuration>`), the choice and its status, and the overhead split into selection time and total decision time (`adaptive_selection_ms`, `adaptive_overhead_ms`).

The benchmark times `infer` like any fixed placement, so **sensing, prediction and selection are inside the adaptive policy's measured latency.**

### The `r2_component` predictor in one paragraph

Latency is the sum of four terms fitted on the training cells only: the edge stage, the server, the network and a small client residual. The edge stage is linear in the crop count *k* per edge-load level (L0/L1); the cloud server is linear in *k* per cloud-load level (C0/C2), and the split server is a median per configuration and load. The network term is linear in Φ, a TCP slow-start transfer model of the request bytes over the measured goodput and RTT, with an initial window of 64 KiB chosen on training; at long RTT this tracks the measured transfers within a few per cent, where bytes ÷ rate misses by up to 2.4×. Its validation P90 error was 1.4–6.0% per configuration, with 98.3% top-1 agreement ([G3](../../results/aws-campaign-20261119/predictor_r2/README.md)).

## Background cloud sensing (A4b)

Pilot 1 polled the cloud's health endpoint before each decision. On the slow links of the grid that cost 126 ms per request, which failed the overhead check. `BackgroundCloudState` takes the poll off the decision path:
- a daemon thread polls the server's health response every second (`cloud_poll_interval_s`); a failed poll keeps the last sample and counts the error;
- every remote response also carries the server's latest `server_state_*` sample, and `observe` stores it;
- `latest()` returns the newer sample with its age (the server's own sample age plus the time since it arrived).

On the cloud, `adaptive-server --report-resource-state` samples NVML (power, utilisation, memory, clocks) every 0.5 s in a background thread, so neither side measures on the request path. The poll adds a few hundred bytes per second on the shaped link, disclosed in A4b.

## Where it runs

The adaptive policy is a **sixth placement of the interleaved benchmark** (`benchmark/interleaved.py`, placement `adaptive`). For each request, the five fixed configurations and the adaptive policy run back to back in a seeded random order, so the fixed runs give a per-request measured oracle under the same conditions. It is enabled by the `--adaptive-*` options of `interleaved-benchmark`:

| Option | Value in G5 |
| --- | --- |
| `--adaptive-calibration` | `configs/g5/calibration_r2_component.json` (a byte copy of the G3 calibration) |
| `--adaptive-sensing` | the calibrated sensing file, pinned in the [freeze record](../../results/g5-aws-20261120/freeze/g5_freeze.json) |
| `--adaptive-features` | `configs/g5/task_features-short.json` |
| `--adaptive-network-probe` | the segment's `network_pre.json` |
| `--adaptive-edge-probe` | `torch` (or `fixed:<ms>` in the loopback rehearsal) |

The campaign supervisor passes these when the grid has an `adaptive` section. A G5 grid lists **traces** instead of cells: ordered segments, each a network condition with an edge and a cloud load level. The trace order is shuffled per repetition; segments keep their order, so the state changes from one segment to the next while the adaptive policy runs (`campaign/grid.py`).

## Analysis

`prediction/r2/g5.py` computes, per measured request:
- the **oracle**: the fixed configuration with the lowest observed latency among that request's five fixed attempts (hindsight, not achievable online);
- the **regret**: the adaptive policy's latency minus the oracle's;
- the **label counterfactual**: the frozen selector rerun offline with the true load labels and the same network probe, taking that choice's observed fixed latency. It separates sensing error from prediction error.

`scripts/campaign/analyze_g5.py` writes the G5 summary; `scripts/campaign/final_analysis.py` adds the breakdowns by load state, trace and repetition, and the figures ([final analysis](../../results/final-analysis-20261121/README.md)).

## Design choices and their evidence

| Choice | Why |
| --- | --- |
| Per-request decisions, no hysteresis or dwell | Switching costs nothing measurable: latency after the same versus a different configuration differs by −0.12% to +0.9%, and stage preparation takes 0 ms (roadmap phase 5.2) |
| Edge load from a timing probe, not from GPU counters | The L4-6Q vGPU's utilisation counter reads 0% at L0 and 6% at L1 (A4) |
| A short, synchronised probe | The original 2048 × 5 burst could not separate L0 from L1 on the time-sliced vGPU (A4a) |
| Cloud load from GPU power | Separates C0 from C2 on training data (95th percentile at C0: 50.6 W; 5th at C2: 65.2 W) |
| Network from the periodic probe | The link is constant within a segment; the traces test step changes between segments, not drift within one |
