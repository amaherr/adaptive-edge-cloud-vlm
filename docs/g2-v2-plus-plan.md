# G2 v2+ plan: full configuration space and cloud load (option B)

> **Platform:** this campaign now runs on AWS. Phases, platform work and retargeted steps are in the [AWS roadmap](g2-v2-plus-aws-roadmap.md); its section 4 overrides the off-site and on-site steps below where they differ (B0 is split, B3 moves to AWS, section 6 "on site" becomes phases D–F).

**Status: plan, 17 November 2026. Nothing here is built yet.** This extends the frozen G2 v2 design (commit `bb6b48d`, [preparation notes](g2-v2-preparation.md)) so that one on-site campaign covers everything the thesis aim names, except the dynamic run, which needs the fitted selector (section 8). It replaces G2 v2 as the campaign to run; G2 v2's frozen dataset, prompts, load levels and tooling are all kept.

## 1. Why

The thesis asks whether a lightweight, task- and system-aware scheduler can pick a **fine-grained internal VLM execution configuration** under changing **edge, cloud and network** conditions, approaching the oracle while cutting tail latency and communication **without degrading task quality**. G2 v2 covers network, edge load and task type, but has three gaps against that aim:

| Aim | G2 v2 (frozen) | G2 v2+ |
| --- | --- | --- |
| Multiple internal execution configurations | 3: edge/full, cloud/raw image, one split | **5**: adds a prompt-first pipeline and an int8 payload on the same split |
| Cloud resource state | cloud always idle (state only logged) | **cloud load C0/C1** on the cells where cloud otherwise wins |
| Task quality as a real constraint | every configuration lossless | **int8 is lossy**: admitted only under a predeclared quality rule |
| Network, edge state, request type | 22 conditions, L0/L1/L2, short/long | unchanged |

Other split points don't add useful choices on this model. Before the projector the tensor is larger (729 × 2,304); inside the vision encoder there are per-crop tensors of 3.4–17 MB; after prefill the KV cache is about 150 MB. So the extra configurations vary the **pipeline** and the **payload encoding** at the one useful split, not the split location.

## 2. Design

### 2.1 Configurations

| ID | Edge | Wire | Cloud | Status |
| --- | --- | --- | --- | --- |
| `edge/full` | vision + text | — | — | exists |
| `cloud/raw_image` | read image | JPEG (~160 KB) | vision + text | exists |
| `distributed/projected/synchronous` | vision | bf16 tensor, 2.99 MB | text | exists |
| `distributed/projected/prompt_first` | vision, while the prompt is already on the cloud | bf16 tensor, 2.99 MB | prompt prefill overlapped, then text | policy and server exist; runner, profile and dataset don't |
| `distributed/projected/int8` | vision + per-row int8 quantization | int8 tensor + fp16 row scales, ~1.50 MB | dequantize, then text | new |

`prompt_first` uses the same arithmetic as `synchronous`, so its outputs must match the synchronous control exactly. `int8` is lossy and is admitted only under the section 3 quality rule; if it fails, the campaign runs with four configurations.

### 2.2 Factors and cells

- **Network:** 22 conditions, split by condition (training 12, validation 5, evaluation 5). Unchanged.
- **Edge load:** L0; L1 and L2 on T02, T04, T06, T09, V03 and E03. Unchanged.
- **Task:** short (P1, 120 IDs) and long (E1, 60 IDs, 14 cells). Unchanged.
- **Cloud load (new):** C1 is a GPU competitor on the cloud, the same worker as the edge load, to be calibrated on site (section 6). It runs on the **L2 cells**, where cloud wins at C0:

| Split | C1 cells |
| --- | --- |
| training | T02, T04, T06, T09 at L2 (short); T06, T09 at L2 (long) |
| validation | V03 at L2 (short); V04 at L2 (long) |
| evaluation | E03 at L2 (short); E04 at L2 (long) |

- **Repetitions:** 2 instead of 3. Interleaving already controls drift within a sub-block, and the saved time pays for the extra configurations.

### 2.3 Size

| Part | (Condition, ID) pairs | × reps × configs | Measured attempts |
| --- | ---: | ---: | ---: |
| Short task, C0 | 1,620 | × 2 × 5 | 16,200 |
| Long task, C0 | 330 | × 2 × 5 | 3,300 |
| C1 cells | 390 | × 2 × 5 | 3,900 |
| **Total** | 2,340 | | **23,400** |

There are 58 cells, 116 sub-blocks and 580 excluded warmups. The estimate is about 15.6 h of inference (≈2.4 s per attempt) plus about 5.8 h of per-sub-block overhead (edge model load ~85 s, probes and shaping, cloud-load start), plus 1 h of preflight: **about 22–23 h, range 20–27 h, over two nights**. With four configurations (int8 not admitted) it is about 18–19 h.

### 2.4 Predicted winners (pre-run estimate)

Same model as `estimate_g2v2_winners.py`, with int8 at half the transfer plus 12 ms, prompt-first hiding about 90 ms of cloud prompt prefill, and C1 as a 1.5× cloud slowdown:

| | C0 | C1 (L2 cells) |
| --- | --- | --- |
| short L0/L1 | edge | — |
| short L2 | cloud | **edge** |
| long L0 | edge | — |
| long L1 | **int8 / prompt-first** (distributed variants) | — |
| long L2 | cloud | **distributed int8 / prompt-first** |

All five configurations are predicted to win somewhere, and each split contains winner changes driven by the network, edge load, cloud load and task. Several C1 margins are small (4–133 ms), which is acceptable: near-ties are where a scheduler's prediction quality shows.

## 3. Decisions to predeclare (protocol amendment A2)

Commit these **before** any int8 quality run and before calibrating cloud load:

1. **Configuration space** as in 2.1, with configuration IDs and run-directory names.
2. **int8 encoding:** per-row symmetric int8 over the 729 × 2,048 projected tensor, with fp16 scale per row, dequantized to bf16 on the cloud before text generation.
3. **int8 quality rule** (off-site, calibration set only, section 4 B3):
   - Short task (P1, 200 calibration questions): official VQA accuracy with int8 within **1.0 pp** of lossless, and exact answer agreement at least **95%**.
   - Long task (E1, same 200 images): report token agreement with lossless and the length distribution. There is no gate, because there is no quality metric for this task.
   - If the rule fails, int8 is dropped. The campaign runs four configurations and the scheduler never sees int8.
4. **Quality-constrained oracle and selector:** int8 is a candidate only for buckets where the rule passed. The oracle reports both the unconstrained and the quality-constrained winner.
5. **Cloud load:**
   - C1 target: cloud/raw server time **1.4–1.6×** its C0 median at P04 (35/20, unshaped). Pick the smallest duty in the sweep that lands in range (section 6).
   - C1 cells as in 2.2.
   - Validity gate: every cloud and distributed response in a C1 sub-block reports the cloud competitor alive with a heartbeat younger than 5 s; no response in a C0 sub-block reports it alive.
6. **Correctness:**
   - `prompt_first` outputs must equal the per-GPU synchronous control.
   - `int8` outputs must equal a per-GPU **int8 control** (deterministic and repeatable), bound in the reference bundle.
7. **Inputs available to the scheduler:** the edge snapshot before each request, plus the **previous response's** cloud state (`server_state_*` lagged by one request), and the edge and cloud load levels. The current response's cloud state is an outcome, never an input.
8. **Repetitions 2**, the updated counts, run estimate and pilot.

## 4. Workstreams (off site)

| ID | Work | Main files | Effort |
| --- | --- | --- | --- |
| B0 | Write and commit amendment A2 (section 3) | `docs/protocols/g2-v2-protocol.md` | 0.5 day |
| B1 | Configuration space: runner, profile, dataset, oracle and selector on configuration IDs | `benchmark/interleaved.py`, `commands/interleaved.py`, `benchmark/profile.py`, `scheduler/configuration.py`, `campaign/*`, `prediction/v2/dataset/*`, `evaluation/measured_oracle.py` | 1.5 days |
| B2 | int8 payload | `transport/tensors.py`, `execution/distributed.py`, `transport/server.py`, `commands/server.py`, `models/fake.py` | 1 day |
| B3 | int8 quality calibration under rule A2.3 | new `scripts/campaign/calibrate_int8.py`; `results/int8-quality-<date>/` | 0.5 day |
| B4 | prompt-first in the runner, plus fake-server support | `commands/interleaved.py`, `transport/server.py`, `models/fake.py` | 0.5 day |
| B5 | Cloud load | `benchmark/edge_load_worker.py`, new `campaign/cloud_load.py`, `campaign/grid.py`, `campaign/subblock.py`, `campaign/supervisor.py`, `resource_state.py`, `commands/server.py` | 1.5 days |
| B6 | Grids, pilot, rehearsal and estimate | `configs/profiles/*`, `configs/cloud_load/levels_v1.json`, `estimate_g2v2_winners.py` | 0.5 day |
| B7 | Correctness references with int8 controls | `campaign/reference.py`, `build_g2v2_reference.py`, `benchmark/heterogeneous_correctness.py` | 1 day |
| B8 | Downstream data: config IDs, cloud level, lagged cloud state | `prediction/v2/dataset/campaign_v2.py`, `evaluation/measured_oracle.py`, dataset contracts | 1 day |
| B9 | Tests, rehearsal, docs, freeze, edge references | tests, runbook, checklist | 1 day |

**Total: about 9 working days of effort, 6–8 calendar days with overlap.** That is more than the 3–5 days first quoted; the configuration space touches every downstream stage, and int8 adds a quality calibration and its own references.

### B0. Amendment A2
1. Write section 3 into the protocol as amendment A2, dated, with the counts from 2.3 and the estimate from 2.4.
2. Commit before B3 or any cloud-load measurement.

*Done when* A2 is committed and no int8 or cloud-load result exists yet.

### B1. Configuration space
1. Add `DISTRIBUTED_PROMPT_FIRST` and `DISTRIBUTED_INT8` to `ConfigurationId` and `CONFIGURATIONS` (split `projected`, pipelines `prompt_first` and `synchronous`, payload `bf16` or `int8`).
2. Interleaved runner:
   - key policies, run directories and plans by **configuration**, not by mode;
   - use the run directories `edge`, `cloud`, `distributed`, `distributed_prompt_first` and `distributed_int8`;
   - prepare the vision stage for all three distributed variants.
3. Profile builder: map (mode, pipeline, payload) to the configuration ID. Make the existing `distributed` directory explicit as `synchronous`/`bf16`.
4. Sub-block gates, Predictor V2 partitions, the measured oracle and `derive_g2v2_tables.py`: iterate the grid's configuration list instead of `PLACEMENTS`.
5. Grid: `configurations` list (default: the three G2 v2 ones, so old grids keep working).

*Done when* a loopback run with five configurations yields profiles, partitions and an oracle table, and G2 v2 grids still produce identical plans.

### B2. int8 payload
1. `TensorPayload`: `quantize_int8_rows` / `dequantize`, metadata `quantization: int8_rowwise`, and `scales` carried in the payload bytes after the data.
2. `DistributedExecutionPolicy(payload_format="int8")`: quantize on the edge, record `quantize_ms` and the payload bytes.
3. Server: advertise `projected_int8`, dequantize to bf16 on the text device, record `dequantize_ms`.
4. Fake adapter and fake server support it.
5. Tests:
   - round trip within int8 error bounds;
   - wire bytes 0.50 ± 0.01 of bf16;
   - the server rejects unknown quantization;
   - deterministic output.

*Done when* the loopback run shows int8 requests at about half the request bytes, with all checks green.

### B3. int8 quality calibration (off site, edge GPU only)
1. On the edge laptop, run the decomposed path in the same process: vision → int8 quantize → dequantize → text. Run it against lossless on both calibration manifests (P1: 200 questions; E1: 200 images).
2. Score with the pinned official VQA evaluator, using the section 9 scoring choice (lowercased primary and unchanged reported, as for P1).
3. Apply rule A2.3 and commit `results/int8-quality-<date>/` with the predictions, scores and decision.

*Done when* the decision is committed. If int8 fails, remove it from the grid's configuration list; the rest of the plan stands.

### B4. prompt-first
1. Interleaved command: build a second `DistributedExecutionPolicy(pipeline="prompt_first")` sharing the same encoder.
2. Fake server: add `prompt_prepare_function` and `prepared_tensor_inference_function` for the fake adapter.
3. Check that the server's pending-prompt state can't leak between interleaved requests (one pending prompt per request ID, cleaned on timeout).

*Done when* prompt-first rows appear in the loopback run with `prompt_round_trip_ms` recorded, and their outputs equal synchronous.

### B5. Cloud load
1. Worker:
   - `--detached`, which skips `exit_with_parent` for SSH launch;
   - apply the WSL NVML workaround (`torch.cuda._device_count_nvml = lambda: -1`) before CUDA initializes;
   - write the heartbeat to a fixed cloud path, `/tmp/adaptive-vlm-cloud-load/heartbeat.json`.
2. `campaign/cloud_load.py`, with a `Runner` that is SSH on the physical setup and a local shell in loopback rehearsal:
   - `start(level)`: `setsid nohup` the worker, then wait for a running heartbeat with the working set allocated;
   - `check()`: read the heartbeat;
   - `stop()`: `pkill -TERM -f` on the marker, then verify no process remains and keep the final heartbeat;
   - `reap()`: stop any leftover worker before every sub-block (C0 and C1).
3. Server, `--load-heartbeat PATH`: the background sampler adds `load_level`, `load_alive`, `load_heartbeat_age_s` and `load_duty` to `server_state_*`.
4. Grid: `cloud_load_levels`, plus C1 cells as `[condition, edge level, task]` triples. The cell and sub-block IDs gain `-C1`; C0 IDs are unchanged.
5. Supervisor: start C1 after shaping and before the benchmark; stop it after; reap before each sub-block and at every exit.
6. Gate A2.5 in `campaign/subblock.py`.
7. Tests:
   - the start, check, stop and reap lifecycle with a fake runner;
   - a dead worker invalidates the sub-block;
   - a stray worker during C0 invalidates it;
   - the supervisor still stops the worker after a crash.

*Done when* the loopback rehearsal runs a C1 sub-block with the real GPU worker started through the local runner, and the fault rehearsal (B9) passes a kill of the cloud worker.

### B6. Grids, pilot, rehearsal, estimate
1. `physical_network_grid_v2.json`: five configurations, repetitions [0, 1], C1 cells, and `cloud_load_levels_path`.
2. `configs/cloud_load/levels_v1.json`: the candidate duties for the on-site sweep (512 MiB at 25/50/75/100% busy), with C1 left unset until calibration.
3. Pilot grid: add `P05-L2-C1` (short) and `P05-L2-long-C1`, so the pilot shows cloud load flipping a winner.
4. Rehearsal grid: one C1 cell, five configurations.
5. `estimate_g2v2_winners.py`: model the int8, prompt-first and cloud-load scaling used in 2.4, so the predicted table is reproducible.

### B7. Correctness references
1. Reference bundle: per task, `distributed_controls` for `synchronous` and `int8` (1 warmup + 3 repetitions each). Expected outputs:
   - `prompt_first`: from the synchronous control;
   - `int8`: from the int8 control.
2. `build_g2v2_reference.py control --payload int8`.
3. Extend the verifier and `evaluate_rows` to the configuration IDs.
4. Tests: a stub bundle with both controls is accepted; an int8 output mismatch is rejected.

### B8. Downstream
1. Predictor V2 schema-3 inputs gain `cloud_load_level` and `previous_cloud_state`. The previous response's `server_state_*` in the same sub-block is null for the first request.
2. Dataset contract: the configuration list, plus the int8 admission decision from B3.
3. Measured oracle: five configurations, plus unconstrained and quality-constrained winners per cell.
4. Selector availability: the configuration list comes from the grid.

### B9. Tests, rehearsal, docs, freeze
1. Full suite, then the fault rehearsal extended with a cloud-worker kill and a C1 sub-block.
2. Runbook and checklist: cloud-load calibration step, two controls per task, new pilot checks.
3. Freeze commit and push, then rebuild both edge same-device references on that commit (about 70 min).

## 5. Order

```
B0 ─┬─ B1 ─┬─ B4 ─────────────┐
    │      ├─ B2 ── B3 ───────┤
    │      └─ B5 ─────────────┼─ B6 ── B7 ── B8 ── B9 (freeze, edge refs)
    └─────────────────────────┘
```

B1 is the critical path. B2/B3 (int8) and B5 (cloud load) are independent of each other once B1 is in. B3 decides whether int8 stays, so it should finish before B6 fixes the grid.

## 6. On site, in addition to the current checklist

1. **Cloud-load calibration** (about 30 min, before the pilot). On the cloud, unshaped, edge idle, run 20 cloud/raw requests at C0 and at each candidate duty. Choose the smallest duty whose median cloud server time is 1.4–1.6× C0, and record it in `levels_v1.json` **(commit)**. If no duty reaches 1.4×, record a dated amendment before continuing.
2. **References.** Per task: the cloud `same-device` run, a synchronous control and an int8 control, then `bundle`. That is about 25 min more than G2 v2.
3. **Pilot** (about 2 h). Additional go/no-go checks:
   - every configuration in the grid wins at least one pilot cell, or is within the CI of the winner;
   - `P05-L2-C1` changes the winner compared with `P05-L2`;
   - C1 sub-blocks pass the cloud-load gate;
   - int8 pilot outputs match the int8 control.
4. **Run:** two nights, about 22–23 h, resuming between them with `start`.

## 7. Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| int8 fails the quality rule | Four configurations only | Predeclared drop; the plan and counts already cover it |
| Cloud WSL NVML crash in the cloud worker | Worker dies; C1 sub-blocks invalid | NVML workaround in the worker; on-site calibration exercises it first |
| C1 cannot reach 1.4× | Cloud load too weak to change winners | Duty sweep up to 100%; amendment before collecting |
| Detached cloud worker survives a supervisor crash | Silent load on C0 sub-blocks | Reap before every sub-block, plus the C0 "no live worker" gate |
| Prompt-first leaves pending state on the server | Rejected or mismatched requests | One pending entry per request ID with timeout cleanup; tested in B4 |
| Run longer than two nights | Fewer supervised hours | Resume between sessions; repetitions already cut to 2 |
| More `src/` changes after the freeze | Edge references invalid | Freeze only after B9; rebuilding takes about 70 min |

## 8. Not covered here

- **The dynamic evaluation (G4/G5).** It needs Predictor V2 r2 fitted on G2 v2+ training cells, validated once on the validation cells, and a frozen selector. Then one adaptive campaign changes conditions during the run (network traces, edge and cloud load switching on and off) and compares the selector with each fixed configuration and with the measured oracle on p50/p95 latency, bytes and quality. Most of the pieces exist from Phases 5–7 (`adaptive-benchmark`, trace replay, the live monitor) and need updating to the v2+ configuration space. This is a second site visit, after the analysis of G2 v2+.
- **A reduced vision budget** (global crop only) as a sixth configuration. It is the strongest "internal configuration" in principle, but it changes the model contract and needs its own references and quality rule. It's a stretch goal if B1–B9 finish early.
