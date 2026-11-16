# G2 v2 baseline: preparation roadmap

G2 v2 is the physical baseline the rest of the thesis builds on: Predictor V2 r2 (G3), the selector against the measured oracle (G4), and the adaptive and final evaluations after that. This document lists everything that has to be finished **before going on site**, what has to happen **on site before the unattended run**, and how the run is closed out. Each item has a "done when" condition so readiness can be checked rather than estimated.

The original G2 ([results/physical-g2-20261110](../results/physical-g2-20261110/README.md)) stays valid evidence for its own contract and is reported alongside G2 v2. It is not modified.

## Progress

| Workstream | Status |
| --- | --- |
| W0 Source baseline | Done (`3bf33d8`, merged to `main`) |
| W1 Protocol | Drafted: [g2-v2-protocol.md](protocols/g2-v2-protocol.md). Sections 3–4 are frozen and executed; the rest is frozen after W8 |
| W2 Dataset and prompt | Done 15 November 2026: P1 selected ([calibration](../results/prompt-calibration-20261115/README.md)); 120-question manifest and split frozen |
| W3 Interleaved runner | Done 15 November 2026: `interleaved-benchmark`, CUDA-failure exit and restart helper, core-dump guard, swap residency and free-VRAM preflight in the campaign scripts; checked on the real edge GPU |
| W4 Resource-state logging | Done 16 November 2026: edge NVML, host memory and Windows GPU counters in every record; cloud state via `adaptive-server --report-resource-state`; deliberate spill visible |
| W5 Edge-load generator | Done 16 November 2026: supervised worker with heartbeat; L1/L2 calibrated and reproduced over three runs ([results](../results/edge-load-calibration-20261116/README.md)) |
| W6 Grid v2 and supervisor | Done 16 November 2026: `adaptive_vlm.campaign` and `scripts/campaign/run_g2v2_campaign.py`; loopback rehearsal of four sub-blocks (one loaded) completed 54/54 |
| W7 Downstream compatibility | Done 16 November 2026: `derive_g2v2_tables.py` turns rehearsal output into Predictor V2 partitions and measured-oracle tables with no manual edits |
| W8 Rehearsal and edge pilot | Local part done 16 November 2026: fault-injection rehearsal passed and the real-model edge pilot passed ([results](../results/g2v2-local-rehearsal-20261116/README.md)). The on-site pilot, run-length update and protocol freeze are in the on-site checklist below |
| W9 Runbook and on-site checklist | Done 16 November 2026: [runbook G2 v2 section](runbooks/physical-campaign.md#g2-v2-campaign), [printable checklist](runbooks/g2v2-onsite-checklist.md), v2 cloud unit, and tracked reference tooling |
| Long-answer task (added after the W8 winner estimate) | Done 16 November 2026: predeclared rule; D1–D3 failed the floor, amendment A1 selected E1 (32 tokens on 200/200); long manifest, split, 14 grid cells and pilot cells; [estimate](protocols/g2-v2-protocol.md#6-edge-gpu-load-draft) predicts all three placements winning in every split |

## Why a new baseline

| G2 (November 2026) | Problem | G2 v2 |
| --- | --- | --- |
| Edge/full ran resident and spilled into Windows shared memory (~11 s per request) | Edge timing reflects the 4 GiB VRAM limit, not the model | Edge/full uses swap residency (~1.5 s, identical tokens) |
| Free-form sentence prompt | Every placement scores 2.0% on official VQA, so a quality floor can't be enforced | Frozen short-answer prompt contract, quality-checked before the run |
| Nine blocks; cloud wins seven, distributed two, edge none | The best placement barely changes | Conditions chosen so the best placement changes with both network and edge GPU load |
| Request IDs crossed with every block; N2/N7 spent on r1, N4/N9 inspected | No clean validation or test data is left | Fresh request IDs and fresh network conditions for validation and test |
| One placement after another within each block; no resource state recorded | Block-to-block compute shifts (~490 ms in N5/N6) are confounded with network | Placements interleaved per request in random order; resource state logged per request |

## Definition of ready

Go on site only when all of these are true:

- [ ] One tagged, CI-green commit contains everything below, pushed to `origin/main`. *(Open: GitHub Actions is billing-locked, so the local suite stands in for CI; tag the commit after the protocol is frozen on site.)*
- [ ] The G2 v2 protocol document is frozen (committed and hash-recorded) before any G2 v2 measurement. *(On site, after the pilot: runbook step 7.)*
- [x] The new dataset manifest and split file are frozen, with validation and test IDs never seen by any earlier run.
- [x] The short-answer prompt was selected on a calibration set disjoint from the dataset, by a rule written down before scoring.
- [x] The interleaved runner, resource logging, edge-load generator and v2 grid support pass unit tests and a full local rehearsal (fake cloud), including a kill-and-resume.
- [x] An edge-only pilot on the real model (this laptop, swap residency, short prompt, each edge-load level) shows the expected timings and no spill at zero load. Spill is judged on dedicated usage; see W4.
- [x] Downstream tools (profile builder, Predictor V2 dataset preparation, oracle) read a rehearsal campaign's output without errors.
- [x] The runbook and an on-site checklist are updated for G2 v2.

## Decisions to freeze in the protocol

These shape the code, so settle them first. Each one has a recommendation; the protocol records what was chosen.

**D1. Edge GPU load becomes a campaign factor.** With swap residency, unloaded edge/full (~1.5 s) beats cloud (~2.3–2.9 s) and distributed in every condition of the planned network grid. Network conditions alone would therefore never change the winner. With an active 1 GiB competitor, edge rises to ~3.7 s and cloud or distributed wins, depending on the network. *Recommendation:* three edge-load levels: none, medium (512 MiB active competitor) and high (1 GiB). Run the full network grid at no load. Run the loaded levels on a subset of conditions drawn from all three splits (for example four training, one validation and one test condition), so validation and test both contain winner changes. Calibrate the medium level so edge sits near the cloud/distributed tie. *Decided after calibration (W5): both levels use a 512 MiB working set with different duty cycles (L1 1.85 s, L2 2.81 s); 1 GiB was dropped because it measures memory pressure rather than load.*

**D2. Network grid.** *Recommendation:* the 22-condition grid already proposed: 12 training conditions (60 IDs), 5 validation conditions (30 new IDs) and 5 test conditions (30 new IDs). Bandwidth runs from 5 to 200 Mb/s and added RTT from 5 to 150 ms. Every validation and test condition lies inside the training range, and none was used in G2. Set the netem `limit` explicitly (for example 10,000 packets) and require zero drops. Run each condition as two sub-blocks in a randomized order, never sweeping bandwidth in sequence. Keep one TCP connection per request and record that in the protocol. If the pilot shows the Wi-Fi uplink can't sustain 200 Mb/s, cap the grid at 150 Mb/s.

**D3. Short-answer prompt contract.** *Recommendation:* the question followed by a fixed instruction, such as "Answer with a single word or phrase." Pick from two or three candidate templates by official VQA accuracy on a calibration set of about 200 questions disjoint from every other set. Keep greedy decoding and `max_new_tokens: 32`. Store the template in the manifest revision so the adapter and model contract stay unchanged.

**D4. Edge weight state between interleaved requests.** Edge/full (swap) and distributed both run on the edge GPU. They can't be separate long-lived processes, because distributed's resident vision weights (~856 MiB) plus the swap peak (~3,164 MiB reserved) exceed the ~3.3 GiB free. Both must share one edge adapter, and that adapter's current stage depends on the previous request. *Recommendation:* before each timed request, put the adapter untimed into the state that a run of same-placement requests would leave it in: text stage active before edge/full, vision stage active before distributed. Record that activation's cost as a separate switch-cost field. Each record then holds the steady-state placement latency, has no order effect, and keeps the switch cost available to the selector.

**D5. Cloud precision.** The RTX 2070 Super emulates BF16, which accounts for much of cloud compute time. FP16 on the cloud would be faster, but it changes the output contract and the heterogeneous-correctness evidence. *Recommendation:* keep BF16 on both hosts, and note this as a limitation.

**D6. Dataset.** *Recommendation:* 120 new IDs (60/30/30), sampled reproducibly. Exclude the 100 G2 IDs and the prompt calibration IDs. Stratify by crop count so about 20% of images need two crops.

**D7. Repetitions and warmups.** *Recommendation:* three measured repetitions per (condition, request, placement), plus one excluded warmup per placement after each process start or load-level change.

**D8. Gates and analysis plan.** Write down before collection what counts as a valid condition (pre and post network evidence within tolerance, zero netem drops, correctness passing, attempt count as planned, failure rate below a set limit). Also record the quality floor used later by the selector and the exact analyses G3 r2 and G4 will run. Nothing is chosen after seeing G2 v2 results.

## Off-site workstreams

Everything here can be done now on the edge laptop. Estimates are working days for focused implementation and testing.

### W0. Source baseline (0.5 day)

1. Commit the descriptive-layout refactor and edge swap residency (`refactor/descriptive-layout`), open a PR, let CI pass, merge to `main`.
2. Every later workstream branches from that merge.

*Done when* `main` contains both pieces of work and CI is green.

### W1. Protocol draft (1 day, frozen at the end)

1. Write `docs/protocols/g2-v2-protocol.md` covering D1–D8, the condition table with split assignment, the load levels, the expected attempt count, the gates and the analysis plan.
2. Keep it as a draft until W2–W6 confirm feasibility. Freeze it (commit, record its SHA-256 in the grid config) after the pilot in W8.

*Done when* the protocol is committed and its hash is pinned by the campaign config.

### W2. Dataset and prompt contract (1.5–2 days)

1. Get the VQA v2 val2014 **annotations** file. *(Done: the copy from G2 scoring at `artifacts/official-vqa-g2/` matched the pinned hash.)*
2. Extend `adaptive-vlm import-vqa` (`src/adaptive_vlm/data/vqa_import.py`) with:
   - an exclusion list of request IDs;
   - crop-count stratification, reusing the existing crop computation;
   - a prompt template recorded in the manifest revision.
   Add tests.
3. Import the prompt calibration set (~200 questions) and fetch its images with `scripts/vqa/fetch_vqa_subset_images.py`.
4. Run each candidate template on the edge with swap residency (about 5 minutes per template). Score with the pinned official evaluator and apply the selection rule written down in advance. Record answer lengths and how often the 32-token cap is hit.
5. Import the frozen 120-ID G2 v2 manifest and build the split file (`configs/evaluation/physical_vqa_split_v2.json`), with the dataset and model-contract hashes.
6. Add the experiment config `configs/experiments/physical_vqa_v2.yaml`. The correctness smoke config `physical_pair_v1.yaml` uses synthetic fixtures, so it is reused unchanged.

*Done when* the manifest, split, configs and calibration results (with the selection record) are committed, and the chosen prompt's calibration accuracy is reported.

### W3. Interleaved runner (2.5–3 days, the critical path)

Today `scripts/campaign/run_physical_profile_block.py` calls `adaptive-vlm benchmark --mode X` once per placement, which is why placements were confounded with time in G2.

1. Add an interleaved benchmark command that does all of the following:
   - keeps one edge process holding a single swap-residency adapter, shared by edge/full and the edge half of distributed (D4);
   - uses the same cloud server for raw-image and projected requests;
   - runs the placements for each (request, repetition) in a seeded random order;
   - does the untimed state preparation with a logged switch cost;
   - writes a separate `requests.jsonl` per placement, in the existing record schema plus new fields, so the profile builder and later stages keep working.
2. Keep resume, `--max-consecutive-failures`, warmup exclusion and the clean-tree and hash checks.
3. **Recover from CUDA context failures.** During calibration, one `CUDA error: unknown error` appeared in about 730 swap-residency requests. After it, every later request in that process failed. The supervisor must restart the edge process and resume the sub-block. It must also run every campaign process with core dumps disabled (`ulimit -c 1`). WSL's crash handler ignores `ulimit -c 0`, and the 5.4 GB dump it wrote filled the Windows C: drive.
4. Pass `--edge-residency swap` through `run_physical_campaign.py`, `run_physical_profile_block.py` and the settings file. Have preflight refuse to start unless measured free VRAM covers the swap peak plus a margin at the current load level.
5. Write unit tests for order generation, state preparation, resume in the middle of an interleaved set, and record compatibility.

*Done when* a fake-model interleaved run on loopback produces three per-placement run directories that `build_profile` accepts, and a kill-and-resume produces no duplicates or gaps.

**Done (15 November 2026).**
- `adaptive-vlm interleaved-benchmark` (`src/adaptive_vlm/benchmark/interleaved.py`) writes per-placement run directories with an `interleave` block. `build_profile` verifies that block by regenerating the order.
- The fake adapter and `adaptive-server` now serve all three placements, so the full path runs over loopback in the test suite. That suite covers the profile check and kill-and-resume.
- A local CUDA error records the attempt and exits with code 75. `run_with_device_restarts` resumes it; the v2 supervisor uses it in W6.
- `disable_core_dumps()` is applied by the command and both campaign scripts.
- `edge_residency` is a campaign setting, and preflight requires 3,400 MiB free VRAM for swap.
- On the real RTX 3050:
  - an edge-only run reproduced the P1 calibration answers exactly;
  - alternating edge/full and distributed vision on one swap adapter kept the edge answers identical and the projected features bit-identical;
  - the untimed preparation moved the switch cost (about 90 ms to vision, about 267 ms to text) out of the timed request;
  - peak reserved memory was 3,184 MiB.
- Not yet run: a real three-placement run with a cloud server, which needs the cloud PC (W8 pilot, on site).

### W4. Resource-state instrumentation (1 day)

1. Before each timed request, record a snapshot on both hosts:
   - GPU clocks, temperature, power, utilization and used/free VRAM, from NVML on the edge and the cloud server;
   - on the edge only, the Windows shared GPU memory counter, so spill is visible. The `gpu_counters.ps1` approach from the memory study works for this.
2. Include the snapshots in the request records or in a sidecar file joined by attempt ID, and seal them with the block evidence.

*Done when* rehearsal records carry snapshots for both hosts and a deliberate spill shows up in them.

**Done (16 November 2026).**
- `adaptive_vlm.resource_state` reads NVML through ctypes, so no extra dependency is needed; every field is supported on the RTX 3050 under WSL. It also reads host memory and swap, and streams the Windows GPU adapter counters through `powershell.exe`.
- `interleaved-benchmark --resource-state` records the edge snapshot after stage preparation, outside the timer.
- `adaptive-server --report-resource-state` polls the cloud every 0.5 s and returns the latest sample with each response as `server_state_*`. The loopback test checks this end to end.
- On the real GPU, the resident edge model's spill shows as Windows dedicated usage at the 3,960 MiB maximum, against 3,271 MiB under swap.
- Shared usage is **not** a spill signal under swap: Windows counts the pinned weight copies (about 4.4 GB) as shared memory. Protocol gate 8.4 was changed to check dedicated usage.
- The real cloud-side numbers wait for the on-site pilot.

### W5. Edge-load generator (1 day)

1. Move the memory-study competitor (`artifacts/edge-memory-study/active_competitor.py`, currently untracked) into `scripts/campaign/`. Parameterize it by load level, then have the supervisor start it and confirm it's running before a loaded sub-block. Stop it afterwards and verify it stopped.
2. Calibrate the levels on this laptop: measure edge/full latency at each level, and check that medium load puts edge near the cloud/distributed range seen in G2.
3. Log the competitor's own throughput, so the effective load can be verified afterwards.

*Done when* the load levels reproduce their target edge latencies on three separate runs, and the supervisor recovers cleanly if the competitor dies.

**Done (16 November 2026).**
- The worker is `adaptive_vlm.benchmark.edge_load_worker`, a package module rather than a script. It is controlled by `adaptive_vlm.benchmark.edge_load.EdgeLoadGenerator`, which starts it, waits for its heartbeat, checks it before each attempt, records snapshots, and stops it and verifies the stop.
- A dead or stalled generator exits the run with code 76, which `run_with_device_restarts` resumes. Tests cover a killed worker, a stalled heartbeat and a failed start.
- The calibration chose a 512 MiB working set for both levels: L1 at 1.85 s and L2 at 2.81 s, against 1.27 s at L0. Both reproduced within 1.5% over three runs. The 1 GiB level was dropped because it is driven by memory pressure; see [results](../results/edge-load-calibration-20261116/README.md).
- Whether L1 sits near the cloud/distributed range is checked at the pilot, with a predeclared re-selection rule in protocol section 6.

### W6. Grid v2 and the campaign supervisor (1.5–2 days)

1. Add a schema-2 grid config (`configs/profiles/physical_network_grid_v2.json`) with:
   - conditions, their split, sub-blocks and edge-load level;
   - a seeded randomized execution order;
   - the netem `limit`;
   - the protocol hash.
2. Generalize `run_physical_campaign.py` beyond its hard-coded N1–N9 loop and 8,100-attempt check. Iterate the planned order, compute the expected attempt count from the config, and add a `tc -s qdisc` zero-drop check before and after each sub-block.
3. Keep `shape`, cleanup, the stall watchdog, host memory checks and the preflight sequence as they are, and extend the tests.

*Done when* the supervisor's tests cover the v2 grid, and a rehearsal runs at least two conditions (one loaded) end to end.

**Done (16 November 2026).** The G2 v1 scripts were left unchanged so G2 can still be reproduced; G2 v2 has its own package and script.
- **Grid:** [`physical_network_grid_v2.json`](../configs/profiles/physical_network_grid_v2.json) and `campaign.grid` give 34 cells and 102 sub-blocks, in a seeded order where no cell's sub-blocks are adjacent. Expected counts are computed from the split: 14,580 measured attempts and 306 warmups. The predeclared 200 Mb/s fallback is a setting (`rate_fallback`), and the protocol hash must be pinned in the grid before a physical run.
- **Network:** `campaign.network` applies netem with `limit 10000` on both hosts. It refuses to replace a foreign qdisc and reads the drop counters from `tc -s`. Probes use G2's tolerances. The unshaped baseline RTT is measured once at preflight and reused on resume, so it isn't hand-entered.
- **Sub-blocks:** `campaign.subblock` runs the protocol section 8 gates. An invalid sub-block is moved to `invalid/` and rerun, at most `max_subblock_attempts` times. Exit codes 75 and 76 are resumed up to `max_restarts_per_subblock` times, and each restart is logged in `restarts.jsonl`.
- **Supervisor:** `campaign.supervisor` with [`run_g2v2_campaign.py`](../scripts/campaign/run_g2v2_campaign.py) provides start, run, preflight, status and stop, with locking. Preflight covers both hosts' commit and clean tree, `sudo tc`, the cloud ports, disk, free VRAM, the reference evidence and runtime versions, the test suite and the correctness smoke. A stall watchdog follows journal progress, and shaping is cleaned up on every exit.
- **Tests:** [`test_g2v2_campaign.py`](../tests/campaign/test_g2v2_campaign.py) runs the supervisor over loopback with in-process children. It checks spill-triggered reruns, restarts, netem drops, a dead load generator and settings validation.
- **Rehearsal:** [`g2v2_loopback_rehearsal.json`](../configs/campaigns/g2v2_loopback_rehearsal.json) ran the real script against a real `adaptive-server --report-resource-state` (fake model), with the real GPU load worker for T01-L1. All four sub-blocks were valid, 54/54 attempts, with edge and cloud resource state in the records.
- **Not exercised yet:** the physical path (`tc`, SSH, iperf3) is covered only by unit tests with fake hosts. Its first real run is the on-site pilot.

### W7. Downstream compatibility (1 day)

1. Run a rehearsal campaign's output through the profile builder, `scripts/prediction/prepare_predictor_v2_data.py` and the oracle tools.
2. Extend the Predictor V2 dataset contract to carry edge-load level, resource-state features and the new split structure, where split conditions and request IDs are paired rather than crossed with every block.
3. Add tests so the dataset contract rejects mixed-split rows.

*Done when* rehearsal output produces a valid Predictor V2 training table and an oracle table with no manual edits.

**Done (16 November 2026).**
- **Pipeline:** [`derive_g2v2_tables.py`](../scripts/campaign/derive_g2v2_tables.py) runs the steps `features`, `profiles`, `partitions` and `oracle`, or all four with `all`.
- **Profiles:** each valid training and validation sub-block is profiled with `build_profile`. Evaluation sub-blocks are profiled only with `--splits evaluation --unseal-evaluation`. For an interleaved run, the profile rows now also carry the edge-load level, the snapshot taken before each request and the stage-switch cost. Fixed-placement (v1) profiles are unchanged.
- **Partitions** (`prediction.v2.dataset.campaign_v2`) use a new input schema, version 3. Each row adds the sub-block, cell, condition, split, edge-load level and 19 edge-state readings taken before the timer started. Cloud `server_state_*` values arrive with the response, so they are treated as outcomes, not inputs.
- **Split isolation:** request IDs and conditions are paired by split. A row whose request ID belongs to another split is rejected as a "mixed-split row". Validation targets are sealed with `000` permissions, and evaluation sub-blocks are counted from their plans and never opened.
- **Contracts:** [g2v2_dataset_contract.json](../configs/predictor/g2v2_dataset_contract.json) pins the model, tokenizer, host pair, correctness protocol, controlled network and clean source. Its same-device reference path is filled in once that reference is regenerated. [rehearsal_dataset_contract.json](../configs/predictor/rehearsal_dataset_contract.json) is the rehearsal version.
- **Measured oracle** (`evaluation.measured_oracle`): per attempt, the latency of each placement and the winner; per cell, the median, p90 and bootstrap 95% CI of the median (seed 1707, 2,000 resamples), the winner and whether it is a tie. It covers training by default, and evaluation needs explicit unsealing.
- **Fake model:** it now reports Moondream's metric names, so rehearsal rows pass the Predictor V2 target checks. A new 32-token fake config (`configs/model/fake_vqa.yaml`) is used by the rehearsal.
- **Rehearsal result:**
  - 36 training rows (18 at L0, 18 at L1);
  - 9 validation inputs, with targets sealed;
  - the evaluation sub-block was not opened;
  - a 2-cell training oracle table.

  [`test_g2v2_campaign.py`](../tests/campaign/test_g2v2_campaign.py) runs the same pipeline and checks mixed-split rejection and changed-profile rejection.
- **Not yet done:** Predictor V2 fitting itself still reads the G2 v1 schema-2 rows. Fitting r2 on schema-3 rows (adding load level and edge state as features) is model work for after the campaign, not campaign preparation.

### W8. Rehearsal and edge pilot (1 day)

1. **Local rehearsal.** Run the full supervisor on this laptop against a fake cloud over loopback:
   - two conditions, one of them loaded;
   - one hard kill and resume;
   - one forced competitor failure.
2. **Edge pilot (real model).** Run about 20 requests of edge/full and the edge half of distributed at each load level, using the short prompt. This checks timing, swap stability, and that no shared memory is used at zero load.
3. Replace the run-length estimate in the protocol with the pilot's measured timings.
4. Freeze the protocol (W1).

*Done when* every item in the definition of ready above is checked.

**Local part done (16 November 2026).** See [results](../results/g2v2-local-rehearsal-20261116/README.md).
- **Item 1:** [`rehearse_g2v2_faults.py`](../scripts/campaign/rehearse_g2v2_faults.py) ran the loopback campaign, hard-killed the supervisor mid-sub-block, restarted it, and killed the load worker during the loaded sub-block. All eight checks passed: completion, every attempt recorded, all sub-blocks valid, children exiting with the killed supervisor, the load failure resumed with exit 76, no stray processes, no reruns, and derived tables built.
- **Gap fixed:** a supervisor killed outright used to leave its benchmark and GPU load worker running, which would have loaded later L0 sub-blocks undetected. Children now exit with their parent, SIGTERM runs cleanup, a run accepts one writer, and the supervisor stops stray campaign processes before each sub-block.
- **Item 2, edge part:** the real model at L0/L1/L2 gave edge/full medians of 1.15, 1.62 and 2.42 s, and distributed's edge half 0.52, 0.82 and 1.36 s. There was no spill at L0, and edge answers were identical to calibration at every level.
- **Still on site:** the network part of the pilot, item 3 (measured run length) and item 4 (freezing the protocol). They are in the checklist below, and [`g2v2_pilot.example.json`](../configs/campaigns/g2v2_pilot.example.json) with [`pilot_grid_v2.json`](../configs/profiles/pilot_grid_v2.json) is ready for the pilot.

### W9. Runbook and on-site checklist (0.5 day)

1. Update `docs/runbooks/physical-campaign.md` for G2 v2, covering the settings file, the v2 grid, edge residency and the load generator.
2. Add a printed on-site checklist.
3. Update `configs/systemd/adaptive-vlm-cloud.service` to the v2 experiment config.

*Done when* someone following only the runbook could start the campaign.

**Done (16 November 2026).**
- **Runbook:** the [G2 v2 section](runbooks/physical-campaign.md#g2-v2-campaign) now runs end to end in eleven steps: code freeze and rehearsal, cloud PC, network, correctness reference, settings, pilot with go/no-go, protocol freeze, launch, monitoring, recovery (a table of what is automatic and what needs you), and after the run. It ends with the output layout.
- **Checklist:** [g2v2-onsite-checklist.md](runbooks/g2v2-onsite-checklist.md) is the printable version, with a progress log.
- **Cloud service:** `configs/systemd/adaptive-vlm-cloud.service` now serves `physical_vqa_v2.yaml` with `--report-resource-state`.
- **Gap 1, the correctness reference.** For G2 it was built on site with untracked scripts, one of which imports a module the refactor renamed. A runbook reader could not have rebuilt it, and preflight refuses to start without it. [`build_g2v2_reference.py`](../scripts/campaign/build_g2v2_reference.py) now produces it from tracked code, with the logic in `adaptive_vlm.campaign.reference`. Its four steps are `same-device`, `runtime`, `control` and `bundle`. The bundle step verifies the result exactly as preflight does, and the tests check that the real verifier accepts a stub bundle and rejects a single mismatch.
- **Gap 2, NVML on the cloud.** The cloud's WSL NVML segfaults on device enumeration, which is why `run_physical_cloud_gpu.py` exists. W4's in-process NVML reads could therefore have crashed the inference server. NVML is now probed once in a throwaway process, on both hosts. If the probe fails, state reporting continues without GPU fields and records `nvml_available: false`.

**Total off-site effort: about 10–12 working days.** That's more than the 1–1.5 weeks estimated earlier. The single shared edge adapter (D4) makes the interleaved runner more involved, and the load generator now has to be a full campaign factor (D1). W2, W4 and W5 can run in parallel with W3. W3 → W6 → W8 is the critical path.

## On site, before the unattended run (about half a day)

The steps are in the [runbook's G2 v2 section](runbooks/physical-campaign.md#g2-v2-campaign), and as tick boxes in the printable [on-site checklist](runbooks/g2v2-onsite-checklist.md). In short:
1. Same commit on both hosts.
2. Cloud services, including the v2 unit with resource state.
3. Network: addresses, and the decision on the 200 Mb/s fallback.
4. Correctness reference with `build_g2v2_reference.py`.
5. Settings files.
6. The pilot and its go/no-go checks, which also complete W8 items 2–3.
7. Freeze the protocol and pin its hash (W8 item 4).
8. Host hygiene.
9. Launch.

## The run

| Part | Attempts | Estimate |
| --- | ---: | ---: |
| No-load grid: 22 conditions × 3 placements × 3 repetitions (60 train / 30 validation / 30 test IDs) | 9,180 | ~6.2 h |
| Loaded conditions: 2 load levels × 6 conditions | ~5,400–6,500 | ~4.7–5.6 h |
| Per-sub-block network probes, shaping and load start/stop | — | ~2–3 h |
| Per-sub-block edge model load (78–90 s × 102, confirm at the pilot) | — | ~2.2–2.6 h |
| Long-answer task (section 3.5): 14 cells, 42 sub-blocks, 2,970 attempts at ~2.8 s, plus model loads and probes | 2,970 | ~4.0–4.5 h |
| Preflight and GPU correctness campaign | — | ~1 h |
| **Total** | **17,550** | **~20–23 h (range 19–27 h): plan two sessions** |

These are model-based estimates calibrated on G2 (the model reproduces G2's 14.3 h of inference). The edge pilot and the on-site pilot replace them with measured numbers. The supervisor resumes from validated checkpoints, so the run can be split over two sessions without losing data.

## After the run

1. Validate every sub-block. Check the attempt count against the plan and archive the evidence under `results/physical-g2v2-<date>/` with `SHA256SUMS`, following the G2 results layout.
2. Run the official VQA evaluation on all placements and check the quality floor from D8.
3. Write the G2 v2 report: winner by condition and load, crossover bands, switch costs and resource-state summaries. Report it alongside the original G2.
4. Only after that, fit Predictor V2 r2 on the training split and open the validation split once (G3). The test split stays sealed until the selector is frozen (G4).

## Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Wi-Fi uplink can't sustain 200 Mb/s, or varies during the run | High-bandwidth conditions are invalid | Pilot check; predeclared 150 Mb/s cap; pre/post iperf3 per sub-block |
| Edge VRAM margin too small under the load generator | Spill makes loaded edge timing unstable | Free-VRAM preflight; shared-memory counter per request; calibrate levels off site |
| Short prompt still scores low | The selector's quality floor isn't meaningful | Calibration set and selection rule in W2; report calibration accuracy before freezing |
| Interleaving changes thermal behaviour compared with G2 | Timings not comparable with G2 | Expected and intended; G2 v2 is a new baseline, and resource state is logged |
| CUDA context failure on the edge (one in ~730 swap requests during calibration) | Every later request in that process fails; a crash dump can fill C: | Supervisor restarts the edge process and resumes (W3); core dumps disabled; keep at least 15 GB free on C: |
| Run longer than one night | Fewer hours of supervision | Checkpointed resume; plan two sessions if the pilot points to more than 16 h |
