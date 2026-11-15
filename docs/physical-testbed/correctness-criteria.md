# Physical testbed: correctness criteria

How output equivalence is judged across placements, from the strict v1 contract through the observed CPU/GPU divergence to the user-approved heterogeneous-GPU criterion used for G2.

## Correctness contract v1

This contract applies prospectively to physical runs using the pinned Moondream2 model, frozen dataset manifest, and three configurations in `physical_pair_v1.yaml`. Record the Git SHA, model artifact SHA-256, model fingerprint, dataset SHA-256, host, device placement, seed, and output token IDs for every compared request. A changed model or generation contract requires a new contract and campaign revision.

### Transport and decomposition

For each identical request and the same cloud text placement, compare `cloud/raw_image` with `distributed/projected/synchronous`. Require both to finish without protocol/model errors and to produce exactly identical token IDs and decoded text. Any mismatch fails the transport gate and requires investigation before performance data is accepted. Report bytes and projected tensor dtype/shape for every split request.

### Cross-placement generation

Compare `edge/full` with the cloud-backed paths separately. Report exact token equality and decoded-text equality per request; do not treat a cross-device token mismatch as a transport failure when the two cloud-backed paths match. For a divergent request, repeat each affected placement at least five times with the same input and record whether tokens are stable within each placement. An unstable placement fails the reproducibility gate.

For labeled short-answer tasks, compute the same frozen task score for every placement and report both per-request and aggregate score differences. A placement is task-quality equivalent on a frozen set only if its per-request normalized answer and aggregate task score match the reference. For unscored descriptive prompts, report text and token differences; they cannot establish task-quality equivalence without a separately frozen evaluation rule. When official VQA evaluation is used, apply the official scorer to all placements and report paired accuracy differences with uncertainty. Do not replace this with visual inspection or post hoc normalization.

### Decision rules

| Observation | Decision |
| --- | --- |
| Cloud raw and projected tokens match on every frozen request | Transport/decomposition PASS |
| Any cloud raw/projected token mismatch | Transport/decomposition FAIL |
| A placement changes tokens across identical repetitions | Reproducibility FAIL |
| Stable cross-placement token difference with equal frozen task scores | Exact-token FAIL; task-quality PASS on that scored set |
| Stable cross-placement difference on an unscored request | Exact-token FAIL; task-quality UNDETERMINED |
| Different model fingerprint, manifest hash, or generation contract | Comparison INVALID |

The 2026-11-02 edge CPU/cloud GPU synthetic campaign has transport/decomposition PASS (12/12) and strict three-path exact-token FAIL (11/12). `correctness-001` is a descriptive prompt; its task-quality status remains UNDETERMINED under this contract. The later clean edge GPU/cloud GPU campaign has strict three-path exact-token PASS (12/12) on the same frozen manifest. These are separate device placements and separate gate results. Freeze this contract before using it for the next physical campaign and obtain edge/coordinator review of the task-quality criteria.

## CPU reference divergence

### Experiment

On 2026-11-02, the cloud host ran the pinned Moondream2 model with both vision and text on CPU. The runner was `scripts/campaign/physical_cpu_reference.py` at Git commit `e520538`; the working tree was clean for the full 12-request run. It used the unchanged `physical_pair_v1.yaml` model contract, seed 42, offline verified cache, and frozen correctness manifest. It compared its tokens with the archived physical report from edge commit `ee29a81`.

Raw results are in ignored local artifacts:

- `artifacts/phase1/cpu-reference-v1/single.json`
- `artifacts/phase1/cpu-reference-v1/repeat-001.json`
- `artifacts/phase1/cpu-reference-v1/full.json`
- `artifacts/phase1/physical-v1/phase1_smoke.json` (baseline)

The model contract fingerprint is `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`; the model artifact SHA-256 is `70a7d94c0c8349eb58ed2d9e636ef2d0916960f321ecabeac6354b8ba3d7403f`. The dataset manifest SHA-256 is `c5bc9df4a010dcefbcbb7fd60c9dd1e47dc0e0cf798acff579a1a004006ce8d1`.

### Result

| Request set | Cloud CPU vs edge CPU | Cloud CPU vs cloud GPU raw | Cloud CPU vs cloud GPU projected |
| --- | ---: | ---: | ---: |
| All 12 requests | 11/12 exact | 12/12 exact | 12/12 exact |
| `correctness-001` | Diverges at generated token 20 | Exact | Exact |

The cloud CPU generated the same `correctness-001` token sequence in five consecutive runs. All five matched both archived cloud GPU paths and differed from the archived edge CPU path. The initial single CPU run matched as well. This disproves the simple explanation that CPU placement alone causes the difference. The result is consistent with a host-specific backend, software, or CPU behavior, but the cause is not yet isolated.

Both hosts report the same package versions: PyTorch `2.7.1+cu128`, Transformers `4.52.4`, tokenizers `0.21.4`, Pillow `11.2.1`, and NumPy `2.4.6`.

A further diagnostic sent the edge CPU's projected BF16 tensor to a temporary cloud CPU text server, using the same smoke image and prompt. The 32 generated tokens matched the cloud CPU and GPU results exactly and differed from edge-local CPU `edge/full` at token 20. The raw edge output is archived at `artifacts/phase1/cpu-reference-v1/split_edge_vision_cloud_cpu.txt`. This narrows the mismatch to edge-local text inference or another host-specific runtime effect after the projection boundary. The cloud GPU server was restored after this diagnostic.

Three additional edge-local CPU `edge/full` runs returned identical 32-token sequences, all matching the earlier corrected smoke run and the archived physical campaign's edge row. Their raw output is archived at `artifacts/phase1/cpu-reference-v1/edge_repeat.txt`. Thus five observed edge CPU runs agree, while six observed cloud CPU runs (one initial run plus five repeats) agree on the other sequence. The observed divergence is repeatable, not random.

### Gate decision and next checks

P1 diagnostic isolation **passes**: transport/decomposition is exact for all 12 archived requests, each CPU placement is internally repeatable for the divergent request, and the cross-host difference is localized to edge-local text inference or another host-specific effect after the projection boundary. The exact low-level kernel or instruction responsible is unproven. The original strict three-path token-equivalence gate remains **failed** at 11/12. The next phase must define and freeze separate transport and cross-placement correctness criteria before physical profiling. Do not change the frozen model contract or silently relax the old gate.

## Heterogeneous-GPU correctness v2 (approved)

### Scope and decision

This version separates exact implementation/transport correctness from agreement between different GPU arithmetic implementations. The user approved this direction on 2026-11-10 after the exploratory N3 pilot. Native inference remains unchanged: pinned BF16 weights, Pillow preprocessing, greedy decoding, 32-token budget, the same 100 VQA requests, three repetitions, three placements, and frozen rotated ordering. No quantization, CPU weight offload, forced math attention, or FP32-linear runtime is adopted.

The old strict gate remains available and is the default unless this protocol is explicitly selected. Historical reports are immutable: the original N3 pilot still fails its strict cross-GPU token gate. A separate v2 assessment may reuse those raw exploratory measurements with explicit provenance; it does not relabel the original report.

### Required evidence

1. **Identical task and implementation inputs.** Bind model artifact/contract, original frozen manifest and image hashes, prompt/generation settings, pinned model source, active tokenizer/configuration, and runtime versions. Relocated reference inputs must have identical semantic request content, excluding only filesystem paths.
2. **Exact same-device equivalence across the complete frozen manifest.** On each physical GPU, all 100 monolithic/decomposed comparisons must have bit-identical projected tensors, output token IDs and text. Bind both full reports and per-request journals by SHA-256, including the tested model implementation hashes. Missing, changed, incomplete or mismatching reference evidence blocks launch.
3. **Exact transport invariants.** Keep bounded framing, payload checksum validation, exact BF16 serialization, client/server request-byte equality, expected projected payload size, and total byte accounting. No tolerance is introduced for transport errors. All existing provenance, schedule and timing checks remain active.
4. **Within-placement repeatability.** Successful measured repetitions of each frozen request must produce identical token sequences and text within that placement, and match its frozen same-placement control outputs. Edge/cloud controls come from the complete same-GPU parity checks; mixed-placement controls come from the preserved, repeatable native N3 pilot and are bound to its full plan and journal. Every request must have successful coverage in all three configurations, with repeated successful observations sufficient to assess repeatability. Failed attempts remain recorded and in failure-rate denominators; they are never erased or automatically retried into successes.
5. **Per-request task-score agreement.** Every successful placement/repetition for a request must have the same finite score under the frozen benchmark scoring function. Missing/unscored successes and unequal scores fail this gate. This is exact score-vector agreement, not an acceptance margin chosen to fit the results.
6. **Full discrepancy reporting.** Preserve the legacy cross-GPU exact-token result, counts/request IDs, decoded outputs, token lengths, quality scores, and all failures. Record projected-feature error diagnostics separately. This protocol does not claim bit-identical cross-GPU features, universal numerical equivalence, or a newly established numerical-error tolerance.

The scoring function is the project's provisional VQA metric. Agreement establishes no measured score change on this frozen 100-request cohort; it does not establish high VQA accuracy, official benchmark closure, equivalence on unseen requests, or identical wording. Official accuracy/quality analysis remains separately required by the research protocol. All actual generated token counts remain available so differing generation lengths are visible in performance analysis.

### Operational requirements

Keep measured bidirectional bandwidth/RTT validation for every network block, raw elapsed-time clock provenance, source/config/data contracts, durable journals, failure/stall supervision, disk reserves and queue cleanup. Both clean checkouts must match before launch.

The full edge model is a **WDDM shared-memory baseline**, not a claim of fitting in 4 GiB dedicated VRAM. Its native pilot reserved 4,570 MiB; dedicated capacity remains 4,096 MiB in hardware/planner budgets. Record this limitation explicitly. Close competing memory-heavy applications, require at least 6 GiB available Windows host RAM before launch and monitor a 2 GiB runtime floor during supervision (both thresholds are frozen in the site settings). A memory-pressure stop preserves all saved attempts and must be investigated before resume.

The cloud requires a live foreground WSL keepalive or separately validated equivalent. Systemd services alone do not keep WSL alive. Keep the user-confirmed `sleep infinity` console open, AC sleep/hibernate disabled, and the cloud model/probe services supervised. The exploratory pilot's clean cloud shutdown and telemetry gap remain part of its record.

### Evidence motivating the revision

At clean inference revision `1fb63014e67ff2ee0429ac3fbb08dbefc9d67beb`, the N3 pilot collected 900 successful measured attempts plus three warmups with zero OOMs/inference failures, valid pre/post network measurements, exact reconstructed application-frame lengths and no negative stage/transport residuals. All nine network conditions passed independent checks.

The native pilot has 222/300 exact cross-placement triplets; 26/100 unique requests differ in wording/tokens. Each placement is internally repeatable, and provisional per-request score vectors agree exactly (mean 0.03 on each placement). All 100 decoded RGB images, crop arrays and CPU-normalized BF16 tensors match across hosts; all 256 pixel normalization values match across CPUs/GPUs. Pinned snapshot sources, actual dynamic module sources, active tokenizer/configs and package versions match.

The edge selects native Flash/efficient attention; the Turing cloud selects math attention. A six-case common-math diagnostic improved exact agreement from 0/6 to 5/6 but increased edge mean latency to 42.7 seconds and reserved memory to 5,380 MiB. A targeted FP32-linear reference still diverged and took 203.8 seconds on edge. These diagnostic policies are not adopted.

PyTorch documents that attention implementations can produce different floating-point results and that reproducibility is not guaranteed across platforms: [SDPA, PyTorch 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html), [reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html). WSL lifetime behavior is documented by [Microsoft](https://learn.microsoft.com/en-us/windows/wsl/systemd#how-does-enabling-systemd-affect-wsl-architecture).

Raw evidence and the current-runtime estimate are in ignored `artifacts/unattended/site-preflight-20261110/`. The estimate is about **16.6 hours**, with a practical **14–22 hour** planning range including preflight, warmups, loading and network checks. It excludes outage/repair time. Changes to inference or memory policy require new timing evidence.

### Launch authorization

The user will launch the 8,100-request campaign manually. The assistant must not start it. Select v2 explicitly only after the complete same-device reference evidence, protocol implementation/tests and final site preflight pass. Configuration and evidence are frozen for resume; use a fresh campaign output for a changed protocol. Preserve the original strict pilot and any separate v2 assessment side by side.
