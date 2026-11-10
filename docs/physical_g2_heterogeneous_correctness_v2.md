# Physical G2 heterogeneous-GPU correctness v2

## Scope and decision

This version separates exact implementation/transport correctness from agreement between different GPU arithmetic implementations. The user approved this direction on 2026-11-10 after the exploratory N3 pilot. Native inference remains unchanged: pinned BF16 weights, Pillow preprocessing, greedy decoding, 32-token budget, the same 100 VQA requests, three repetitions, three placements, and frozen rotated ordering. No quantization, CPU weight offload, forced math attention, or FP32-linear runtime is adopted.

The old strict gate remains available and is the default unless this protocol is explicitly selected. Historical reports are immutable: the original N3 pilot still fails its strict cross-GPU token gate. A separate v2 assessment may reuse those raw exploratory measurements with explicit provenance; it does not relabel the original report.

## Required evidence

1. **Identical task and implementation inputs.** Bind model artifact/contract, original frozen manifest and image hashes, prompt/generation settings, pinned model source, active tokenizer/configuration, and runtime versions. Relocated reference inputs must have identical semantic request content, excluding only filesystem paths.
2. **Exact same-device equivalence across the complete frozen manifest.** On each physical GPU, all 100 monolithic/decomposed comparisons must have bit-identical projected tensors, output token IDs and text. Bind both full reports and per-request journals by SHA-256, including the tested model implementation hashes. Missing, changed, incomplete or mismatching reference evidence blocks launch.
3. **Exact transport invariants.** Keep bounded framing, payload checksum validation, exact BF16 serialization, client/server request-byte equality, expected projected payload size, and total byte accounting. No tolerance is introduced for transport errors. All existing provenance, schedule and timing checks remain active.
4. **Within-placement repeatability.** Successful measured repetitions of each frozen request must produce identical token sequences and text within that placement, and match its frozen same-placement control outputs. Edge/cloud controls come from the complete same-GPU parity checks; mixed-placement controls come from the preserved, repeatable native N3 pilot and are bound to its full plan and journal. Every request must have successful coverage in all three configurations, with repeated successful observations sufficient to assess repeatability. Failed attempts remain recorded and in failure-rate denominators; they are never erased or automatically retried into successes.
5. **Per-request task-score agreement.** Every successful placement/repetition for a request must have the same finite score under the frozen benchmark scoring function. Missing/unscored successes and unequal scores fail this gate. This is exact score-vector agreement, not an acceptance margin chosen to fit the results.
6. **Full discrepancy reporting.** Preserve the legacy cross-GPU exact-token result, counts/request IDs, decoded outputs, token lengths, quality scores, and all failures. Record projected-feature error diagnostics separately. This protocol does not claim bit-identical cross-GPU features, universal numerical equivalence, or a newly established numerical-error tolerance.

The scoring function is the project's provisional VQA metric. Agreement establishes no measured score change on this frozen 100-request cohort; it does not establish high VQA accuracy, official benchmark closure, equivalence on unseen requests, or identical wording. Official accuracy/quality analysis remains separately required by the research protocol. All actual generated token counts remain available so differing generation lengths are visible in performance analysis.

## Operational requirements

Keep measured bidirectional bandwidth/RTT validation for every network block, raw elapsed-time clock provenance, source/config/data contracts, durable journals, failure/stall supervision, disk reserves and queue cleanup. Both clean checkouts must match before launch.

The full edge model is a **WDDM shared-memory baseline**, not a claim of fitting in 4 GiB dedicated VRAM. Its native pilot reserved 4,570 MiB; dedicated capacity remains 4,096 MiB in hardware/planner budgets. Record this limitation explicitly. Close competing memory-heavy applications, require at least 6 GiB available Windows host RAM before launch and monitor a 2 GiB runtime floor during supervision (both thresholds are frozen in the site settings). A memory-pressure stop preserves all saved attempts and must be investigated before resume.

The cloud requires a live foreground WSL keepalive or separately validated equivalent. Systemd services alone do not keep WSL alive. Keep the user-confirmed `sleep infinity` console open, AC sleep/hibernate disabled, and the cloud model/probe services supervised. The exploratory pilot's clean cloud shutdown and telemetry gap remain part of its record.

## Evidence motivating the revision

At clean inference revision `1fb63014e67ff2ee0429ac3fbb08dbefc9d67beb`, the N3 pilot collected 900 successful measured attempts plus three warmups with zero OOMs/inference failures, valid pre/post network measurements, exact reconstructed application-frame lengths and no negative stage/transport residuals. All nine network conditions passed independent checks.

The native pilot has 222/300 exact cross-placement triplets; 26/100 unique requests differ in wording/tokens. Each placement is internally repeatable, and provisional per-request score vectors agree exactly (mean 0.03 on each placement). All 100 decoded RGB images, crop arrays and CPU-normalized BF16 tensors match across hosts; all 256 pixel normalization values match across CPUs/GPUs. Pinned snapshot sources, actual dynamic module sources, active tokenizer/configs and package versions match.

The edge selects native Flash/efficient attention; the Turing cloud selects math attention. A six-case common-math diagnostic improved exact agreement from 0/6 to 5/6 but increased edge mean latency to 42.7 seconds and reserved memory to 5,380 MiB. A targeted FP32-linear reference still diverged and took 203.8 seconds on edge. These diagnostic policies are not adopted.

PyTorch documents that attention implementations can produce different floating-point results and that reproducibility is not guaranteed across platforms: [SDPA, PyTorch 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html), [reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html). WSL lifetime behavior is documented by [Microsoft](https://learn.microsoft.com/en-us/windows/wsl/systemd#how-does-enabling-systemd-affect-wsl-architecture).

Raw evidence and the current-runtime estimate are in ignored `artifacts/unattended/site-preflight-20261110/`. The estimate is about **16.6 hours**, with a practical **14–22 hour** planning range including preflight, warmups, loading and network checks. It excludes outage/repair time. Changes to inference or memory policy require new timing evidence.

## Launch authorization

The user will launch the 8,100-request campaign manually. The assistant must not start it. Select v2 explicitly only after the complete same-device reference evidence, protocol implementation/tests and final site preflight pass. Configuration and evidence are frozen for resume; use a fresh campaign output for a changed protocol. Preserve the original strict pilot and any separate v2 assessment side by side.
