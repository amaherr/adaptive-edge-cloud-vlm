# Physical testbed: GPU placement and memory

Where each stage runs on the physical pair and how much GPU memory it needs.

## GPU placements

The pinned Moondream2 contract uses BF16 and greedy generation. The cloud model runs on its RTX 2070 SUPER through `scripts/campaign/run_physical_cloud_gpu.py`. The edge RTX 3050 has 4,096 MiB reported VRAM. The first GPU placement smoke tests used `tests/fixtures/moondream_smoke.ppm` and the prompt `Describe this image.`.

| Scheduler choice | Vision placement | Text placement | Smoke result |
| --- | --- | --- | --- |
| `cloud/raw_image` | Cloud `cuda:0` | Cloud `cuda:0` | Pass |
| `distributed/projected/synchronous` | Edge `cuda:0` | Cloud `cuda:0` | Pass; exact cloud token sequence |
| `edge/full` | Edge `cuda:0` | Edge `cuda:0` | Pass; exact cloud token sequence |

The edge GPU split smoke used a BF16 `[729, 2048]` projection of 2,985,984 bytes. Edge vision took 473.5 ms and peaked at 913.8 MiB allocated, 942 MiB reserved. The edge full GPU smoke took 9,302 ms and peaked at 4,128.2 MiB allocated, 4,238 MiB reserved. Both raw outputs are archived under `artifacts/phase1/edge-gpu-validation-v1/` and ignored by Git.

The full GPU peak exceeds the card's reported 4,096 MiB VRAM, despite that smoke request completing. Classify `edge/full` GPU as **conditionally feasible** pending memory-pressure testing. Do not infer stable 4 GB fit from one success. These timings are diagnostic single requests, not comparable performance estimates.

### Frozen 12-request GPU campaign

The edge ran `scripts/verification/verify_resident_configurations.py` with both edge devices set to `cuda:0`. All 36 inferences completed and all 12 request triples returned exactly equal token IDs and text (`complete_and_equal: true`). The new edge GPU output matches the earlier cloud GPU output for all 12 requests; it differs from the earlier edge CPU output only on `correctness-001`. The raw report is archived at `artifacts/phase1/edge-gpu-validation-v1/phase1_smoke.json`, ignored by Git. Its model fingerprint and dataset manifest SHA-256 match the first physical campaign.

The first report recorded edge Git commit `2415685` with `dirty: true`, so the edge repeated the campaign after `git status --short` returned empty. The second report, archived at `artifacts/phase1/edge-gpu-validation-v1/phase1_smoke_clean.json`, records the same commit with `dirty: false`, 36 completed inferences, and exact equality for all 12 triples. It uses the same model fingerprint and dataset manifest SHA-256. This closes the strict token-equivalence gate for the specified all-GPU placement on the frozen synthetic manifest.

Peak CUDA reserved for the resident full edge model was 4,238 MiB in each request of the first campaign. The split rows in that campaign also report roughly 4,124 MiB peak allocated because the runner kept the full edge model resident for all three policies. The separate vision-only smoke measured 913.8 MiB peak allocated and is the relevant isolated vision footprint. Memory feasibility under additional load remains conditional.

## Memory feasibility (v1)

The pinned BF16 Moondream2 model completed a clean 12-request, three-placement physical campaign with exact token equality. This checkpoint uses the cloud memory snapshot in `artifacts/physical/memory-v1/cloud_memory.json` and the edge measurements in `artifacts/physical/memory-v1/edge_memory.json`. Both are ignored by Git and reference the raw campaign reports by SHA-256.

| Placement | Measured peak CUDA reserved | Reported card VRAM | Classification |
| --- | ---: | ---: | --- |
| Cloud full model (`cloud/raw_image`) | 4,664 MiB | 8,192 MiB | Feasible in measured state |
| Edge vision-only model for projected split | 942 MiB | 4,096 MiB | Feasible in measured state |
| Edge full model (`edge/full`) | 4,238 MiB | 4,096 MiB | Conditionally feasible |

A provisional 10% margin yields 5,130 MiB for the cloud full model, 1,036 MiB for isolated edge vision, and 4,662 MiB for the edge full model. The first two fit their reported capacities; the edge full model does not. Nonetheless, the edge full model completed the smoke request and all 12 requests in the clean campaign. This is evidence of operation in the observed state, not a safe guarantee under competing GPU load.

An idle Windows `nvidia-smi` snapshot on the edge reported 278 MiB used and 3,685 MiB free (driver 610.62). The isolated vision peak reserved plus 10% margin fits this idle headroom; the full-model peak does not. The reported idle free plus used values do not exactly sum to total VRAM, so use the snapshot as an observed headroom value rather than an accounting identity.

The Phase-1 campaign keeps the full edge model resident while testing all three choices. Its projected-split rows therefore show about 4,238 MiB peak reserved on the edge even though the isolated vision-only smoke reserved 942 MiB. Scheduler memory feasibility depends on residency strategy as well as the selected policy. Before performance profiling, measure edge available VRAM under intended concurrent load and decide whether the scheduler may keep the full edge model resident or needs an explicitly documented lower-memory placement. Do not silently change precision or quantize the frozen model.

## Cloud GPU memory

The 2026-11-02 physical Phase-1 campaign ran the cloud model on the NVIDIA RTX 2070 SUPER (8,192 MiB VRAM). The frozen report contains 12 requests per cloud-backed path at model fingerprint `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`.

| Placement | Peak allocated | Peak reserved | Max server RSS after request |
| --- | ---: | ---: | ---: |
| `cloud/raw_image` | 4,550.25 MiB | 4,664 MiB | 1,137.25 MiB |
| `distributed/projected/synchronous` cloud text | 4,550.26 MiB | 4,664 MiB | 1,137.28 MiB |

With the GPU server resident, Windows `nvidia-smi` reported 4,293 MiB used and 3,697 MiB free of 8,192 MiB total, driver 610.60. This is a separate snapshot from the request peaks. The raw machine-readable summary is `artifacts/physical/memory-v1/cloud_memory.json`, ignored by Git, with a SHA-256 reference to the physical Phase-1 report.

Both cloud-backed placements are **feasible in the measured state**. A provisional 10% margin on the 4,664 MiB reserved peak is 5,130.4 MiB, below the card's 8,192 MiB capacity. This does not establish feasibility under unrelated concurrent GPU load. Edge memory and full three-placement feasibility remain for the edge GPU validation phase.

## Edge full-model swap residency (15 November 2026)

The v1 conclusion above, that the full BF16 edge model does not fit the RTX 3050, still holds for
**resident** weights: the fixed footprint is about 3.9 GiB (vision 856 MiB, text 2,705 MiB,
preallocated KV cache 384 MiB) against roughly 3,300 MiB free after Windows and the CUDA context.
A diagnostic study (`scripts/model/edge_memory_study.py`) confirmed through Windows GPU counters
that resident edge/full spills about 780 MiB into WDDM shared memory, which slows the unchanged
vision encoder from about 0.59 s to 10–12 s.

The adapter now supports `weight_residency="swap"` (CLI: `--edge-residency swap` on `benchmark`
and `edge-infer`). Stage weights stay in pinned host memory and only the executing stage is copied
to the GPU, with PyTorch `expandable_segments` enabled. On all 100 physical VQA requests it ran
100/100 with a 1.51 s median (p95 2.06 s), 3,164 MiB peak reserved and no spill; output token IDs
were identical to all 600 checked G2 edge/full trials. Margin is about 110 MiB, and a competing
GPU process holding 1 GiB raised the median to about 3.7 s. G2 and Predictor V2 r1 edge/full
measurements describe the resident runtime and do not represent swap residency.
