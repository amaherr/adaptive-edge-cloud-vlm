# Physical memory feasibility checkpoint v1

The pinned BF16 Moondream2 model completed a clean 12-request, three-placement physical campaign with exact token equality. This checkpoint uses the cloud memory snapshot in `artifacts/physical/memory-v1/cloud_memory.json` and the edge measurements in `artifacts/physical/memory-v1/edge_memory.json`. Both are ignored by Git and reference the raw campaign reports by SHA-256.

| Placement | Measured peak CUDA reserved | Reported card VRAM | Classification |
| --- | ---: | ---: | --- |
| Cloud full model (`cloud/raw_image`) | 4,664 MiB | 8,192 MiB | Feasible in measured state |
| Edge vision-only model for projected split | 942 MiB | 4,096 MiB | Feasible in measured state |
| Edge full model (`edge/full`) | 4,238 MiB | 4,096 MiB | Conditionally feasible |

A provisional 10% margin yields 5,130 MiB for the cloud full model, 1,036 MiB for isolated edge vision, and 4,662 MiB for the edge full model. The first two fit their reported capacities; the edge full model does not. Nonetheless, the edge full model completed the smoke request and all 12 requests in the clean campaign. This is evidence of operation in the observed state, not a safe guarantee under competing GPU load.

The Phase-1 campaign keeps the full edge model resident while testing all three choices. Its projected-split rows therefore show about 4,238 MiB peak reserved on the edge even though the isolated vision-only smoke reserved 942 MiB. Scheduler memory feasibility depends on residency strategy as well as the selected policy. Before performance profiling, measure edge available VRAM under intended concurrent load and decide whether the scheduler may keep the full edge model resident or needs an explicitly documented lower-memory placement. Do not silently change precision or quantize the frozen model.
