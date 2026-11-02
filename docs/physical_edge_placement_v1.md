# Physical placement checkpoint v1

The pinned Moondream2 contract uses BF16 and greedy generation. The cloud model runs on its RTX 2070 SUPER through `scripts/run_physical_cloud_gpu.py`. The edge RTX 3050 has 4,096 MiB reported VRAM. The first GPU placement smoke tests used `tests/fixtures/moondream_smoke.ppm` and the prompt `Describe this image.`.

| Scheduler choice | Vision placement | Text placement | Smoke result |
| --- | --- | --- | --- |
| `cloud/raw_image` | Cloud `cuda:0` | Cloud `cuda:0` | Pass |
| `distributed/projected/synchronous` | Edge `cuda:0` | Cloud `cuda:0` | Pass; exact cloud token sequence |
| `edge/full` | Edge `cuda:0` | Edge `cuda:0` | Pass; exact cloud token sequence |

The edge GPU split smoke used a BF16 `[729, 2048]` projection of 2,985,984 bytes. Edge vision took 473.5 ms and peaked at 913.8 MiB allocated, 942 MiB reserved. The edge full GPU smoke took 9,302 ms and peaked at 4,128.2 MiB allocated, 4,238 MiB reserved. Both raw outputs are archived under `artifacts/phase1/edge-gpu-validation-v1/` and ignored by Git.

The full GPU peak exceeds the card's reported 4,096 MiB VRAM, despite that smoke request completing. Classify `edge/full` GPU as **conditionally feasible** pending memory-pressure testing. Do not infer stable 4 GB fit from one success. These timings are diagnostic single requests, not comparable performance estimates.

## Frozen 12-request GPU campaign

The edge ran `scripts/verify_phase1.py` with both edge devices set to `cuda:0`. All 36 inferences completed and all 12 request triples returned exactly equal token IDs and text (`complete_and_equal: true`). The new edge GPU output matches the earlier cloud GPU output for all 12 requests; it differs from the earlier edge CPU output only on `correctness-001`. The raw report is archived at `artifacts/phase1/edge-gpu-validation-v1/phase1_smoke.json`, ignored by Git. Its model fingerprint and dataset manifest SHA-256 match the first physical campaign.

The first report recorded edge Git commit `2415685` with `dirty: true`, so the edge repeated the campaign after `git status --short` returned empty. The second report, archived at `artifacts/phase1/edge-gpu-validation-v1/phase1_smoke_clean.json`, records the same commit with `dirty: false`, 36 completed inferences, and exact equality for all 12 triples. It uses the same model fingerprint and dataset manifest SHA-256. This closes the strict token-equivalence gate for the specified all-GPU placement on the frozen synthetic manifest.

Peak CUDA reserved for the resident full edge model was 4,238 MiB in each request of the first campaign. The split rows in that campaign also report roughly 4,124 MiB peak allocated because the runner kept the full edge model resident for all three policies. The separate vision-only smoke measured 913.8 MiB peak allocated and is the relevant isolated vision footprint. Memory feasibility under additional load remains conditional.
