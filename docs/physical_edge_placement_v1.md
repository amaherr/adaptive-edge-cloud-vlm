# Physical placement checkpoint v1

The pinned Moondream2 contract uses BF16 and greedy generation. The cloud model runs on its RTX 2070 SUPER through `scripts/run_physical_cloud_gpu.py`. The edge RTX 3050 has 4,096 MiB reported VRAM. The first GPU placement smoke tests used `tests/fixtures/moondream_smoke.ppm` and the prompt `Describe this image.`.

| Scheduler choice | Vision placement | Text placement | Smoke result |
| --- | --- | --- | --- |
| `cloud/raw_image` | Cloud `cuda:0` | Cloud `cuda:0` | Pass |
| `distributed/projected/synchronous` | Edge `cuda:0` | Cloud `cuda:0` | Pass; exact cloud token sequence |
| `edge/full` | Edge `cuda:0` | Edge `cuda:0` | Pass; exact cloud token sequence |

The edge GPU split smoke used a BF16 `[729, 2048]` projection of 2,985,984 bytes. Edge vision took 473.5 ms and peaked at 913.8 MiB allocated, 942 MiB reserved. The edge full GPU smoke took 9,302 ms and peaked at 4,128.2 MiB allocated, 4,238 MiB reserved. Both raw outputs are archived under `artifacts/phase1/edge-gpu-validation-v1/` and ignored by Git.

The full GPU peak exceeds the card's reported 4,096 MiB VRAM, despite that smoke request completing. Classify `edge/full` GPU as **conditionally feasible** pending repeated requests and memory-pressure testing. Do not infer stable 4 GB fit from one success. A frozen 12-request GPU campaign is pending. These timings are diagnostic single requests, not comparable performance estimates.
