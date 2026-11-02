# Cloud GPU memory checkpoint

The 2026-11-02 physical Phase-1 campaign ran the cloud model on the NVIDIA RTX 2070 SUPER (8,192 MiB VRAM). The frozen report contains 12 requests per cloud-backed path at model fingerprint `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`.

| Placement | Peak allocated | Peak reserved | Max server RSS after request |
| --- | ---: | ---: | ---: |
| `cloud/raw_image` | 4,550.25 MiB | 4,664 MiB | 1,137.25 MiB |
| `distributed/projected/synchronous` cloud text | 4,550.26 MiB | 4,664 MiB | 1,137.28 MiB |

With the GPU server resident, Windows `nvidia-smi` reported 4,293 MiB used and 3,697 MiB free of 8,192 MiB total, driver 610.60. This is a separate snapshot from the request peaks. The raw machine-readable summary is `artifacts/physical/memory-v1/cloud_memory.json`, ignored by Git, with a SHA-256 reference to the physical Phase-1 report.

Both cloud-backed placements are **feasible in the measured state**. A provisional 10% margin on the 4,664 MiB reserved peak is 5,130.4 MiB, below the card's 8,192 MiB capacity. This does not establish feasibility under unrelated concurrent GPU load. Edge memory and full three-placement feasibility remain for the edge GPU validation phase.
