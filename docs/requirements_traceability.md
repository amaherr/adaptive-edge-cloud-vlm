# Baseline requirements traceability

Status meanings:

- **Pending:** not part of the completed work yet.
- **Foundation:** supporting contract or infrastructure exists, but real-model
  evidence is still required.
- **Satisfied:** implemented and supported by baseline experiment evidence.

Statuses below include the completed Chunk 3 local decomposition and all-edge
evidence. Requirements spanning remote policies remain at foundation level
until those policies reuse the adapter and are compared in the campaign.

| ID | Requirement | Status | Current evidence / next evidence |
| --- | --- | --- | --- |
| R1 | Identical pretrained VLM in all configurations | Foundation | Moondream commit and artifact are pinned; prove reuse in all real policies. |
| R2 | Identical preprocessing | Foundation | Monolithic and decomposed local paths share the pinned Pillow adapter and produced an identical projected tensor; remote reuse remains pending. |
| R3 | Identical generation settings | Foundation | The adapter enforces BF16, greedy decoding, base variant, and the pinned query template; remote reuse remains pending. |
| R4 | Attempt unmodified all-edge execution | Satisfied | Two GPU loads recorded infeasibility; the identical BF16 artifact completed repeated edge-CPU inference on three fixtures. |
| R5 | All-cloud transmits input and runs full model remotely | Pending | Chunk 4. |
| R6 | Distributed vision encoder and projector run at edge | Pending | Chunk 5. |
| R7 | Distributed LLM runs in cloud | Pending | Chunk 5. |
| R8 | Intermediate visual features transmitted correctly | Pending | Chunk 5 tensor round-trip tests. |
| R9 | Output equivalence verified | Foundation | CPU monolithic and decomposed paths matched all 32 configured token IDs exactly; cross-device and cross-policy campaign remains pending. |
| R10 | Component latency recorded | Foundation | Synchronized all-edge vision, projection, transfer, image-prefill, and generation timings are recorded; remote components remain pending. |
| R11 | End-to-end latency recorded | Foundation | Real all-edge end-to-end measurements are recorded automatically; remote policies remain pending. |
| R12 | Communication volume recorded | Pending | Protocol byte accounting in Chunks 4 and 5. |
| R13 | Edge/cloud memory recorded | Foundation | All-edge CPU RSS and GPU-vision peak allocation/reservation are recorded; cloud memory remains pending. |
| R14 | Warm-up precedes measurement | Foundation | Chunk 2 probe warmed each fixture before timing; policy benchmark runner is Chunk 6. |
| R15 | Reproducible config files | Foundation | `configs/`, strict loader, CLI validation, and `tests/test_config.py`. |
| R16 | Results stored automatically | Foundation | `artifacts.py`, fake smoke artifacts, and artifact tests. |
| R17 | Hardware/software/version information logged | Foundation | WSL/CUDA/PyTorch inventory is verified and edge hardware config updated; cloud remains pending. |
| R18 | Random seeds controlled where applicable | Foundation | `seed_everything`, config seed, fake CLI integration, and seed tests. |

Update this table at every chunk handoff. A requirement becomes **Satisfied**
only when its real-model or experimental evidence is retained and reproducible.
