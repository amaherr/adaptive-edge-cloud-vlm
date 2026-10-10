# Baseline requirements traceability

Status meanings:

- **Pending:** not part of the completed work yet.
- **Foundation:** supporting contract or infrastructure exists, but real-model
  evidence is still required.
- **Satisfied:** implemented and supported by baseline experiment evidence.

Statuses below include the completed Chunk 4 localhost all-cloud evidence.
Requirements spanning the distributed policy or physical cloud remain at
foundation level until those experiments are available.

| ID | Requirement | Status | Current evidence / next evidence |
| --- | --- | --- | --- |
| R1 | Identical pretrained VLM in all configurations | Foundation | Moondream commit and artifact are pinned; prove reuse in all real policies. |
| R2 | Identical preprocessing | Foundation | Edge and localhost cloud reuse the pinned Pillow adapter; the contract fingerprint rejects a different backend. Distributed and physical-host evidence remain pending. |
| R3 | Identical generation settings | Foundation | Edge and localhost cloud share the adapter and a fingerprint covering precision and every generation setting. Distributed evidence remains pending. |
| R4 | Attempt unmodified all-edge execution | Satisfied | Two GPU loads recorded infeasibility; the identical BF16 artifact completed repeated edge-CPU inference on three fixtures. |
| R5 | All-cloud transmits input and runs full model remotely | Foundation | The client sent original image bytes/prompt and the localhost server ran monolithic Moondream with exact output. Repeat on the physical cloud host. |
| R6 | Distributed vision encoder and projector run at edge | Pending | Chunk 5. |
| R7 | Distributed LLM runs in cloud | Pending | Chunk 5. |
| R8 | Intermediate visual features transmitted correctly | Pending | Chunk 5 tensor round-trip tests. |
| R9 | Output equivalence verified | Foundation | Local decomposition and localhost all-cloud each matched the 32-token monolithic reference exactly; distributed and physical-host comparison remain pending. |
| R10 | Component latency recorded | Foundation | Synchronized edge components and returned localhost cloud image/prefill/generation times are recorded; physical cloud remains pending. |
| R11 | End-to-end latency recorded | Foundation | Edge E2E and client-observed localhost cloud E2E are recorded automatically; physical cloud remains pending. |
| R12 | Communication volume recorded | Satisfied | Framing returns exact request/response wire bytes; real-model evidence recorded 504-byte request and 1,344-byte response. |
| R13 | Edge/cloud memory recorded | Foundation | Edge CUDA/RSS and localhost server model RSS are recorded; physical cloud memory remains pending. |
| R14 | Warm-up precedes measurement | Foundation | Chunk 2 probe warmed each fixture before timing; policy benchmark runner is Chunk 6. |
| R15 | Reproducible config files | Foundation | `configs/`, strict loader, CLI validation, and `tests/test_config.py`. |
| R16 | Results stored automatically | Foundation | `artifacts.py`, fake smoke artifacts, and artifact tests. |
| R17 | Hardware/software/version information logged | Foundation | WSL/CUDA/PyTorch inventory is verified and edge hardware config updated; cloud remains pending. |
| R18 | Random seeds controlled where applicable | Foundation | `seed_everything`, config seed, fake CLI integration, and seed tests. |

Update this table at every chunk handoff. A requirement becomes **Satisfied**
only when its real-model or experimental evidence is retained and reproducible.
