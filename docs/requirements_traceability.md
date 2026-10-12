# Baseline requirements traceability

Status meanings:

- **Pending:** not part of the completed work yet.
- **Foundation:** supporting contract or infrastructure exists, but real-model
  evidence is still required.
- **Satisfied:** implemented and supported by baseline experiment evidence.

Statuses below include the Chunk 5 same-host distributed evidence and Chunk 6
runner/fixture evidence. Physical-host and public VQA evidence remain pending.

| ID | Requirement | Status | Current evidence / next evidence |
| --- | --- | --- | --- |
| R1 | Identical pretrained VLM in all configurations | Foundation | [Pinned model](../configs/model/moondream2.yaml), [Chunk 5 evidence](../artifacts/chunk5/distributed_variants_equivalence.json); physical-host repeat pending. |
| R2 | Identical preprocessing | Foundation | [Shared Moondream adapter](../src/adaptive_vlm/models/moondream.py), [Chunk 5 evidence](../artifacts/chunk5/distributed_variants_equivalence.json); physical-host repeat pending. |
| R3 | Identical generation settings | Foundation | [Model fingerprint contract](../src/adaptive_vlm/config.py), [benchmark row hashes](../src/adaptive_vlm/benchmark/runner.py); physical-host repeat pending. |
| R4 | Attempt unmodified all-edge execution | Satisfied | [Chunk 2 probe report](reports/chunk_2_feasibility.md), [edge config](../configs/hardware/edge_rtx3050.yaml); GPU OOM and CPU fallback recorded. |
| R5 | All-cloud transmits input and runs full model remotely | Foundation | [Cloud policy](../src/adaptive_vlm/execution/cloud.py), [Chunk 4 equivalence](../artifacts/chunk4/cloud_equivalence.json); physical cloud pending. |
| R6 | Distributed vision encoder and projector run at edge | Foundation | [Split policy](../src/adaptive_vlm/execution/distributed.py), [Chunk 5 evidence](../artifacts/chunk5/distributed_variants_equivalence.json); physical cloud pending. |
| R7 | Distributed LLM runs in cloud | Foundation | [Text-only server](../src/adaptive_vlm/cli.py), [Chunk 5 evidence](../artifacts/chunk5/distributed_variants_equivalence.json); physical cloud pending. |
| R8 | Intermediate visual features transmitted correctly | Foundation | [Tensor tests](../tests/test_tensor_payload.py), [Chunk 5 evidence](../artifacts/chunk5/distributed_variants_equivalence.json); physical cloud pending. |
| R9 | Output equivalence verified | Foundation | [Chunk 3-5 same-host evidence](../artifacts/chunk5/distributed_variants_equivalence.json), [all-pair comparison tests](../tests/test_benchmark_compare.py); full real manifest and physical hosts pending. |
| R10 | Component latency recorded | Foundation | [Moondream adapter](../src/adaptive_vlm/models/moondream.py), [Chunk 5 report](reports/chunk_5_distributed.md); physical cloud pending. |
| R11 | End-to-end latency recorded | Foundation | [Runner](../src/adaptive_vlm/benchmark/runner.py), [cloud policy](../src/adaptive_vlm/execution/cloud.py); physical cloud pending. |
| R12 | Communication volume recorded | Satisfied | [Wire accounting](../src/adaptive_vlm/transport/client.py), [Chunk 4 report](reports/chunk_4_cloud.md). |
| R13 | Edge/cloud memory recorded | Foundation | [Model adapter](../src/adaptive_vlm/models/moondream.py), [Chunk 5 report](reports/chunk_5_distributed.md); physical cloud pending. |
| R14 | Warm-up precedes measurement | Satisfied | [Runner](../src/adaptive_vlm/benchmark/runner.py) excludes warm-ups; [tests](../tests/test_benchmark_runner.py). |
| R15 | Reproducible config files | Foundation | [Strict config](../src/adaptive_vlm/config.py), [frozen manifest](../src/adaptive_vlm/data/manifest.py), [resume tests](../tests/test_benchmark_runner.py); public VQA and physical hardware pending. |
| R16 | Results stored automatically | Satisfied | [Artifact writer](../src/adaptive_vlm/artifacts.py), [runner](../src/adaptive_vlm/benchmark/runner.py), [OOM/resume tests](../tests/test_benchmark_runner.py). |
| R17 | Hardware/software/version information logged | Foundation | [Environment collector](../src/adaptive_vlm/environment.py), [edge inventory](../configs/hardware/edge_rtx3050.yaml); physical cloud pending. |
| R18 | Random seeds controlled where applicable | Satisfied | [Seed utility](../src/adaptive_vlm/determinism.py), [schedule tests](../tests/test_benchmark_runner.py), [VQA importer tests](../tests/test_vqa_import.py). |

Update this table at every chunk handoff. A requirement becomes **Satisfied**
only when its real-model or experimental evidence is retained and reproducible.
