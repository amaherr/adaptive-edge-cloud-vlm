# Chunk 3 handoff

## Status

Chunk 3 — model decomposition and local all-edge policy — is complete. The
completion gate passed: on CPU BF16, decomposed greedy inference produced the
same projected tensor, 32 native token IDs, and text as monolithic inference.

## Implemented

- strict `729 × 2048` projected-feature contract with byte accounting;
- synchronization-aware component timing;
- lazy, artifact-verifying Moondream2 adapter;
- exact pinned tokenizer replacement;
- monolithic and decomposed native-token execution paths;
- CPU RSS and CUDA peak allocation/reservation metrics;
- shared `LocalExecutionPolicy` integration through `edge-infer`;
- atomic run manifest and request artifacts containing execution placement;
- repeatability/equivalence harness; and
- model boundary and feasibility documentation.

Raw model caches and experiment JSON remain in ignored `.model-cache/` and
`artifacts/chunk3/` directories. The tracked report preserves the decision
evidence.

## Verification

```bash
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest --cov=adaptive_vlm --cov-report=term-missing

.venv-model/bin/python scripts/verify_local_decomposition.py \
  --config configs/experiments/baseline_edge.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache \
  --offline \
  --image tests/fixtures/moondream_smoke.ppm \
  --repetitions 1 \
  --output artifacts/chunk3/equivalence_cpu_bf16_32_tokens.json
```

Final CPU-only gates before documentation: 50 tests passed, strict mypy and
Ruff passed, and coverage was 83.32%. The real adapter is excluded from the
CPU-only coverage calculation because its locked PyTorch/Transformers stack is
tested by the separate model integration commands.

## Key evidence

- full 32-token CPU equivalence: exact tensor, token IDs, and text;
- two repeated eight-token decomposed runs: exact outputs and 0.035 MiB final
  RSS range;
- CPU decomposed end to end: 40,076.7 ms for the measured 32-token fixture;
- GPU vision plus CPU text end to end: 34,031.7 ms;
- GPU vision total: 676.7 ms, including 551.4 ms encoder and 23.1 ms projector;
- projected transfer: 2.49 ms for 2,985,984 bytes;
- GPU peak: 913.8 MiB allocated, 942 MiB reserved.

## Known limitations

- measurements use one synthetic fixture and are not benchmark statistics;
- GPU-vision and CPU-reference output sequences diverged after 19 common
  tokens, so CPU/CPU remains the correctness-preserving default;
- the adapter intentionally depends on private methods of an immutable remote
  code revision; any model upgrade requires a new audit and equivalence gate;
- cloud hardware and real remote policies remain pending;
- the correctness dataset manifest remains pending.

## Next entry point

Begin Chunk 4 with a bounded, versioned, length-prefixed transport protocol and
localhost fragmentation/error tests. Then implement all-cloud request handling
with the same approved adapter, artifact fingerprint checks, preprocessing, and
generation settings. Do not implement the distributed tensor path until the
transport and all-cloud policy pass locally.
