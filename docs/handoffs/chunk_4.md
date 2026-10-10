# Chunk 4 handoff

## Status

Chunk 4 — reliable transport and all-cloud policy — is complete for its
localhost completion gate. The real 32-token cloud output exactly matched the
local monolithic reference, and 100 sequential real-TCP fake requests completed
without deadlock or material RSS growth. Physical-host deployment remains an
external follow-up because no cloud machine has been supplied.

## Implemented

- bounded, versioned, length-prefixed JSON/binary framing with SHA-256 payload
  checksums;
- strict health, raw-image inference, tensor, response, metric, and error
  schemas;
- complete model-contract fingerprinting and compatibility rejection;
- sequential cloud request processor with temporary-file cleanup and structured
  error containment;
- server readiness, socket timeouts, statistics, and graceful shutdown;
- all-cloud execution policy with health checks, exact byte accounting, client
  E2E latency, and returned server component/memory metrics;
- fake and real localhost experiment configurations;
- `cloud-server` and `cloud-infer` commands with standard result artifacts;
- default stderr progress and optional `--verbose` diagnostics for long-running
  inference and verification commands;
- real-TCP 100-request stability and real-model equivalence harnesses; and
- localhost and physical-host deployment instructions.

## Verification

```bash
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/pytest --cov -q

.venv/bin/python scripts/verify_cloud_stability.py \
  --config configs/experiments/chunk4_fake_cloud.yaml \
  --image tests/fixtures/smoke_image.txt \
  --requests 100 \
  --output artifacts/chunk4/cloud_stability.json

.venv-model/bin/python scripts/verify_cloud_equivalence.py \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache \
  --offline \
  --device cpu \
  --image tests/fixtures/moondream_smoke.ppm \
  --output artifacts/chunk4/cloud_equivalence.json
```

Final CPU-only gates after the progress-reporting enhancement: 84 tests passed,
strict mypy and Ruff passed, and coverage was 83.93%. Both real-TCP harnesses
completed successfully.

## Next entry point

Chunk 5 should reuse the same framing, server lifecycle, contract fingerprint,
and error behavior. Add a distributed processor for `tensor_request`, preserve
the BF16 `[729, 2048]` payload bit-for-bit, and prove that the server does not
run vision before adding overlap. Physical-host Chunk 4 validation can proceed
independently once the cloud inventory and reachable address are available.
