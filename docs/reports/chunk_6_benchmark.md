# Chunk 6 benchmark and correctness report

## Implemented

The unified `benchmark --mode edge|cloud|distributed` command selects one of
the existing policies without changing model, request data, generation, or
measurement code. It verifies a frozen request manifest and every source image
hash, seeds and shuffles each measured repetition, records warm-ups separately,
fsyncs one raw JSONL row per attempt, and resumes only an identical plan.
Plans, run manifests, raw rows, and summaries are saved in each run directory.
Every row includes the plan, config, model-contract, dataset, and image hashes.
Failures remain visible by status and are excluded from successful-latency
statistics without disappearing from the attempt count. Summaries include
median/p95/IQR latency, failure rate, wire bytes where applicable, and fixture
quality. `compare-runs` audits every measured request/repetition pair, with
exact token IDs, decoded text, failures, and policy quality scores.

The repository includes a 12-request synthetic correctness manifest and a
100-request synthetic benchmark manifest. An importer can freeze a seeded
subset of the official VQA v2 val2014 questions/answers and hash the required
COCO image bytes when those files are available locally. No public dataset was
downloaded. The synthetic exact-match score is a diagnostic only; the current
VQA consensus normalization is provisional and must not be reported as
official VQA accuracy.

## Evidence so far

| Gate | Observation | State |
| --- | --- | --- |
| G1 projected-tensor split | Chunk 3/5 local decomposition evidence | Passed on prior single fixture |
| G2 tensor serialization | Exact byte/dtype/shape round-trip and network tests | Passed |
| Fake edge/cloud policy comparison | 24/24 measured pairs had identical token IDs and text; no failures | Passed |
| 100-request fixture runner check | 100/100 measured attempts succeeded after 10 warm-ups; raw rows and summary complete | Passed with fake adapter |
| Real all-edge correctness | 12/12 measured requests succeeded after one warm-up; median 38.5 s; fixture exact match 3/9 scored prompts | Passed for all-edge |
| Real all-cloud correctness | 12/12 measured requests succeeded after one warm-up; median 35.1 s; fixture exact match 3/9 scored prompts | Passed on localhost |
| Real edge/cloud comparison | 12/12 measured pairs had identical token IDs and text; no failures | Passed on same-host CPU |
| G3 same-host all three policies | 12/12 measured pairs had identical token IDs and decoded text; no failures or mismatches | Passed on same-host CPU |
| G4 physical hosts | Requires a separate cloud host | Pending |
| Public VQA task quality | Requires local VQA/COCO files and official evaluation | Pending |

Final code verification: Ruff and strict mypy passed; the CPU-safe suite had
122 passed and one sandbox-only socket skip, with 81.60% coverage (above the
80% threshold). The skipped subprocess/ephemeral-port test passed separately
when localhost binding was allowed.

The real policy runs are retained at
`artifacts/chunk6/correctness-edge/` and
`artifacts/chunk6/correctness-cloud/` and
`artifacts/chunk6/correctness-distributed/` (ignored working data). The
complete three-way audit is `artifacts/chunk6/correctness-comparison.json`.
The fake
edge/cloud run and comparison were executed from the same frozen manifest in
temporary directories; they are software-path checks, not model-quality or
latency evidence. The real-model runs use pinned Moondream2 BF16 and greedy
32-token generation on CPU so G3 tests the same device/dtype contract.

| Policy | Measured successes | Median observed E2E | Mean application wire bytes | Fixture exact match |
| --- | ---: | ---: | ---: | ---: |
| All edge | 12/12 | 38.5 s | Not applicable | 3/9 |
| All cloud | 12/12 | 35.1 s | 1,580 bytes | 3/9 |
| Distributed synchronous | 12/12 | 30.9 s | 2,987,580 bytes | 3/9 |

The cloud client sent original image bytes; the distributed client sent
2,985,984 BF16 projected-tensor payload bytes per request plus framing and
response bytes. The policy runs were sequential and unshaped, not paired or
randomized across policy order. Their latency differences are **not** evidence
of a distributed speedup. A
one-repetition correctness pass is not a benchmark latency estimate; the
default 10 warm-ups and 30 repetitions remain for controlled experiments.

## Limits

The current machine has a 4 GiB GPU, and the user confirmed that no separate
cloud host is available yet. The
full-size all-edge GPU attempt was previously recorded infeasible; CPU is used
here to establish deterministic same-device equivalence. Localhost latency
and communication measurements must not be interpreted as physical
edge-to-cloud performance. Actual VQA quality and cross-device agreement must
be measured once the dataset and hardware are supplied.
