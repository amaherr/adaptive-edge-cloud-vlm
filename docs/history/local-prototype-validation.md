# Local prototype validation (October 2026)

Validation records from the single-host CPU/loopback prototype, before the physical pair existed. They are kept as evidence for the implemented components; later physical results supersede their performance conclusions.

## Verified local baseline

Completed: 2026-10-26 (Africa/Cairo). Scope: G0-local from the [scheduler plan](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/dynamic_scheduler_implementation_plan.md). **G0-local passed; G0-physical remains pending** because a separate cloud PC is not available.

### What was made

- Added `scripts/verification/run_local_correctness_campaign.sh`, a fresh-directory, offline, one-sweep runner for edge, cloud and synchronous distributed CPU policies. It captures input hashes, environment, logs, raw benchmark directories and `compare-runs` output, then asserts 12 successful measured requests per policy and exact equality for all 12 paired requests. It refuses to overwrite a previous run and stops its local server on exit.
- Increased its server startup allowance to 600 seconds and guarded the readiness-log check against a process-start race. The first attempt's 180-second limit was too short for a cold model load.
- Re-ran Ruff, strict mypy and the full CPU test suite with loopback socket access; the TCP subprocess test ran instead of being skipped.
- Repeated the real pinned Moondream2 CPU/BF16 correctness campaign. The complete raw artifacts are local and ignored by Git; archive the directory separately before moving machines.

No scheduler or model behavior was changed in this phase.

### Inputs and provenance

The successful edge, cloud and distributed-retry runs used the clean source commit `ce325d24345084cd9c9213948f8cebd32795eb06`, the shared `configs/experiments/local_campaign.yaml` settings, seed 42, one excluded warm-up and one measured sweep of the 12-request synthetic correctness manifest. Each policy used CPU for all model components and the same pinned Moondream2 revision and greedy 32-token cap. Cloud and distributed servers ran on loopback (`127.0.0.1:5050`) on the same WSL host; no physical network or separate cloud host was involved.

| Frozen input | SHA-256 |
| --- | --- |
| `data/manifests/correctness.jsonl` | `c5bc9df4a010dcefbcbb7fd60c9dd1e47dc0e0cf798acff579a1a004006ce8d1` |
| `configs/experiments/local_campaign.yaml` | `0994e0fbbd05c8e7fd7e4c5b82413d5a0531ba39805d201b8422e32a374d64ff` |
| `configs/model/moondream2.yaml` | `831f7255a11c5be66816e821497e2f54cb1c114ca34bfe8e043c5f95174f2d09` |
| `configs/model/moondream2_artifact.yaml` | `369640ff5a46c7f11c6b3f20a22f780d9ae4501e34db80fe2eaa11263df3a650` |
| `requirements/model-cu128.lock` | `96d07ba6bb6310b29d259c64652de18f4c5f72f2f0629a17844d09ffacedffbd` |

The model contract fingerprint recorded by `compare-runs` is `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`. The local environment record reports WSL2, an i7-11800H, 7.637 GiB RAM, and a 4 GiB RTX 3050; inference in this campaign remained CPU-only. See `environment.json`, `inputs.sha256`, and each policy's `run_manifest.json` in the [local artifact directory](../../artifacts/phase0/2026-10-25/local-campaign-1/). The local [compressed archive](../../artifacts/phase0/2026-10-25/local-campaign-1.tar.gz) has SHA-256 `71050d767dbb2de58ffdeb169e8c75e005d7f0bcccdf42edfafd1ce18f977b95`. Both paths are ignored by Git; copy the archive to durable storage before moving machines.

### Gate evidence and interpretation

| G0-local check | Result |
| --- | --- |
| Ruff | Passed, `All checks passed!` |
| Strict mypy | Passed, 33 source files |
| Pytest and coverage | 123 passed, 0 skipped; 81.60% coverage, above the 80% gate |
| Edge measured requests | 12/12 `ok`, 1 excluded warm-up |
| Cloud measured requests | 12/12 `ok`, 1 excluded warm-up |
| Distributed measured requests | 12/12 `ok`, 1 excluded warm-up, synchronous pipeline |
| `compare-runs` | `complete_and_equal=true`; 12/12 token matches, 0 text mismatches, 0 failures |

The comparison uses `edge/`, `cloud/`, and **`distributed-retry/`**. All three report the same provisional synthetic fixture score, 3 correct among 9 scored prompts. That score is a correctness diagnostic, not official VQA accuracy.

Observed medians were 51.8 s edge, 27.1 s cloud and 28.3 s distributed; mean application wire bytes were not applicable for edge, 1,579 for cloud and 2,987,582 for distributed (rounded). These are one-sweep, sequential localhost CPU observations. They must **not** be used as controlled latency estimates or evidence that one placement is faster under a real network.

The first scripted attempt completed `edge/` but timed out while loading the cloud server. Its `cloud-server.log` records the startup timeout; no cloud requests were sent by that attempt. A later `distributed/` sweep was interrupted by the tool-session restart: 10/12 measured requests completed and matched edge token IDs, then one `PeerDisconnected` and one `ConnectionRefusedError` were recorded. These failed records were kept. The subsequent fresh `distributed-retry/` run passed 12/12; the interrupted `distributed/` directory is **excluded** from the passing comparison. No model-correctness conclusion is drawn from the connection failures.

### How to reproduce and view the results

From the repository root, with `.venv`, `.venv-model` and the verified local `.model-cache` installed, run the updated script in an environment that permits loopback TCP binding:

```bash
scripts/verification/run_local_correctness_campaign.sh artifacts/phase0/NEW_LOCAL_RUN
```

Use a **new** path each time; the script deliberately refuses an existing directory. It requires the pinned model cache and `--offline` execution. A full one-sweep run takes minutes on CPU. A failure leaves partial artifacts for diagnosis; do not treat a benchmark command's exit code alone as a passing gate.

To inspect this campaign's passing comparison and per-policy summaries:

```bash
.venv/bin/python -m json.tool artifacts/phase0/2026-10-25/local-campaign-1/comparison.json
.venv/bin/python -m json.tool artifacts/phase0/2026-10-25/local-campaign-1/edge/summary.json
.venv/bin/python -m json.tool artifacts/phase0/2026-10-25/local-campaign-1/cloud/summary.json
.venv/bin/python -m json.tool artifacts/phase0/2026-10-25/local-campaign-1/distributed-retry/summary.json
```

The per-request rows are in each run's `requests.jsonl`; `plan.json` fixes its schedule and hashes. `ruff.log`, `mypy.log` and `pytest.log` preserve the check output. Recalculate the comparison at any time with:

```bash
.venv/bin/adaptive-vlm compare-runs \
  artifacts/phase0/2026-10-25/local-campaign-1/edge \
  artifacts/phase0/2026-10-25/local-campaign-1/cloud \
  artifacts/phase0/2026-10-25/local-campaign-1/distributed-retry
```

### Remaining Phase 0 gate

G0-physical is pending. Once a separate cloud PC is available, record its verified hardware/software inventory in `configs/hardware/`, place the same pinned artifact and dependency lock on both hosts, configure a protected reachable TCP endpoint, and repeat the same frozen manifest with CPU/CPU numerical placement first. Archive both environment records, server readiness metadata, network topology and independently measured bandwidth/RTT. Require all 12 pairs present, zero measured failures and `complete_and_equal=true` before claiming physical-host correctness. Controlled latency comparisons belong to later profiling/evaluation phases, not this local gate.

## Executable configuration space and resident dispatcher

Completed: 2026-10-26 (Africa/Cairo). **G1 passed for the local CPU/loopback deployment.** A separate cloud PC is still unavailable, so physical-host concurrency and memory behavior remain unverified. This handoff follows the [implementation plan](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/dynamic_scheduler_implementation_plan.md) and the [Phase 0 baseline](local-prototype-validation.md#verified-local-baseline).

### What changed

- Added stable configuration IDs and typed `SystemState`, `TaskState`, capability, and decision records in `src/adaptive_vlm/scheduler/`. A dispatcher routes one request to exactly one already loaded `ExecutionPolicy` and logs the full availability snapshot. Phase 1 selection is **manual**; predicted or automatic selection belongs to later phases.
- Added a capability request/response to the bounded TCP protocol. The response identifies the model contract and supported request paths. An adaptive client excludes a remote configuration if the endpoint is unreachable, its fingerprint differs, or its required path is absent.
- Added `adaptive-server`, which loads one pinned full Moondream2 model and accepts raw-image and projected-tensor requests on the same endpoint. Existing `cloud-server`, `distributed-server`, and fixed-policy commands remain available.
- Added `adaptive-infer` for one manually selected configuration. Its `--edge-components full` mode loads one full edge adapter and shares it between edge inference and edge vision encoding, so switching those paths does not reload weights. `vision` registers cloud and split only; `none` registers cloud only. `prompt_first` is advertised by the server but is not a scheduler candidate until its overlap cost is profiled.
- Added `scripts/verification/verify_resident_configurations.py`, which keeps one client model and one dual-path server resident while executing all three configurations over either one request or a frozen manifest. It saves each decision, result, elapsed time, client RSS reading, model fingerprint, environment, and equality verdict.
- Made memory-budget checks require a supplied **peak** requirement. The post-load RSS was about 1.07 GiB in this run, while request-time RSS reached 4.52 GiB; using the startup value as a safety bound would be wrong. Without a measured requirement, an explicit budget rejects the candidate with `edge_memory_requirement_unknown`.

Implementation commits: `1cff2e0` (contracts, protocol, server, CLI and tests), `e8cfa78` (resident smoke runner), and `c0a4d20` (peak-memory preflight and manifest-wide verification). The real campaign's environment record reports clean commit `c0a4d20b8efdf819e02af3656369a4d90d1023bb`.

### Gate evidence

The local dual-path server advertised `raw_image`, `projected_synchronous`, and `projected_prompt_first` with model contract fingerprint `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`. The client used CPU vision and CPU text with the pinned Moondream2 artifact, the unchanged 32-token greedy contract, and the [same 12-request manifest](../../data/manifests/correctness.jsonl) as Phase 0 (SHA-256 `c5bc9df4a010dcefbcbb7fd60c9dd1e47dc0e0cf798acff579a1a004006ce8d1`). The server and client each loaded once and stayed resident for the campaign.

| G1-local check | Result |
| --- | --- |
| Registered configurations | `edge/full`, `cloud/raw_image`, `distributed/projected/synchronous` available together |
| Frozen requests | 12/12 completed through each configuration; 36 result rows |
| Output equivalence | 12/12 request triples have identical token IDs and decoded text |
| Baseline equivalence | 12/12 match the Phase 0 edge token sequence |
| Routing and exclusion | Tests cover exact policy dispatch, missing remote path, wrong remote fingerprint, unreachable endpoint, and insufficient/unknown memory |
| Static checks | Ruff and strict mypy passed |
| Full test suite | 131 passed, 0 skipped with loopback access; 82.31% coverage |

The [campaign report](../../artifacts/phase1/2026-10-26/campaign-1/phase1_smoke.json) records `complete_and_equal=true` and `matches_baseline_edge=true`. Its `rows` contain 12 entries for each configuration, including all decisions and output token IDs. The one-request [smoke report](../../artifacts/phase1/2026-10-26/smoke-1/phase1_smoke.json) also passed. The local [results archive](../../artifacts/phase1/2026-10-26/campaign-1.tar.gz) has SHA-256 `6523f60d9e23c8016ad075ae5d550bb27c0c71e1523663422e663472c97abf3b`. All `artifacts/` paths are ignored by Git; copy the archive to durable storage.

#### Memory and limits of this evidence

The client measured about 1,072 MiB RSS immediately after loading and a maximum of 4,522 MiB after requests. Server-side request metrics reported a maximum of about 4,869 MiB RSS. Both processes shared a WSL host with 7.637 GiB reported RAM, so their combined residency can cause contention or swapping. The distributed choice in `full` mode executes only vision on the edge but retains the full edge model in memory for immediate edge-only availability. Use `--edge-components vision` when the full edge model cannot remain resident; then `edge/full` is explicitly unavailable.

The observed request times are sequential localhost CPU correctness observations. They do not establish the best policy, network crossovers, or physical-host performance. The memory numbers are observations for this workload, not a validated universal peak: Phase 2 must profile per-task peaks and add headroom before using them as production constraints. Manual bandwidth and RTT values are recorded by Phase 1 but do not influence the chosen configuration yet.

### How to run it

From the repository root, with the pinned `.venv-model` and verified `.model-cache`, start the server in terminal A:

```bash
.venv-model/bin/adaptive-vlm adaptive-server \
  --config configs/experiments/local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
```

Wait for its readiness JSON to list both `raw_image` and `projected_synchronous`. In terminal B, select one configuration manually:

```bash
.venv-model/bin/adaptive-vlm adaptive-infer \
  --config configs/experiments/local_campaign.yaml \
  --configuration-id distributed/projected/synchronous \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --edge-components full \
  --vision-device cpu --text-device cpu \
  --image tests/fixtures/moondream_smoke.ppm \
  --prompt "Describe this image." --output-dir artifacts/manual-adaptive
```

The command prints `decision` and `request` JSON. With `--output-dir`, it writes `run_manifest.json`, `decision.json`, and `request.json` under a fresh run-ID directory. `--configuration-id` may also be `edge/full` or `cloud/raw_image`. For cloud-only use, `--edge-components none` avoids loading model weights on the client; for cloud plus projected split, use `vision`. When passing `--edge-memory-budget-mib`, also pass a conservative profile-derived `--edge-required-peak-mib`; the budget applies to the memory footprint of the currently resident client process.

To repeat the full G1-local check without reloading between requests, keep terminal A running and execute:

```bash
.venv-model/bin/python scripts/verification/verify_resident_configurations.py \
  --config configs/experiments/local_campaign.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --manifest data/manifests/correctness.jsonl \
  --baseline-edge-requests \
    artifacts/phase0/2026-10-25/local-campaign-1/edge/requests.jsonl \
  --output-dir artifacts/phase1/NEW_CAMPAIGN
```

Use a new output path. Omit `--baseline-edge-requests` if the local Phase 0 artifacts are unavailable; the script will still compare all three Phase 1 outputs with each other. Inspect the saved report and test logs with:

```bash
.venv/bin/python -m json.tool \
  artifacts/phase1/2026-10-26/campaign-1/phase1_smoke.json
tail -5 artifacts/phase1/2026-10-26/campaign-1/pytest.log
```

### Next handoff

Phase 2 should define timing boundaries and a profile schema, then measure compute, serialization, transfer, server work, and conservative memory peaks per configuration/task/network bucket. The automatic predictor and oracle are later gates. The separate physical cloud host is still needed for physical deployment claims.

## Stage instrumentation and local profile

Date: 2026-10-26 (Africa/Cairo). Phase 2 follows the [implementation plan](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/dynamic_scheduler_implementation_plan.md) and [Phase 1 handoff](local-prototype-validation.md#executable-configuration-space-and-resident-dispatcher). This document records the executable local slice of G2 and separates it from the still pending physical-host and controlled-network slice.

### What changed

- The pinned Moondream adapter now records exact prompt token count, prompt preparation, prompt prefill, decode, decode steps and decoded image dimensions in each edge/cloud path. The generation token sequence and 32-token greedy contract remain unchanged.
- The TCP client records connect, frame-encode/send and response-wait/decode intervals. Cloud inference records image file-read and full exchange time. Both remote policies already record exact request/response frame bytes; the projected split also records payload serialization and server tensor deserialization. The Phase 1 manual dispatcher now records its selection overhead for future adaptive runs.
- The benchmark now timestamps every attempt. `profile-runs` validates complete matching schedules, raw trial provenance, model/config/manifest and image hashes, exact prompt/output counts, client/server request bytes, and payload bounds. It emits versioned rows containing measured successes, warmups, failures, stage residuals, null reasons, memory readings, output hashes and raw metrics. Its summary reports sample coverage, failure rates and cross-configuration equality.
- The [profile schema](../design/profile-format.md) defines every timing boundary and limitation. The [split inspection](../design/split-boundary.md) identifies possible vision/text boundaries in the pinned graph. None passed the new tensor-contract and end-to-end equivalence gate, so the executable registry remains the three Phase 1 configurations.

Implementation was checkpointed regularly from `624d3f8` through `187ca7a` (see `git log --oneline 624d3f8^..187ca7a`). The run artifacts preserve their own Git commit and dirty flag; profile rows link their run manifests and the pinned model/dataset hashes.

### Local campaign evidence

The [profile summary](../../artifacts/phase2/2026-10-26/local-1/profile/summary.json) reports `local_gate_passed=true` for the one declared 4×4 fixture/short-prompt/loopback bucket and `controlled_network_gate_passed=false`. There are 39 rows: three excluded warmups and 36 measured results. Each configuration has 12/12 successful measured trials across 12 distinct request IDs, zero failures, and no negative stage or transport residual. Every successful three-way trial has matching prompt-token count and output-token hash. The independent [comparison](../../artifacts/phase2/2026-10-26/local-1/comparison.json) reports 12/12 exact token and text matches, and each configuration's 12 measured token sequences also match the Phase 0 edge reference.

| Local CPU configuration | Median observed request | Median stage residual | Largest observed post-request process RSS | Median full request bytes |
| --- | ---: | ---: | ---: | ---: |
| Edge/full | 27.657 s | 2.356 ms | 4,451 MiB edge | No network request |
| Cloud/raw image | 28.293 s | 0.744 ms | 4,870 MiB cloud | 526.5 B |
| Distributed/projected synchronous | 29.333 s | 2.429 ms | 1,604 MiB edge; 4,825 MiB cloud | 2,986,434.5 B |

The prompt input length was 9–13 exact tokens, median 11. The projected payload is exactly 2,985,984 bytes in every split row, with additional framing and metadata bytes recorded separately. The no-op CPU timing-wrapper estimate in the final summary is 0.000673 ms per call. The median cloud transport residual is 3.458 ms; the split residual is 16.591 ms and includes transfer/framing on localhost. These times are descriptive observations under one WSL host, **not** a network-performance ranking. The reported RSS values are snapshots after each request, not isolated per-request peaks.

The [archive](../../artifacts/phase2/2026-10-26/local-1.tar.gz) contains the three raw runs, final profile, comparison, specification, and [per-file hashes](../../artifacts/phase2/2026-10-26/local-1/file_hashes.json). Archive SHA-256: `b995bf01874725d8f6d0d89dea8b6194d6107755e685fc07a4038f088daed1f2`. The profile specification SHA-256 is `8c194ed46b5752bf4a5c8af5ead19fb1be082803f4f7e6fdf8b8207254402f8e`, model contract `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`, and dataset manifest `c5bc9df4a010dcefbcbb7fd60c9dd1e47dc0e0cf798acff579a1a004006ce8d1`.

Ruff, strict mypy, and the complete test suite passed (`135 passed`, zero skipped with loopback access). The cloud and edge run manifests truthfully show `git_dirty=true` because profile tooling/documentation was being edited while their inference processes ran. The distributed run started clean. The measured inference path had already been committed as `624d3f8` and was not modified during the campaign. A clean-tree rerun is advisable before publication-quality latency claims.

### How to reproduce and view the local results

From the repository root, with the verified `.model-cache`, start the dual-path server in terminal A:

```bash
.venv-model/bin/adaptive-vlm adaptive-server \
  --config configs/experiments/local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
```

In terminal B, run the same 12-request, seeded schedule once per baseline mode. Use fresh directories, or `--resume` after an interruption with the exact same arguments:

```bash
.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/local_campaign.yaml \
  --mode edge --run-dir artifacts/phase2-replay/edge \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --vision-device cpu --text-device cpu \
  --warmups 1 --repetitions 1

.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/local_campaign.yaml \
  --mode cloud --run-dir artifacts/phase2-replay/cloud \
  --warmups 1 --repetitions 1 --offline

.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/local_campaign.yaml \
  --mode distributed --run-dir artifacts/phase2-replay/distributed \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --vision-device cpu \
  --pipeline synchronous --warmups 1 --repetitions 1

.venv/bin/adaptive-vlm profile-runs \
  artifacts/phase2-replay/edge artifacts/phase2-replay/cloud \
  artifacts/phase2-replay/distributed \
  --specification configs/profiles/local_fixture.json \
  --output-dir artifacts/phase2-replay/profile
```

Open `profile/summary.json` for coverage, failures, residual-error counts, and equality; `profile/profile.jsonl` has one complete JSON object per attempt. The individual `edge/`, `cloud/`, and `distributed/` run directories retain the raw `requests.jsonl`, frozen `plan.json`, `run_manifest.json`, and `summary.json`. `compare-runs` offers a separate full token-equivalence audit.

### Gate scope and remaining work

The local campaign uses three 4×4 synthetic image fixtures, short prompts, one 32-token generation budget, one excluded warmup and one measured sweep. Its specification labels localhost bandwidth/RTT as **uncontrolled and unknown**. This establishes measurement plumbing and local sample coverage for a narrow task bucket; it cannot calibrate a network crossover or imply a physical edge/cloud speedup. The CPU no-op timing-wrapper calibration is only a proxy for instrumentation overhead; CUDA synchronization and end-to-end A/B perturbation remain unmeasured. CPU RSS is an after-request process reading, not a per-request high-water mark.

To close full G2, provision the separate cloud PC, freeze its hardware/software and model artifact evidence, then run a predeclared image/prompt/output-budget task grid under controlled uplink and RTT states. Each supported task/network bucket needs at least 10 successful measured samples per configuration from three request IDs, with failures retained. Record a direct instrumentation A/B overhead bound and physical-host memory peaks, rerun the profile and equality audit, and keep the new internal split candidates inactive until their own tensor round-trip, partition/memory and token-equivalence evidence exists.

## Local component predictor and holdout

Date: 2026-10-26 (Africa/Cairo). Phase 3 follows the [implementation plan](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/dynamic_scheduler_implementation_plan.md) and [Phase 2 profile handoff](local-prototype-validation.md#stage-instrumentation-and-local-profile). The predictor and local evaluation pipeline are implemented. **G3 has not passed:** the predeclared local request holdout misses the ranking target, and there is no controlled network-condition holdout while the separate cloud PC is unavailable.

### What changed

- Added typed predictor inputs, a provenance-bound calibration, named latency components, wire-byte estimates, an uncertainty interval, and explicit invalid/coverage reasons in `scheduler/predictor.py`. Matching the host pair, model contract, placement profile, task bucket, generation contract, feature ranges and network regime is required before prediction.
- Added `calibrate-predictor`, which fits only successful measured training rows from a Phase 2 profile and evaluates untouched request IDs and network blocks. Warmups and failures cannot fit coefficients. The command compares against a per-configuration constant training-median predictor and reports total-latency median/p90 APE, signed bias, interval coverage and paired ranking accuracy by task/network bucket.
- Frozen the [local split](../../configs/predictor/local_holdout.json) in commit `cb14032` **before** fitting: requests `correctness-001` through `correctness-010` train; `correctness-011` and `correctness-012` are held out. The split pins the Phase 2 profile SHA-256 and 15% median / 30% p90 APE targets. No network block can be held out from the one uncontrolled loopback bucket.
- Added tests for byte-to-bit units, monotonic transfer cost, deterministic predictions, missing and out-of-range state, model/host/placement mismatch, CPU memory exclusion, controlled-network fitting, profile provenance, and output-length leakage. The [predictor contract](../design/predictor-v1-format.md) defines the component equation and coverage behavior.

Implementation commits: `cb14032` (frozen split), `b089cab` (predictor, fitter, evaluator and CLI), and `8b1c03d` (calibration validation). The resulting calibration identifier is `2917acf00f8df890e474d4e5d9c0f9ca66555631a298d004702539f5a85b208d`.

### Held-out local evidence

The [evaluation](../../artifacts/phase3/2026-10-26/clean-1/evaluation.json) has two held-out requests per configuration, six prediction rows, no failures and no out-of-coverage inputs. The evaluator passes `expected_output_tokens=null` for every held-out task and substitutes the **training** median of seven output tokens. It never uses a held-out generated length to predict that request.

| Configuration | Median APE | p90 APE | Mean signed bias | Constant-median APE | Held-out interval coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Edge/full | 3.64% | 4.92% | +404 ms | 3.61% | 2/2 |
| Cloud/raw image | 2.87% | 3.14% | −116 ms | 2.92% | 2/2 |
| Distributed/projected synchronous | 4.89% | 6.35% | −597 ms | 5.07% | 2/2 |

All candidate total-latency APEs meet the numeric 15%/30% thresholds on this very small local holdout. **Paired ranking misses the predeclared target:** the component predictor selected the observed fastest configuration in 0/2 cases, while the constant-median baseline did so in 2/2. The predictor placed cloud about 6 ms ahead of edge for each request; edge was the observed winner both times. These are one-sweep CPU/loopback observations and do not establish a stable policy ranking. The intervals use training residuals and 2/2 held-out coverage is too little for an uncertainty claim.

The [calibration](../../artifacts/phase3/2026-10-26/clean-1/calibration.json), [per-trial predictions](../../artifacts/phase3/2026-10-26/clean-1/predictions.jsonl), and [environment record](../../artifacts/phase3/2026-10-26/clean-1/environment.json) preserve the fitted components, byte estimates, fallback output estimate, observed latency, constant baseline, coverage decision, and Git/software provenance. The environment records clean commit `e6bbd17ab50911637e5eb2553350f2635184932c`. The [archive](../../artifacts/phase3/2026-10-26/clean-1.tar.gz) contains these files, the frozen split and [per-file hashes](../../artifacts/phase3/2026-10-26/clean-1/file_hashes.json). Archive SHA-256: `dc4dec11f286bba21aeca379f42b960275ca2a8018ccdb0267fb7ea621e7b8eb`. Source profile SHA-256: `2a009d880b9c861ee5eedd8d1a60d982d59e91fbd359397d22f0274d1356e054`.

Ruff and strict mypy passed; the full suite passed with `141 passed`, zero skipped using loopback access.

### How to run and inspect it

From the repository root, use a fresh output directory:

```bash
.venv/bin/adaptive-vlm calibrate-predictor \
  --profile artifacts/phase2/2026-10-26/local-1/profile/profile.jsonl \
  --split configs/predictor/local_holdout.json \
  --output-dir artifacts/phase3-replay
```

Open `artifacts/phase3-replay/evaluation.json` for the gate flags and grouped error/ranking tables. Inspect `calibration.json` for fitted components and coverage bounds, and `predictions.jsonl` for each held-out task snapshot, component estimate and observed result. `Calibration.load(path)` and `Predictor(calibration).predict(spec, state, task)` expose the pure runtime API. Out-of-coverage predictions return `valid=false` with a reason and are not execution decisions.

### Gate scope and next work

The local predictor supports only the measured 4×4 synthetic fixture bucket on the same WSL host and uncontrolled loopback. It uses the measured loopback transport residual; physical uplink/downlink and RTT coefficients remain uncalibrated. CPU memory readings are after-request snapshots, so an edge memory budget makes edge/split prediction invalid with `edge_memory_requirement_unknown`. Live clients still need a pre-inference exact prompt-token feature and a justified output-length estimate.

Full G3 needs a separate cloud host, controlled bandwidth/RTT blocks, a frozen request and network holdout, and enough training coverage after excluding both. The local ranking miss blocks using this fit as an automatic performance selector. If the model or decision rule is revised, declare a **new** evaluation split/campaign before measuring it; do not treat the two already inspected holdout requests as a fresh test. Keep the constant-median policy as a reference and report both ranking and latency error on the new holdout. Phase 4 selector logic can be implemented and tested synthetically, but performance claims and automatic activation must wait for a passing gate and safe memory/executor evidence.

## Selector and offline oracle

Date: 2026-10-26 (Africa/Cairo). Phase 4 implements the selector and retrospective oracle from the [scheduler plan](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/dynamic_scheduler_implementation_plan.md). **G4 remains open**, as do the physical G2 and failed local G3 gates. The separate cloud machine is not yet available. The implemented interfaces are ready to consume a future physical-host campaign.

### What changed

- Added a pure `select` function that enumerates the three registered configurations, excludes unsafe or unknown candidates with explicit reasons, and chooses the lowest predicted latency with stable tie handling. It records all candidate predictions, capability/health status, SLA result, and constraints. It performs no inference or dispatch.
- Added a separate predicted oracle for deterministic selector checks and an offline measured oracle over matched held-out observations. The latter reports regret, top-1 agreement, exclusions, SLA violations, incomplete/no-feasible blocks, paired bootstrap intervals, and repeated-condition oracle uncertainty. Measured outcomes cannot alter predecision feasibility; outcome-derived failed-trial predictions are rejected.
- Added `evaluate-selector`, the frozen [local oracle specification](../../configs/oracle/local_holdout.json), and synthetic gate fixtures. The Phase 2 profile and Phase 3 prediction pipeline now carry repetition as a condition-block ID, and Phase 3 ranking counts repeated blocks separately. See the [selector contract](../design/selector-and-oracle.md) for the exact API and JSON behavior.

Implementation commits: `b5527e8` (selector, oracle, specification, CLI, fixtures) and the documentation/follow-up commit recorded in Git history. The code is deliberately outside `adaptive-infer`: its current manual configuration choice remains in force until the earlier gates pass and Phase 5 explicitly wires adaptive dispatch.

### Local diagnostic result

The frozen input is the already inspected [Phase 3 predictions](../../artifacts/phase3/2026-10-26/clean-1/predictions.jsonl), SHA-256 `a4abe75aa6704d735b5d35081e78bab500fa03bea561b0219e4b2960f899bc42`, with calibration `2917acf00f8df890e474d4e5d9c0f9ca66555631a298d004702539f5a85b208d`. The [Phase 4 evaluation](../../artifacts/phase4/2026-10-26/local-3/evaluation.json) and [decision trace](../../artifacts/phase4/2026-10-26/local-3/decisions.jsonl) contain two complete request/condition blocks and all three candidates in each. The selector agrees with the predicted oracle in both blocks. Both selected cloud/raw image; edge/full was the observed fastest in both.

| Measure | Local diagnostic |
| --- | ---: |
| Paired blocks | 2/2 |
| Top-1 agreement with measured oracle | 0/2 |
| Median regret | 514.5 ms; 1.90% |
| Median regret paired bootstrap 95% interval | 337.9–691.2 ms; 1.19–2.61% |
| Incomplete/no-feasible blocks in this replay | 0/0 |
| Full G4 evaluable/passed | No/No |

The 1.90% median is below the proposed 10% regret target, but it is descriptive only: two single-host CPU/loopback requests, one repetition, no network holdout, and no physical memory/health validation cannot establish robust regret or oracle uncertainty. The 0/2 ranking also reflects the already failed G3 local check. The two-point bootstrap interval cannot estimate the tails of a real deployment. No automatic performance selection is enabled.

The [local archive](../../artifacts/phase4/2026-10-26/local-3.tar.gz) contains the evaluation, decision trace, frozen oracle specification, Phase 3 predictions/calibration/evaluation, and [per-file hashes](../../artifacts/phase4/2026-10-26/local-3/file_hashes.json). Archive SHA-256: `61ffa1eb1b911fc64bacfc58976ac6e7518a1204f65bb0b21e5295402affe502`. Generated `artifacts/` are ignored by Git, so preserve this archive when transferring the project.

### Run and view it now

From the repository root, use a fresh directory:

```bash
.venv/bin/adaptive-vlm evaluate-selector \
  --predictions artifacts/phase3/2026-10-26/clean-1/predictions.jsonl \
  --specification configs/oracle/local_holdout.json \
  --output-dir artifacts/phase4-replay
```

Open `artifacts/phase4-replay/evaluation.json` for gate flags and aggregate regret/agreement. Open `decisions.jsonl` for each request's complete candidate predictions, reasons, selected ID, measured winner and regret. In Python, construct `CandidateSnapshot` values from `CapabilityRegistry.statuses()` and `Predictor.predict(...)`, then call `select(tuple(snapshots), SelectionConstraints(...))`; inspect `decision.to_dict()` before any future dispatch. A caller must handle `no_feasible_candidate` and `sla_unmet` explicitly.

### When the second machine becomes available

1. Follow [separate cloud PC setup](../../README.md#using-a-separate-cloud-pc). Verify host inventory, matching model contract, reachable dual-path server, remote capabilities and the same frozen manifest. Keep fixed-policy benchmark runs as controls. Declare the physical host pair and controlled network buckets in a **new** Phase 2 profile specification; run matching edge/cloud/distributed schedules with repetitions. The resulting profile carries each repetition into a condition block. Measure memory safety and host health before treating a candidate as available.
2. Freeze a **new** Phase 3 training/request/network holdout split and its profile SHA-256 before fitting. Run `calibrate-predictor` against that profile/split and inspect `evaluation.json`. G3 must pass its latency and ranking targets on the fresh holdouts. A failed trial's outcome-derived prediction cannot be used for a Phase 4 decision; collect a valid predecision prediction or mark the block unusable and investigate.
3. Before examining observed winners, copy the local Phase 4 JSON specification under a new name. Replace `predictions_sha256` with `sha256sum NEW_PREDICTIONS.jsonl`, the calibration ID, actual held-out IDs/sections, physical predecision availability and health, and `evidence_scope` with `physical-controlled`. Set `safety_evidence_verified=true` only after archiving the matching remote capability, health, and memory checks. Set an SLA, quality floor or memory budget only when supported by measurements; unknown edge memory with a budget excludes edge/split. Commit the frozen specification before running `evaluate-selector --predictions NEW_PREDICTIONS.jsonl --specification NEW_SPEC.json --output-dir NEW_DIR`.
4. Check every `decisions.jsonl` block for candidate coverage and unsafe exclusions. A full G4 assessment requires at least ten complete paired blocks, a network holdout, median regret ≤10%, and selector/predicted-oracle agreement. Repeated condition blocks produce p10/p90 oracle latency; the paired bootstrap provides uncertainty for aggregate regret and agreement. If G3/G4 fails, revise the model or policy and create a **new** split/campaign rather than retuning against the inspected holdout.

The same selector API and CLI remain valid for the second machine; only the physical campaign data, frozen split/specification and availability snapshot change. Phase 5 can later integrate a proven selector into adaptive execution with explicit decision/retry logs.

### Verification

Synthetic fixtures cover ties, nonfinite/negative predictions, invalid/missing profiles, unloaded policy, health, edge memory, SLA feasible/unmet, unknown quality, zero feasible candidates, outcome leakage, repeated blocks, physical gate conditions, and frozen-file hash rejection. The full suite passed with `159 passed`, zero skipped, including loopback integration. Ruff and strict mypy passed; all six archived file hashes were verified.

## Mixed adaptive execution

Date: 2026-10-26 (Africa/Cairo). Phase 5 implements end-to-end mixed configuration execution from the [scheduler plan](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/dynamic_scheduler_implementation_plan.md). The **local G5 integration gate passed** on the pinned Moondream2 CPU/loopback fixture. This is a diagnostic run: local G3 ranking failed, physical G4 is open, and the separate cloud machine is unavailable. Production-style automatic selection remains gated.

### What changed

- Added `adaptive-benchmark` with exact pre-inference image and prompt features, a frozen manual/replay network and health state, the Phase 4 selector, and resident edge/cloud/split dispatch. The plan and each terminal trial row pin the manifest, config, model contract, calibration, replay hash, selection trace, observed result, and all timing. The run can resume only under the identical plan.
- Added one bounded retry for a TCP connection failure that occurs before request bytes are sent. It rechecks capabilities, excludes the failed configuration, and logs both attempts and added time. Other failures are explicit terminal rows without an automatic retry. Existing fixed-policy and manual inference paths remain intact.
- Added `analyze-adaptive` to match each selected request with its same-mode fixed control on the exact schedule. It audits output tokens, placement, wire bytes, provenance, decisions, retries, and terminal-row coverage. The [execution contract](../design/adaptive-execution.md) defines the full input and failure behavior.
- Froze [the local replay](../../configs/adaptive/local_replay_health.json) before the real run. It deliberately assigns four request IDs to each of the three available configurations through declared health exclusions. These are replayed conditions, not measured failures.

Implementation checkpoint: `68a81ac` contains the committed execution path and replay used by the real run. The run manifest records full source commit `68a81acf53b37c85aaf61a38aa893598b5c2852f` and `git_dirty=false`. The subsequent analysis/doc update is checked by the final suite.

### Local Moondream diagnostic

On one WSL host, the dual-path CPU server and full edge CPU adapter ran one excluded warmup and one measured pass over all 12 frozen synthetic requests. The [summary](../../artifacts/phase5/2026-10-26/local-1/summary.json), [analysis](../../artifacts/phase5/2026-10-26/local-1/analysis.json), and [raw rows](../../artifacts/phase5/2026-10-26/local-1/requests.jsonl) show:

| Measure | Result |
| --- | ---: |
| Terminal measured trials | 12/12 successful; zero retries |
| Selected edge / cloud / projected split | 4 / 4 / 4 |
| Same-placement output token mismatches | 0/12 |
| Decision trace / wire-byte / control errors | 0 / 0 / 0 |
| Median observed end-to-end request | 29,776.3 ms |
| Median selection overhead | 2.906 ms |
| Mean application wire bytes: edge / cloud / split | 0 / 1,837.5 / 2,987,753.25 |
| `g5_local_gate_passed` | `true` |

The same request IDs and fixed controls share model contract SHA-256 `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1` and dataset manifest SHA-256 `c5bc9df4a010dcefbcbb7fd60c9dd1e47dc0e0cf798acff579a1a004006ce8d1`. Replay SHA-256: `bfcdf06a5dc283ae596abbda51e1da1096bab097d83a7288fe0f78db865ef7d8`. The source calibration ID is `2917acf00f8df890e474d4e5d9c0f9ca66555631a298d004702539f5a85b208d`.

The fixed controls were run earlier; CPU load and run order were uncontrolled. The replay deliberately forces a mixed choice, so the median latency and wire totals describe this integration campaign only. They are not a scheduler speedup or physical-network crossover estimate. No retry happened in the real run; the bounded retry and no-retry-after-unknown-stage paths passed synthetic tests.

The [archive](../../artifacts/phase5/2026-10-26/local-1.tar.gz) contains the adaptive plan, run manifest, raw rows, summary, analysis, frozen replay/calibration/gate reports, all three fixed control runs, and [per-file hashes](../../artifacts/phase5/2026-10-26/local-1/file_hashes.json). Archive SHA-256: `4659f9fc8804cc140aafbbadb83e0c52ed389e8825f8d0ea3aa2f9e42c381fb6`. Generated `artifacts/` are ignored by Git; preserve the archive when transferring machines.

### Run and inspect it

With the pinned cache and model environment ready, start the dual-path server in terminal A:

```bash
.venv-model/bin/adaptive-vlm adaptive-server \
  --config configs/experiments/local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
```

In terminal B, choose a **new** run directory:

```bash
.venv-model/bin/adaptive-vlm adaptive-benchmark \
  --config configs/experiments/local_campaign.yaml \
  --manifest data/manifests/correctness.jsonl \
  --calibration artifacts/phase3/2026-10-26/clean-1/calibration.json \
  --replay configs/adaptive/local_replay_health.json \
  --run-dir artifacts/phase5-replay \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --vision-device cpu --text-device cpu --cloud-device cpu \
  --warmups 1 --repetitions 1 --diagnostic-allow-unpassed-gates
```

On interruption, repeat the command with `--resume`; the config, manifest, calibration, replay and schedule must be identical. `summary.json` shows final outcomes and choices. Each `requests.jsonl` row shows `attempts[*].decision.candidates` with predictions and exclusions, the selected mode, time, bytes and generated tokens. `run_manifest.json` records the Git commit, cleanliness, software and placement.

Analyze against the original same-schedule fixed controls:

```bash
.venv/bin/adaptive-vlm analyze-adaptive \
  artifacts/phase5-replay \
  artifacts/phase2/2026-10-26/local-1/edge \
  artifacts/phase2/2026-10-26/local-1/cloud \
  artifacts/phase2/2026-10-26/local-1/distributed \
  --output artifacts/phase5-replay/analysis.json
```

The analyzer refuses mismatched schedules, model/manifest/config hashes and incomplete trial sets. Use the bundled controls in the archive if the original `artifacts/phase2` directory was not transferred.

### Wiring a separate cloud machine

1. Verify the cloud host and dual-path server as in the [deployment guide](../../README.md#using-a-separate-cloud-pc). Run fresh fixed edge, cloud, and distributed controls on one frozen physical manifest and identical schedule. Establish real memory and capability evidence before declaring candidate availability.
2. Finish physical G2 profiling and freeze a new G3 training/request/network holdout **before** fitting. Recalibrate on the physical host pair and satisfy G3 ranking and error targets. Evaluate Phase 4 on matched physical holdouts and satisfy G4 safety/regret evidence. Do not reuse the failed local calibration or its inspected requests as a fresh test.
3. Copy the local replay JSON under a new campaign name. Replace manifest/profile/calibration/model/host hashes, task bucket, exact per-candidate placement profiles, and network bucket/source/bandwidth/RTT. The loader checks the placement profiles against runtime device/address arguments and calibration hashes. Use `trial_network_states` for a predeclared condition schedule; use `unhealthy_by_request` only for declared manual fault scenarios. Keep `quality_floor=null` until a validated predecision quality estimate exists. Freeze and commit this file before the adaptive campaign.
4. Start the physical dual-path server, then run `adaptive-benchmark` with the new config/manifest/calibration/replay and matching devices. Once both full gates pass, supply `--predictor-evaluation NEW_G3/evaluation.json --selector-evaluation NEW_G4/evaluation.json`; the diagnostic flag is then unnecessary. Preserve all raw trials and use `analyze-adaptive` with the fresh three fixed controls. Any placement mismatch is disclosed instead of counted as a same-device equality success.

The current CLI requires a resident **full edge model** to count exact prompt tokens before selection. A cloud-only edge that cannot load it needs a separately validated predecision tokenizer/feature path before this mode can run; the Phase 5 command will refuse that unsupported placement.

### Verification and gate status

Synthetic tests exercise mixed 12-request selection, resume without duplicate trials, a single safe pre-send retry, no retry for unknown-stage failure, zero feasible candidates, token and wire matching, and decision-trace tamper detection. The full suite passed with `163 passed`, zero skipped, including loopback integration. Ruff and strict mypy passed. **Local G5 passed; physical G5, G2, G3, and G4 remain open.** Phase 6 live sensing has not been enabled.

## Live probes and network trace replay

Date: 2026-10-26 (Africa/Cairo). Phase 6 follows the [scheduler plan](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/dynamic_scheduler_implementation_plan.md) and [Phase 5 handoff](local-prototype-validation.md#mixed-adaptive-execution). The **local synthetic trace and live-monitor integration checks passed**. The physical G6 gate remains open because the separate cloud host and controlled physical network are unavailable. The Phase 3 local ranking miss and open G4 gate still prohibit validated automatic performance selection.

### What changed

- Added `replay-network-trace` to run frozen bandwidth/RTT/health/memory samples through the real predictor and selector. The deterministic fixture crosses from edge to projected split, stays stable through a noisy fast segment without hysteresis, and falls back safely on stale/failed probes or unknown memory.
- Added `LiveMonitor` with bounded application health and capability probes, timestamped age/cache/timeout provenance, and fail-closed remote eligibility. It computes a bounded passive **application send-rate** estimate after remote inference and leaves downlink goodput unknown. Probe RTT is kept separate from the predictor's declared path RTT. Controlled calibrations now pin and check their network-state source.
- Extended `adaptive-benchmark` with `--monitor-mode live`, per-attempt monitoring records, monitoring/selection overhead, and passive update count. Added `analyze-monitor` to audit sample completeness, unsafe choices, switch count, and overhead impact. Existing replay/default and fixed-policy commands retain their behavior. The [monitor contract](../design/live-monitor-and-trace-replay.md) defines measurements and failure handling.

Implementation checkpoint `ab97c91` contains the committed Phase 6 monitor, trace inputs, three-request live manifest/replay and tests used by the real run. The live run manifest records full source commit `ab97c9171eb317ce0da26af18543c9fccffb422f` with `git_dirty=false`. The final audit/source-provenance refinement and documentation are verified in the follow-up commit.

### Synthetic controlled trace

The [trace evaluation](../../artifacts/phase6/2026-10-26/synthetic-trace-1/evaluation.json) and [decision rows](../../artifacts/phase6/2026-10-26/synthetic-trace-1/decisions.jsonl) contain nine deterministic samples. At 1 Mbit/s uplink, edge/full is selected; at 100 Mbit/s, projected split is selected. The declared fast segment varies from 85 to 100 Mbit/s after warmup with **zero switches**. A stale sample, failed probe, and edge-memory-budget exclusion yield the expected safe outcomes, including an explicit no-feasible decision when remote health is lost and edge memory is insufficient. The report has zero expected-choice mismatches and zero unsafe selections. It counts three configuration transitions overall and zero per-switch model loads because all candidates are assumed resident. `synthetic_trace_gate_passed=true`.

This fixture was designed to exercise the decision boundary and is not trained from physical observations. The [synthetic archive](../../artifacts/phase6/2026-10-26/synthetic-trace-1.tar.gz) contains its calibration, trace, decisions, evaluation and [file hashes](../../artifacts/phase6/2026-10-26/synthetic-trace-1/file_hashes.json). Archive SHA-256: `5a7df280576c3c009ed116070a89036ad260de3cc568e3dc18a3c09508b2f268`.

### Local Moondream live-monitor diagnostic

The [three-request frozen manifest](../../data/manifests/local_monitor_probe.jsonl) and [replay](../../configs/adaptive/local_live_monitor_replay.json) select one edge, one cloud, and one projected-split request on one WSL CPU/loopback host. A separate dual-path server was running; its full-model load and the client's resident edge-model load both finished before the request schedule. The replay's health exclusions are declared laboratory inputs rather than real observed outages. The [raw run](../../artifacts/phase6/2026-10-26/local-live-1/requests.jsonl), [summary](../../artifacts/phase6/2026-10-26/local-live-1/summary.json) and [monitor audit](../../artifacts/phase6/2026-10-26/local-live-1/monitor_analysis.json) show:

| Measure | Local result |
| --- | ---: |
| Measured requests / successful | 3 / 3 |
| Edge / cloud / projected split choices | 1 / 1 / 1 |
| Timestamped decision samples / probe errors | 3 / 0 |
| Unsafe selections / audit failures | 0 / 0 |
| Median / maximum monitor sampling cost | 1.505 / 34.441 ms |
| Median monitoring share of observed latency | 0.00466% |
| Successful passive send-rate updates | 1 |
| Switches across successful measured trials | 2 |
| `local_live_monitor_gate_passed` | `true` |

All three probes were fresh (`age_ms=0`); their health-probe round trips were approximately 0.792, 30.45 and 1.063 ms. The longest probe occurred before the cloud request under local CPU load. The one projected-tensor transfer yielded a final bounded application send-rate estimate of about **2.20 Gbit/s**. That value reflects local socket buffering and is **not physical uplink goodput**; no later request consumed it. The live selector kept the original declared loopback state for prediction. No probe failure or safe retry occurred during this live run, so those behaviors are established by synthetic tests rather than measured fault recovery. The local run's three choices came from replay exclusions, not a measured bandwidth crossover.

The [live archive](../../artifacts/phase6/2026-10-26/local-live-1.tar.gz) includes the complete adaptive run, monitor audit, manifest, replay, calibration and prior G3/G4 gate reports with [per-file hashes](../../artifacts/phase6/2026-10-26/local-live-1/file_hashes.json). Archive SHA-256: `c7a663d37befc8533d4d88fac59a0dd121109b6fc518239f6975a4c5a12ca0a3`. Generated `artifacts/` are ignored by Git; preserve both archives when moving machines.

### Run and view it now

Replay the deterministic trace without loading Moondream:

```bash
.venv/bin/adaptive-vlm replay-network-trace \
  --calibration configs/network_traces/synthetic_calibration.json \
  --trace configs/network_traces/synthetic_trace.json \
  --output-dir artifacts/phase6-trace-replay
```

For the real local probe, start `adaptive-server` in terminal A:

```bash
.venv-model/bin/adaptive-vlm adaptive-server \
  --config configs/experiments/local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
```

In terminal B, choose a new output directory:

```bash
.venv-model/bin/adaptive-vlm adaptive-benchmark \
  --config configs/experiments/local_campaign.yaml \
  --manifest data/manifests/local_monitor_probe.jsonl \
  --calibration artifacts/phase3/2026-10-26/clean-1/calibration.json \
  --replay configs/adaptive/local_live_monitor_replay.json \
  --run-dir artifacts/phase6-live-replay \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --vision-device cpu --text-device cpu --cloud-device cpu \
  --warmups 0 --repetitions 1 --diagnostic-allow-unpassed-gates \
  --monitor-mode live --probe-interval-ms 0 \
  --probe-max-age-ms 5000 --probe-timeout-ms 2000 --passive-alpha 0.25

.venv/bin/adaptive-vlm analyze-monitor artifacts/phase6-live-replay \
  --output artifacts/phase6-live-replay/monitor_analysis.json
```

Inspect `requests.jsonl` → `attempts[*].monitoring` for the probe timestamp, RTT, age, ready paths, error and passive estimate beside each decision; `monitor_analysis.json` reports safety/coverage and monitor overhead. Use the same command with `--resume` only after an interrupted run and identical plan inputs. The diagnostic flag remains required for the current local calibration.

### Wiring a second machine

1. Complete the verified hardware and dual-path server setup from the [deployment guide](../../README.md#using-a-separate-cloud-pc). Rebuild Phase 2 profiles across controlled physical bandwidth/RTT blocks and record exactly how each state was measured. Freeze a new Phase 3 split with request and network-block holdouts. Calibrate on the new host pair; controlled fits now bind their source name, and live states from another source are rejected rather than extrapolated.
2. Replace the synthetic trace with a frozen physical trace using the new calibration/task bucket and in-coverage bandwidth/RTT samples. Declare the expected crossover and a constant-condition noisy segment before collecting held-out outcomes. Run `replay-network-trace` to verify transitions and switch count. Do not use the synthetic 1/100 Mbit/s crossover as a prediction of physical hardware.
3. Freeze a new Phase 5/6 replay with actual manifest, calibration, per-candidate placement profiles and physical network state. Use `trial_network_states` for a declared condition sequence. Keep `unhealthy_by_request` empty during normal operation, or populate it only for an explicitly documented fault scenario. For edge memory limits, supply a measured peak and budget; the monitor will not infer free GPU memory from process RSS. Start the physical server and run `adaptive-benchmark --monitor-mode live` with bounded probe settings, plus passing G3/G4 reports when available.
4. Archive the raw requests, probe records, predictor inputs, calibration, replay, controlled network measurements and environment. Run `analyze-monitor` and the Phase 5 `analyze-adaptive` against new matching fixed controls. Measure actual path goodput independently; the passive socket-send estimate and application health-probe RTT are labeled diagnostics and must not be substituted for that measurement. If real noisy traces switch undesirably, predeclare and evaluate a hysteresis rule in a new campaign, counting switching and load costs.

### Gate status and verification

Synthetic fixtures cover the controlled crossover, stable noisy segment, stale/failed probes, no-feasible memory exclusion, cache age, bounded timeout, passive rate clamping/smoothing and resume restoration, and live decision record/audit integration. The full suite passed with `168 passed`, zero skipped, including loopback integration. Ruff and strict mypy passed; both archive hashes and their internal file hashes were verified. **The local synthetic and live G6 checks passed. Physical G6, G2–G4, and validated automatic selection remain open.**

## Task-aware evaluation pipeline

Date: 2026-10-26 (Africa/Cairo). This handoff follows the [Phase 7 protocol](../design/evaluation-protocol.md) and [scheduler plan](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/dynamic_scheduler_implementation_plan.md). **The local analysis pipeline is implemented and verified; full G7 is open.** The separate cloud PC, controlled physical link, and official VQA files are still unavailable. The current calibration's local fastest-choice ranking failed G3, and the Phase 4 performance gate is open. Do not report the local comparison as an adaptive improvement claim.

### Changes made

- Froze the [local specification](../../configs/evaluation/local_thesis_diagnostic.json) in commit `b04b156` before running the Phase 7 analyzer. It pins the model/manifest/calibration hashes, predictor-training IDs, static-validation IDs, evaluation IDs, SLA, bootstrap seed, task label and uncontrolled network regime. The prospective [protocol](../design/evaluation-protocol.md) defines the physical experiment and G7 gate.
- Added `analyze-thesis` to verify complete four-policy schedules, row/run/plan provenance, split disjointness, decision audit, token equivalence, and matching model/config/manifest hashes. It picks best-static using validation SLA success then median latency, reports held-out paired latency/SLA differences and measured-oracle regret, and retains failures in denominators. Hierarchical bootstrap samples network blocks, then request IDs within each block; the report states its counts and seed. It includes successful median/p95 latency, wire bytes, status counts, decision distribution, selection/monitor overhead and prediction error against matched fixed observations. `paired_trials.jsonl` and `report.json` include source hashes. The pipeline checkpoint is commit `1cecb69`.
- Added input-only `task_type` classification to frozen requests and adaptive rows. VQA is `vqa_short_answer`; exact-match prompts are `labeled_short_answer`; other prompts are `unscored_prompt`. A VQA replay must declare its type, and each adaptive campaign must contain one task type. The exact prompt token count and frozen task bucket are already computed before selection; the predictor uses its training-median output-length estimate unless an independently predeclared estimate is supplied. No generated answer is used as a scheduler input.
- Added `prepare-vqa-eval-subset` and `export-vqa-predictions`. The former writes official-schema question/annotation JSON restricted to the imported IDs, which the official loader requires; the latter writes official-format prediction JSON for one repetition, exporting failed requests as empty answers and recording provenance. `scripts/vqa/run_official_vqa_eval.py` invokes a pinned upstream evaluator and hashes its source and inputs. The upstream [GT-Vision-Lab/VQA evaluator](https://github.com/GT-Vision-Lab/VQA/tree/master/PythonEvaluationTools) uses Python 2 syntax, so run it in a compatible environment. It has **not** run here because the official files are absent.
- Updated the README's project state, CLI, VQA workflow and Phase 7 reproduction steps. No model or official dataset files were added to Git.

### Local descriptive result

The [report](../../artifacts/phase7/2026-10-26/local-diagnostic-3/report.json) and [paired table](../../artifacts/phase7/2026-10-26/local-diagnostic-3/paired_trials.jsonl) use the already archived Phase 2 fixed and Phase 5 adaptive CPU/loopback runs. The frozen specification chooses `correctness-001` through `010` for static-policy validation and reports `011`–`012` for evaluation. The predictor had trained on `001`–`010`; its two evaluation IDs were already inspected in prior phases. One measured repetition and uncontrolled policy order make all intervals **descriptive only**.

| Held-out local measure | Adaptive | Edge | Cloud | Projected split |
| --- | ---: | ---: | ---: | ---: |
| Successful / attempted | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| Median request latency | 28.605 s | 27.442 s | 27.957 s | 28.881 s |
| Nearest-rank p95 latency | 29.679 s | 28.425 s | 28.763 s | 30.318 s |
| Mean application wire bytes / attempt | 2,987,709 | 0 | 1,795 | 2,987,716 |
| Success within frozen 60 s SLA | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |

Validation chooses **edge** as best-static. On the two evaluation pairs, adaptive selected projected split twice and averaged **1.162 s slower than edge**; the request-cluster 95% bootstrap interval is 1.070–1.254 s. The same signed value is the measured-oracle regret because edge happened to be fastest in both pairs. Relative to cloud, adaptive was 0.648 s slower on average; relative to fixed projected split, it was 0.276 s faster. Two observations cannot establish an improvement or stable uncertainty bound. Candidate prediction median APE against these separately timed fixed runs was 3.64% edge, 2.87% cloud and 4.89% split. The earlier Phase 5 audit found zero adaptive-versus-selected-control token mismatches; the Phase 7 report also records fixed-policy token comparisons. Official VQA accuracy, measured edge-memory peak, and physical goodput remain `null`. `g7_passed=false`.

The [self-contained archive](../../artifacts/phase7/2026-10-26/local-diagnostic-3.tar.gz) contains the specification, all four run plans/manifests/raw JSONL files, calibration and predictor gate context, report, paired rows and [per-file hashes](../../artifacts/phase7/2026-10-26/local-diagnostic-3/file_hashes.json). Its 18 hashed payload files were verified, and an analysis rerun using only the bundled inputs reproduced `report.json` and `paired_trials.jsonl` byte for byte. Archive SHA-256: `fb552994d808bcc2c72e9ad407f43164aae599c1ed5ad71defc55124aade2917`. `artifacts/` is Git-ignored; preserve the archive when moving machines.

### Reproduce and inspect

With the original local artifacts available, run this from the repository root; no model weights or server are needed:

```bash
.venv/bin/adaptive-vlm analyze-thesis \
  --specification configs/evaluation/local_thesis_diagnostic.json \
  --adaptive-run artifacts/phase5/2026-10-26/local-1 \
  --fixed-runs artifacts/phase2/2026-10-26/local-1/edge \
    artifacts/phase2/2026-10-26/local-1/cloud \
    artifacts/phase2/2026-10-26/local-1/distributed \
  --output-dir artifacts/phase7-reproduction
```

To reproduce using only the Phase 7 archive, extract it and replace the five paths above with `local-diagnostic-3/inputs/specification.json`, `inputs/adaptive`, `inputs/edge`, `inputs/cloud`, and `inputs/distributed`. The generated `report.json` shows frozen hashes, status and metric tables, validation choice, paired bootstrap results, prediction errors and oracle regret. `paired_trials.jsonl` shows one record per held-out trial. Preserve complete raw rows; the report is a derived view.

### Complete the prospective physical and VQA campaign

1. Verify the second host using the [deployment guide](../../README.md#using-a-separate-cloud-pc) and close G0 physical. Freeze new task manifests and the 32-token generation contract. Import the official VQA v2 val2014 subset using the README command. Define a separate short-answer VQA bucket and a distinct descriptive task bucket if the pinned model supports it; profile and calibrate each. Do not reuse the synthetic task fit or mix a 100/300-token contract into the 32-token campaign.
2. Measure controlled uplink/downlink/RTT blocks and memory peaks on the real host pair. Freeze a new Phase 2 profile and a Phase 3 split with disjoint fitting, static-validation and evaluation request IDs **and** held-out network blocks. Refit until the new G3 predictor and G4 selector gates pass; replay the physical trace and validate G6 health/safety. Record shaper settings and independent achieved-network measurements for every policy/block. A replayed state label alone is not proof of a physical network condition.
3. Copy the local Phase 7 JSON specification under a new campaign name and replace all hashes, task label, IDs, SLA, bootstrap settings and block IDs **before running the evaluation**. Use `design: "prospective"` and nonempty, disjoint validation/evaluation network-block lists. Use the same randomized per-request schedule, warmup count, repetition count and model contract for all four policies within each block; rotate the policy execution order between blocks. Run fixed `benchmark` controls and `adaptive-benchmark --monitor-mode live` using the fresh calibration/replay and passing G3/G4 reports. A fresh output directory is required per run; keep failures and all warmups. Supply the actual measured edge-memory peak/budget to the runtime rather than inferring it from RSS.
4. Run `analyze-thesis` on each matching block campaign, preserving the raw runs and the report. The analyzer checks the adaptive rows' declared block IDs against the specification; independent physical-network logs and fixed-policy condition verification still need to be archived and reviewed. The current CLI analyzes one matched run quartet per invocation, so combine per-block results only with a separately frozen aggregation protocol; do not pool a local loopback diagnostic with physical observations. Use the official scorer below for each VQA policy/repetition. G7 remains open until all predeclared cases and external evidence are present.

For a VQA campaign, after the four raw runs complete:

```bash
.venv/bin/adaptive-vlm prepare-vqa-eval-subset \
  --manifest data/manifests/vqa_val2014_seed42.jsonl \
  --questions data/raw/vqa/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations data/raw/vqa/v2_mscoco_val2014_annotations.json \
  --output-dir artifacts/vqa-official-subset

.venv/bin/adaptive-vlm export-vqa-predictions \
  --manifest data/manifests/vqa_val2014_seed42.jsonl \
  --run-dir artifacts/NEW_VQA_POLICY_RUN \
  --repetition 0 --output artifacts/vqa-policy-r0.json

python2 scripts/vqa/run_official_vqa_eval.py \
  --vqa-root /path/to/pinned/GT-Vision-Lab-VQA \
  --questions artifacts/vqa-official-subset/questions.json \
  --annotations artifacts/vqa-official-subset/annotations.json \
  --predictions artifacts/vqa-policy-r0.json \
  --output artifacts/vqa-policy-r0-official-eval.json
```

Pin the official evaluator checkout commit and retain the generated source/input hashes and per-question scores. Repeat export/scoring for every policy and repetition. The runner's provisional VQA `quality_score` cannot replace this result.

### Verification and remaining gates

The Phase 7 tests cover validation-only static selection, failure-inclusive SLA accounting, paired-success coverage, training/evaluation leakage rejection, incomplete raw rows, deterministic official-format VQA export with failed answers and matching subset creation. Ruff, strict mypy and the full test suite pass; the complete suite has **171 passed** including loopback integration. The local report and archive hashes are verified. The VQA adapter's CLI was checked but official scoring could not run without the files and compatible evaluator checkout. The physical-host campaign, official VQA scoring, fresh held-out controlled network comparisons, measured edge-memory peak, and G3/G4/G6 physical gates remain open. A complete negative adaptive result on that prospective campaign would still satisfy the thesis evaluation design.
