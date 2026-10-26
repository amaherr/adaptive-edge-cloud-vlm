# Phase 0 handoff — verified local baseline

Completed: 2026-10-26 (Africa/Cairo). Scope: G0-local from the [scheduler plan](dynamic_scheduler_implementation_plan.md). **G0-local passed; G0-physical remains pending** because a separate cloud PC is not available.

## What was made

- Added `scripts/run_phase0_local.sh`, a fresh-directory, offline, one-sweep runner for edge, cloud and synchronous distributed CPU policies. It captures input hashes, environment, logs, raw benchmark directories and `compare-runs` output, then asserts 12 successful measured requests per policy and exact equality for all 12 paired requests. It refuses to overwrite a previous run and stops its local server on exit.
- Increased its server startup allowance to 600 seconds and guarded the readiness-log check against a process-start race. The first attempt's 180-second limit was too short for a cold model load.
- Re-ran Ruff, strict mypy and the full CPU test suite with loopback socket access; the TCP subprocess test ran instead of being skipped.
- Repeated the real pinned Moondream2 CPU/BF16 correctness campaign. The complete raw artifacts are local and ignored by Git; archive the directory separately before moving machines.

No scheduler or model behavior was changed in this phase.

## Inputs and provenance

The successful edge, cloud and distributed-retry runs used the clean source commit `ce325d24345084cd9c9213948f8cebd32795eb06`, the shared `configs/experiments/chunk6_local_campaign.yaml` settings, seed 42, one excluded warm-up and one measured sweep of the 12-request synthetic correctness manifest. Each policy used CPU for all model components and the same pinned Moondream2 revision and greedy 32-token cap. Cloud and distributed servers ran on loopback (`127.0.0.1:5050`) on the same WSL host; no physical network or separate cloud host was involved.

| Frozen input | SHA-256 |
| --- | --- |
| `data/manifests/correctness.jsonl` | `c5bc9df4a010dcefbcbb7fd60c9dd1e47dc0e0cf798acff579a1a004006ce8d1` |
| `configs/experiments/chunk6_local_campaign.yaml` | `0994e0fbbd05c8e7fd7e4c5b82413d5a0531ba39805d201b8422e32a374d64ff` |
| `configs/model/moondream2.yaml` | `831f7255a11c5be66816e821497e2f54cb1c114ca34bfe8e043c5f95174f2d09` |
| `configs/model/moondream2_artifact.yaml` | `369640ff5a46c7f11c6b3f20a22f780d9ae4501e34db80fe2eaa11263df3a650` |
| `requirements/model-cu128.lock` | `96d07ba6bb6310b29d259c64652de18f4c5f72f2f0629a17844d09ffacedffbd` |

The model contract fingerprint recorded by `compare-runs` is `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`. The local environment record reports WSL2, an i7-11800H, 7.637 GiB RAM, and a 4 GiB RTX 3050; inference in this campaign remained CPU-only. See `environment.json`, `inputs.sha256`, and each policy's `run_manifest.json` in the [local artifact directory](../artifacts/phase0/2026-10-25/local-campaign-1/). The local [compressed archive](../artifacts/phase0/2026-10-25/local-campaign-1.tar.gz) has SHA-256 `71050d767dbb2de58ffdeb169e8c75e005d7f0bcccdf42edfafd1ce18f977b95`. Both paths are ignored by Git; copy the archive to durable storage before moving machines.

## Gate evidence and interpretation

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

## How to reproduce and view the results

From the repository root, with `.venv`, `.venv-model` and the verified local `.model-cache` installed, run the updated script in an environment that permits loopback TCP binding:

```bash
scripts/run_phase0_local.sh artifacts/phase0/NEW_LOCAL_RUN
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

## Remaining Phase 0 gate

G0-physical is pending. Once a separate cloud PC is available, record its verified hardware/software inventory in `configs/hardware/`, place the same pinned artifact and dependency lock on both hosts, configure a protected reachable TCP endpoint, and repeat the same frozen manifest with CPU/CPU numerical placement first. Archive both environment records, server readiness metadata, network topology and independently measured bandwidth/RTT. Require all 12 pairs present, zero measured failures and `complete_and_equal=true` before claiming physical-host correctness. Controlled latency comparisons belong to later profiling/evaluation phases, not this local gate.
