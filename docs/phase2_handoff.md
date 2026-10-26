# Phase 2 handoff — local instrumentation and profile campaign

Date: 2026-10-26 (Africa/Cairo). Phase 2 follows the [implementation plan](dynamic_scheduler_implementation_plan.md) and [Phase 1 handoff](phase1_handoff.md). This document records the executable local slice of G2 and separates it from the still pending physical-host and controlled-network slice.

## What changed

- The pinned Moondream adapter now records exact prompt token count, prompt preparation, prompt prefill, decode, decode steps and decoded image dimensions in each edge/cloud path. The generation token sequence and 32-token greedy contract remain unchanged.
- The TCP client records connect, frame-encode/send and response-wait/decode intervals. Cloud inference records image file-read and full exchange time. Both remote policies already record exact request/response frame bytes; the projected split also records payload serialization and server tensor deserialization. The Phase 1 manual dispatcher now records its selection overhead for future adaptive runs.
- The benchmark now timestamps every attempt. `profile-runs` validates complete matching schedules, raw trial provenance, model/config/manifest and image hashes, exact prompt/output counts, client/server request bytes, and payload bounds. It emits versioned rows containing measured successes, warmups, failures, stage residuals, null reasons, memory readings, output hashes and raw metrics. Its summary reports sample coverage, failure rates and cross-configuration equality.
- The [profile schema](phase2_profile_schema.md) defines every timing boundary and limitation. The [split inspection](phase2_split_feasibility.md) identifies possible vision/text boundaries in the pinned graph. None passed the new tensor-contract and end-to-end equivalence gate, so the executable registry remains the three Phase 1 configurations.

Implementation was checkpointed regularly from `624d3f8` through `187ca7a` (see `git log --oneline 624d3f8^..187ca7a`). The run artifacts preserve their own Git commit and dirty flag; profile rows link their run manifests and the pinned model/dataset hashes.

## Local campaign evidence

The [profile summary](../artifacts/phase2/2026-10-26/local-1/profile/summary.json) reports `local_gate_passed=true` for the one declared 4×4 fixture/short-prompt/loopback bucket and `controlled_network_gate_passed=false`. There are 39 rows: three excluded warmups and 36 measured results. Each configuration has 12/12 successful measured trials across 12 distinct request IDs, zero failures, and no negative stage or transport residual. Every successful three-way trial has matching prompt-token count and output-token hash. The independent [comparison](../artifacts/phase2/2026-10-26/local-1/comparison.json) reports 12/12 exact token and text matches, and each configuration's 12 measured token sequences also match the Phase 0 edge reference.

| Local CPU configuration | Median observed request | Median stage residual | Largest observed post-request process RSS | Median full request bytes |
| --- | ---: | ---: | ---: | ---: |
| Edge/full | 27.657 s | 2.356 ms | 4,451 MiB edge | No network request |
| Cloud/raw image | 28.293 s | 0.744 ms | 4,870 MiB cloud | 526.5 B |
| Distributed/projected synchronous | 29.333 s | 2.429 ms | 1,604 MiB edge; 4,825 MiB cloud | 2,986,434.5 B |

The prompt input length was 9–13 exact tokens, median 11. The projected payload is exactly 2,985,984 bytes in every split row, with additional framing and metadata bytes recorded separately. The no-op CPU timing-wrapper estimate in the final summary is 0.000673 ms per call. The median cloud transport residual is 3.458 ms; the split residual is 16.591 ms and includes transfer/framing on localhost. These times are descriptive observations under one WSL host, **not** a network-performance ranking. The reported RSS values are snapshots after each request, not isolated per-request peaks.

The [archive](../artifacts/phase2/2026-10-26/local-1.tar.gz) contains the three raw runs, final profile, comparison, specification, and [per-file hashes](../artifacts/phase2/2026-10-26/local-1/file_hashes.json). Archive SHA-256: `b995bf01874725d8f6d0d89dea8b6194d6107755e685fc07a4038f088daed1f2`. The profile specification SHA-256 is `8c194ed46b5752bf4a5c8af5ead19fb1be082803f4f7e6fdf8b8207254402f8e`, model contract `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`, and dataset manifest `c5bc9df4a010dcefbcbb7fd60c9dd1e47dc0e0cf798acff579a1a004006ce8d1`.

Ruff, strict mypy, and the complete test suite passed (`135 passed`, zero skipped with loopback access). The cloud and edge run manifests truthfully show `git_dirty=true` because profile tooling/documentation was being edited while their inference processes ran. The distributed run started clean. The measured inference path had already been committed as `624d3f8` and was not modified during the campaign. A clean-tree rerun is advisable before publication-quality latency claims.

## How to reproduce and view the local results

From the repository root, with the verified `.model-cache`, start the dual-path server in terminal A:

```bash
.venv-model/bin/adaptive-vlm adaptive-server \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
```

In terminal B, run the same 12-request, seeded schedule once per baseline mode. Use fresh directories, or `--resume` after an interruption with the exact same arguments:

```bash
.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --mode edge --run-dir artifacts/phase2-replay/edge \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --vision-device cpu --text-device cpu \
  --warmups 1 --repetitions 1

.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --mode cloud --run-dir artifacts/phase2-replay/cloud \
  --warmups 1 --repetitions 1 --offline

.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --mode distributed --run-dir artifacts/phase2-replay/distributed \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --vision-device cpu \
  --pipeline synchronous --warmups 1 --repetitions 1

.venv/bin/adaptive-vlm profile-runs \
  artifacts/phase2-replay/edge artifacts/phase2-replay/cloud \
  artifacts/phase2-replay/distributed \
  --specification configs/profiles/phase2_local_fixture.json \
  --output-dir artifacts/phase2-replay/profile
```

Open `profile/summary.json` for coverage, failures, residual-error counts, and equality; `profile/profile.jsonl` has one complete JSON object per attempt. The individual `edge/`, `cloud/`, and `distributed/` run directories retain the raw `requests.jsonl`, frozen `plan.json`, `run_manifest.json`, and `summary.json`. `compare-runs` offers a separate full token-equivalence audit.

## Gate scope and remaining work

The local campaign uses three 4×4 synthetic image fixtures, short prompts, one 32-token generation budget, one excluded warmup and one measured sweep. Its specification labels localhost bandwidth/RTT as **uncontrolled and unknown**. This establishes measurement plumbing and local sample coverage for a narrow task bucket; it cannot calibrate a network crossover or imply a physical edge/cloud speedup. The CPU no-op timing-wrapper calibration is only a proxy for instrumentation overhead; CUDA synchronization and end-to-end A/B perturbation remain unmeasured. CPU RSS is an after-request process reading, not a per-request high-water mark.

To close full G2, provision the separate cloud PC, freeze its hardware/software and model artifact evidence, then run a predeclared image/prompt/output-budget task grid under controlled uplink and RTT states. Each supported task/network bucket needs at least 10 successful measured samples per configuration from three request IDs, with failures retained. Record a direct instrumentation A/B overhead bound and physical-host memory peaks, rerun the profile and equality audit, and keep the new internal split candidates inactive until their own tensor round-trip, partition/memory and token-equivalence evidence exists.
