# Phase 2 handoff — local instrumentation and profile campaign

Date: 2026-10-26 (Africa/Cairo). Phase 2 follows the [implementation plan](dynamic_scheduler_implementation_plan.md) and [Phase 1 handoff](phase1_handoff.md). This document records the executable local slice of G2 and separates it from the still pending physical-host and controlled-network slice.

## What changed

- The pinned Moondream adapter now records exact prompt token count, prompt preparation, prompt prefill, decode, decode steps and decoded image dimensions in each edge/cloud path. The generation token sequence and 32-token greedy contract remain unchanged.
- The TCP client records connect, frame-encode/send and response-wait/decode intervals. Cloud inference records image file-read and full exchange time. Both remote policies already record exact request/response frame bytes; the projected split also records payload serialization and server tensor deserialization. The Phase 1 manual dispatcher now records its selection overhead for future adaptive runs.
- The benchmark now timestamps every attempt. `profile-runs` validates complete matching schedules, raw trial provenance, model/config/manifest and image hashes, exact prompt/output counts, client/server request bytes, and payload bounds. It emits versioned rows containing measured successes, warmups, failures, stage residuals, null reasons, memory readings, output hashes and raw metrics. Its summary reports sample coverage, failure rates and cross-configuration equality.
- The [profile schema](phase2_profile_schema.md) defines every timing boundary and limitation. The [split inspection](phase2_split_feasibility.md) identifies possible vision/text boundaries in the pinned graph. None passed the new tensor-contract and end-to-end equivalence gate, so the executable registry remains the three Phase 1 configurations.

Implementation commits: `624d3f8`, `119be9c`, `28d9e2b`, `61a2066`, and `e2fad49`. The run artifacts preserve their own Git commit and dirty flag; profile rows link their run manifests and the pinned model/dataset hashes.

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
