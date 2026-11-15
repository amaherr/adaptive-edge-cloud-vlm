# Physical testbed: hosts and source baseline

The edge laptop (RTX 3050, 4 GiB) and the university cloud PC (RTX 2070 SUPER, 8 GiB), how they were first verified, and the frozen source baseline used before physical work.

## Edge/cloud pair verification (2 November 2026)

### Hosts and contract

- Edge: Windows Wi-Fi `192.168.8.71`, WSL project at `/mnt/d/Programming/adaptive-edge-cloud-vlm`.
- Cloud at the first campaign: Windows Wi-Fi `192.168.8.70`, WSL server on TCP 5050 through Windows portproxy. The later Ethernet connection uses `192.168.8.74`; the experiment transport config and portproxy were updated for subsequent runs.
- Both repositories were at Git commit `ee29a81` for the first physical requests.
- The cloud Windows Public firewall permits inbound TCP 5050 only from `192.168.8.71`.
- Model artifact SHA-256: `70a7d94c0c8349eb58ed2d9e636ef2d0916960f321ecabeac6354b8ba3d7403f`.
- Model contract fingerprint: `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`.

### Completed checks

The cloud server loaded the pinned Moondream2 model offline on `cuda:0` and advertised `raw_image` and `projected_synchronous`. The cloud experiment configuration validated.

| Path | Result | Evidence |
| --- | --- | --- |
| `cloud/raw_image` | PASS | Edge request `adaptive-0001`, run `20261102T085442Z-f40eb341`, 32 output tokens, cloud projected source and target `cuda:0`. |
| `distributed/projected/synchronous` | PASS | Edge request `adaptive-0001`, run `20261102T090644Z-9aa66222`, 32 output tokens. Edge sent a BF16 `[729, 2048]` tensor (2,985,984 payload bytes) from CPU vision; cloud GPU handled text. |
| `edge/full` | PASS for execution; different output | Corrected-prompt edge request `adaptive-0001`, run `20261102T094044Z-29e38feb`, 32 output tokens on CPU. Its tokens match the CPU edge row for `correctness-001` and diverge from the cloud GPU output at token 20 (one-based). |

The two completed paths returned identical decoded text and identical 32 token IDs for `tests/fixtures/moondream_smoke.ppm` with prompt `Describe this image.`. The physical split request transferred 2,986,419 bytes to the server and completed successfully.

### Twelve-request Phase-1 campaign

The edge ran `scripts/verification/verify_resident_configurations.py` on the frozen `data/manifests/correctness.jsonl` manifest at commit `ee29a81`. All 36 inferences returned output tokens. Eleven of 12 request triples have exactly equal token IDs and text. For `correctness-001`, `cloud/raw_image` and `distributed/projected/synchronous` match exactly, while `edge/full` diverges at token 20. The script wrote `complete_and_equal: false` and raised `RuntimeError: Phase 1 outputs differ`. The raw edge report was copied to `artifacts/phase1/physical-v1/phase1_smoke.json` on the cloud host; this path is ignored by Git.

The same divergence occurs on the corrected single-image smoke request. Later cloud CPU reference runs matched the cloud GPU paths exactly, so CPU-versus-GPU placement alone does not explain the difference. Edge CPU vision feeding cloud CPU text also matched the cloud paths, localizing the difference to edge-local text inference or another host-specific effect after projection. See [the P1 diagnostic](correctness-criteria.md#cpu-reference-divergence). The cloud GPU reports about 4,550 MiB peak allocated for both cloud paths. The model contract fingerprint and artifact SHA-256 match across the hosts.

The physical network and split transport checks pass. The strict three-path token-equivalence gate remains failed at 11/12. An edge GPU placement needs its own equivalence check.

## Pre-physical source baseline

This document is part of the **pre-physical baseline commit**. Resolve its exact source SHA with:

```bash
git rev-parse physical-experiment-baseline-v1^{commit}
```

The annotated `physical-experiment-baseline-v1` tag points to the same commit after its GitHub CI run passes. Check out that tag on both edge and cloud machines before collecting baseline correctness or performance observations:

```bash
git fetch --tags origin
git checkout --detach physical-experiment-baseline-v1
git status --short
git rev-parse HEAD
```

Record the commit SHA, whether the working tree is dirty, and the model/configuration, request-manifest, calibration, replay, dependency-lock and hardware-inventory hashes with **every** run. A matching source commit alone does not prove matching model weights, devices or network conditions; keep the existing run manifests and artifact verification checks.

This baseline freezes the implemented three-placement scheduler, diagnostic predictor/selector, bounded live monitor and evaluation pipeline after [Phase A1](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/phase_a1_handoff.md). The separate physical cloud PC has not yet been validated. The existing local predictor ranking gate failed and physical G0/G2–G7 remain open. Keep the pinned Moondream revision, BF16 precision, preprocessing and 32-new-token generation contract unchanged for the first physical correctness campaign. Any later source, model or generation change needs a new identified experiment revision and fresh equivalence/profiling evidence.

The next step is Phase B of the [physical validation roadmap](../roadmap.md): assign fixed edge/cloud roles and collect verified inventories. Do not treat this baseline tag as evidence that the physical-host or controlled-network gates have passed.

## Baseline tag freeze record

Date: 2026-10-27 (Africa/Cairo). Phase A2 of the [physical validation roadmap](../roadmap.md) is complete. The immutable source starting point for future two-machine experiments is:

```text
Tag:        physical-experiment-baseline-v1
Commit SHA: 867f11d68882faae63b83a823489f36d5d57e519
```

The baseline commit is `chore: freeze pre-physical adaptive VLM baseline`. Its [GitHub CI run](https://github.com/amaherr/adaptive-edge-cloud-vlm/actions/runs/40405829702) passed on that **exact SHA**. The annotated tag was pushed and the remote peeled tag was verified to resolve to the same SHA. The working tree was clean before the freeze and after the push. The [baseline record](hosts-and-baseline.md#pre-physical-source-baseline) is included in the tagged commit.

### Use this source on both machines

```bash
git fetch origin --tags
git checkout --detach physical-experiment-baseline-v1
git status --short
git rev-parse HEAD
```

The last command must print `867f11d68882faae63b83a823489f36d5d57e519` on both edge and cloud. Keep this SHA in every experiment's run manifest. The existing benchmark machinery records Git commit and dirty state; also preserve model artifact, configuration, dataset manifest, calibration/replay, dependency lock, device placement and hardware/network evidence hashes. Generated `artifacts/` and model caches are not contained in the Git tag.

This handoff document was added **after** the baseline was tagged so it could record the resolved SHA; it does not change the tagged source tree. Use the tag for the first physical campaign, even if `main` later advances.

### Gate and next step

Phase A1 had 185 passing local tests, Ruff and mypy passes, 80.54% branch-aware coverage against the unchanged 80% threshold, and green GitHub CI. Phase A2 produced a clean, pushed, CI-green baseline commit and remote tag. No model revision, generation contract, scheduler policy or physical-host result changed in A2. Continue with Phase B: assign stable Edge and Cloud roles when the second machine is available, then collect the two verified inventories before physical correctness or performance runs.
