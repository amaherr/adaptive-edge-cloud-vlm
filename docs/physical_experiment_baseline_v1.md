# Physical Experiment Baseline v1

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

This baseline freezes the implemented three-placement scheduler, diagnostic predictor/selector, bounded live monitor and evaluation pipeline after [Phase A1](phase_a1_handoff.md). The separate physical cloud PC has not yet been validated. The existing local predictor ranking gate failed and physical G0/G2–G7 remain open. Keep the pinned Moondream revision, BF16 precision, preprocessing and 32-new-token generation contract unchanged for the first physical correctness campaign. Any later source, model or generation change needs a new identified experiment revision and fresh equivalence/profiling evidence.

The next step is Phase B of the [physical validation roadmap](physical_adaptive_vlm_validation_roadmap.md): assign fixed edge/cloud roles and collect verified inventories. Do not treat this baseline tag as evidence that the physical-host or controlled-network gates have passed.
