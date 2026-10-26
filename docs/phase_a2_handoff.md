# Phase A2 handoff — Physical Experiment Baseline v1

Date: 2026-10-27 (Africa/Cairo). Phase A2 of the [physical validation roadmap](physical_adaptive_vlm_validation_roadmap.md) is complete. The immutable source starting point for future two-machine experiments is:

```text
Tag:        physical-experiment-baseline-v1
Commit SHA: 867f11d68882faae63b83a823489f36d5d57e519
```

The baseline commit is `chore: freeze pre-physical adaptive VLM baseline`. Its [GitHub CI run](https://github.com/amaherr/adaptive-edge-cloud-vlm/actions/runs/40405829702) passed on that **exact SHA**. The annotated tag was pushed and the remote peeled tag was verified to resolve to the same SHA. The working tree was clean before the freeze and after the push. The [baseline record](physical_experiment_baseline_v1.md) is included in the tagged commit.

## Use this source on both machines

```bash
git fetch origin --tags
git checkout --detach physical-experiment-baseline-v1
git status --short
git rev-parse HEAD
```

The last command must print `867f11d68882faae63b83a823489f36d5d57e519` on both edge and cloud. Keep this SHA in every experiment's run manifest. The existing benchmark machinery records Git commit and dirty state; also preserve model artifact, configuration, dataset manifest, calibration/replay, dependency lock, device placement and hardware/network evidence hashes. Generated `artifacts/` and model caches are not contained in the Git tag.

This handoff document was added **after** the baseline was tagged so it could record the resolved SHA; it does not change the tagged source tree. Use the tag for the first physical campaign, even if `main` later advances.

## Gate and next step

Phase A1 had 185 passing local tests, Ruff and mypy passes, 80.54% branch-aware coverage against the unchanged 80% threshold, and green GitHub CI. Phase A2 produced a clean, pushed, CI-green baseline commit and remote tag. No model revision, generation contract, scheduler policy or physical-host result changed in A2. Continue with Phase B: assign stable Edge and Cloud roles when the second machine is available, then collect the two verified inventories before physical correctness or performance runs.
