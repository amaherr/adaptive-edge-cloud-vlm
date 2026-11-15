# Predictor V2 — physical G2 execution revision

The user-provided [execution specification](Predictor_V2_Agent_Execution_Plan.md) is preserved as received. During the implementation audit the user explicitly changed its environment scope: **exclude AWS and Vultr; use all three configurations of the completed physical G2 campaign**. This document records that override; cloud-VM identity requirements and example cloud paths in the original specification do not apply to this revision.

The measured pair is the RTX 3050 laptop (4096 MiB dedicated VRAM) and RTX 2070 SUPER university PC (8192 MiB), under the existing native BF16 model/generation contract. The full-edge run used WDDM shared memory, so successful collection does not establish dedicated-VRAM feasibility. Model training here refers to fitting a latency predictor, never fine-tuning Moondream.

## Frozen data and isolation

The authoritative assignments remain `configs/evaluation/physical_vqa_split_v1.json`: 60 training requests with N1/N3/N5/N6/N8 (2700 attempts), 20 validation requests with N2/N7 (360), 20 evaluation requests with N4/N9 (360), and 4680 crossed attempts excluded. All three fixed placements and three repetitions are retained. No resampling or VQA prompt/scoring revision is authorized by this predictor task.

**Prior inspection:** the physical G2 stage analysis already examined N4/N9 outcomes before this predictor specification arrived. The V2 pipeline will prevent their use in fitting, feature scaling, candidate selection and validation, but cannot claim these are scientifically untouched measurements. No final evaluation is authorized in this training/validation phase. A genuinely unseen confirmatory test needs a later independently frozen experiment.

The new source of truth is the checksum-pinned `results/physical-g2-20261110/evidence.tar.gz`, not old CPU/loopback coefficients or AWS/Vultr results. Keep raw logs unchanged. Preserve V1 APIs and the three-configuration registry.

## Baseline checkpoint

Source starting point: `5740c3c` on main; work continues on `feature/predictor-v2-physical-g3`. The provided specification was initially untracked and is included in this checkpoint. The request manifest hash is `4e62826280e49359e8cf98d85154000c1583d63474be83a2e9a429da9bde72c9`; model contract is `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`.

Actual baseline commands and results:

- `.venv/bin/python -m pytest -q -rs --junitxml=artifacts/phase3/cloud-v2/implementation-audit/baseline-tests.xml`: **227 passed**, zero skipped, 41.10 s. This command ran before the scope override; its evidence was copied without alteration into `artifacts/phase3/physical-v2/implementation-audit/`.
- `.venv/bin/python -m ruff check . --no-cache`: passed.
- `.venv/bin/python -m mypy --cache-dir /tmp/predictor-v2-mypy`: passed, 65 source files.

The baseline audit verifies the exact 60/20/20 assignment, disjoint IDs, complete 100-ID manifest coverage, frozen network memberships, and dataset SHA. Original logs and an audit JSON remain in the ignored artifact directory above.

## Ordered implementation checkpoints

1. Specification/scope and baseline audit (this checkpoint).
2. Strict positive-allowlist partitioning, outcome/feature separation and leakage/provenance tests.
3. Versioned V2 runtime, analytical transport, low-dimensional compute fitting, training-only uncertainty and V1 compatibility.
4. Training-only grouped diagnostics, sealed validation predictions, one candidate selection and honest numerical/ranking/coverage gates.
5. Final tested handoff and hashes. No N4/N9 prediction/evaluation artifacts in this phase.

Regular commits are required after tested phases. Data-dependent gates and final calibration are **not passed or frozen yet**.
