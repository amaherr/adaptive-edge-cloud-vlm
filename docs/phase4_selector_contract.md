# Phase 4 selector and oracle contract

The selector is an offline, pure decision function in `src/adaptive_vlm/scheduler/selector.py`. It consumes exactly one `CandidateSnapshot` for each of the three stable `ConfigurationId` values. A snapshot contains the existing capability result, a Phase 3 prediction or `None`, a health flag, and optionally a validated quality score. `SelectionConstraints` contains an optional latency SLA, quality floor, and edge memory budget. No measured outcome or model inference enters `select`.

## Selection rules

1. Enumerate configurations in registry order: edge/full, cloud/raw image, distributed/projected synchronous. Missing or duplicate snapshots are an input error.
2. Exclude unavailable executors, unhealthy endpoints, missing/invalid/nonfinite/nonpositive predictions, unknown or below-floor quality when a floor is requested, and unknown or excessive edge memory for edge/split when a memory budget is requested. Every exclusion remains in `CandidateDecision` with its reason and full prediction. The existing `CapabilityRegistry` supplies model fingerprint, remote path/readiness, loaded policy, and memory checks; the selector still validates its own inputs.
3. Choose the lowest predicted total latency from feasible candidates. Registry order breaks exact ties. If an SLA is set and no feasible prediction meets it, choose the fastest safe candidate with status `sla_unmet`; the status is visible to the caller. If no candidate is feasible, return `selected_configuration_id=null` and `no_feasible_candidate`. The selector never silently dispatches.

`predicted_oracle(decision)` independently computes the minimum over the logged feasible predictions. It checks the ranking operation, while the synthetic tests check eligibility reasons. An SLA cannot change the minimum because every prediction meeting the SLA is lower than every prediction above it. SLA values are predictions; the retrospective evaluator separately counts observed SLA violations.

The decision's `to_dict()` is a versioned JSON record of the selected ID, status, constraints, and all three candidate snapshots, including each complete prediction. Host/model/placement/task/network coverage is enforced in the Phase 3 predictor. A live integration must capture capability and health **before** selection and must not reuse stale snapshots.

## Retrospective measured oracle

`evaluate-selector` reads Phase 3 held-out `predictions.jsonl` and a frozen oracle specification. The specification pins the predictions file SHA-256, calibration ID, holdout request IDs, sections, static predecision availability/health, constraints, bootstrap seed, evidence scope, and a safety-evidence flag. The file hash is checked before any output is written. For a physical campaign, write a new specification and freeze its hash before inspecting its measured winner or regret. Set `safety_evidence_verified=true` only after recording real remote capability, health, and memory checks for the declared availability snapshot. The provided local specification is a diagnostic reuse of the already inspected Phase 3 holdout, not an independent fresh performance test.

Rows are paired by `(section, request_id, network_bucket_id, task_bucket, condition_block_id)`. For new profiles, `condition_block_id` comes from the benchmark repetition number and is propagated through `profile-runs` and `calibrate-predictor`; old Phase 3 rows default to block `0`. Every feasible candidate in a block must have a successful positive finite observation. Otherwise the block is marked incomplete, with no measured winner or regret. A failed observation never makes that candidate infeasible. Phase 3's outcome-derived `failed_trial` prediction is rejected outright because it could leak an outcome into the decision.

The measured oracle is the minimum observed matched latency among feasible candidates in a complete block. It is used only **after** all runs to evaluate the decision; it cannot route the same live request. `decisions.jsonl` preserves predictions, feasibility reasons, measured winner, selected observation, regret in milliseconds and fraction, and observed SLA status for each block. `evaluation.json` reports paired/incomplete/no-feasible counts, exclusions, top-1 agreement, median regret, paired-block bootstrap 95% intervals, and per-condition p10/p90 oracle latency when at least two repeated blocks exist. The bootstrap resamples complete paired blocks with a frozen seed. With fewer than ten paired blocks, intervals are explicitly descriptive and cannot support a performance claim.

The full G4 gate is evaluable only for a declared `physical-controlled` campaign with verified safety evidence, at least ten complete paired blocks, and a `network_holdout` block. The gate then requires median regret at most 10% and predicted-oracle agreement. The evaluator verifies the boolean and calculation, while the experimenter must substantiate the safety evidence in the physical campaign records. This Phase 4 command always reports `automatic_activation_allowed=false`; a passing G4 report alone does not override G3 or future live integration checks.

## Input and output shape

```bash
.venv/bin/adaptive-vlm evaluate-selector \
  --predictions artifacts/phase3/2026-10-26/clean-1/predictions.jsonl \
  --specification configs/oracle/phase4_local_holdout.json \
  --output-dir artifacts/phase4-replay
```

The output directory must be new. Read `evaluation.json` for aggregate metrics and gate flags, then `decisions.jsonl` for each complete decision and any incomplete block. For a new physical campaign, point the command at the **new** Phase 3 prediction file and a new frozen oracle specification with its SHA-256, calibration ID, held-out IDs, availability snapshot and evidence scope. Do not edit the local diagnostic specification to pretend it describes the new host.
