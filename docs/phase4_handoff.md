# Phase 4 handoff — selector and offline oracle

Date: 2026-10-26 (Africa/Cairo). Phase 4 implements the selector and retrospective oracle from the [scheduler plan](dynamic_scheduler_implementation_plan.md). **G4 remains open**, as do the physical G2 and failed local G3 gates. The separate cloud machine is not yet available. The implemented interfaces are ready to consume a future physical-host campaign.

## What changed

- Added a pure `select` function that enumerates the three registered configurations, excludes unsafe or unknown candidates with explicit reasons, and chooses the lowest predicted latency with stable tie handling. It records all candidate predictions, capability/health status, SLA result, and constraints. It performs no inference or dispatch.
- Added a separate predicted oracle for deterministic selector checks and an offline measured oracle over matched held-out observations. The latter reports regret, top-1 agreement, exclusions, SLA violations, incomplete/no-feasible blocks, paired bootstrap intervals, and repeated-condition oracle uncertainty. Measured outcomes cannot alter predecision feasibility; outcome-derived failed-trial predictions are rejected.
- Added `evaluate-selector`, the frozen [local oracle specification](../configs/oracle/phase4_local_holdout.json), and synthetic gate fixtures. The Phase 2 profile and Phase 3 prediction pipeline now carry repetition as a condition-block ID, and Phase 3 ranking counts repeated blocks separately. See the [selector contract](phase4_selector_contract.md) for the exact API and JSON behavior.

Implementation commits: `b5527e8` (selector, oracle, specification, CLI, fixtures) and the documentation/follow-up commit recorded in Git history. The code is deliberately outside `adaptive-infer`: its current manual configuration choice remains in force until the earlier gates pass and Phase 5 explicitly wires adaptive dispatch.

## Local diagnostic result

The frozen input is the already inspected [Phase 3 predictions](../artifacts/phase3/2026-10-26/clean-1/predictions.jsonl), SHA-256 `a4abe75aa6704d735b5d35081e78bab500fa03bea561b0219e4b2960f899bc42`, with calibration `2917acf00f8df890e474d4e5d9c0f9ca66555631a298d004702539f5a85b208d`. The [Phase 4 evaluation](../artifacts/phase4/2026-10-26/local-3/evaluation.json) and [decision trace](../artifacts/phase4/2026-10-26/local-3/decisions.jsonl) contain two complete request/condition blocks and all three candidates in each. The selector agrees with the predicted oracle in both blocks. Both selected cloud/raw image; edge/full was the observed fastest in both.

| Measure | Local diagnostic |
| --- | ---: |
| Paired blocks | 2/2 |
| Top-1 agreement with measured oracle | 0/2 |
| Median regret | 514.5 ms; 1.90% |
| Median regret paired bootstrap 95% interval | 337.9–691.2 ms; 1.19–2.61% |
| Incomplete/no-feasible blocks in this replay | 0/0 |
| Full G4 evaluable/passed | No/No |

The 1.90% median is below the proposed 10% regret target, but it is descriptive only: two single-host CPU/loopback requests, one repetition, no network holdout, and no physical memory/health validation cannot establish robust regret or oracle uncertainty. The 0/2 ranking also reflects the already failed G3 local check. The two-point bootstrap interval cannot estimate the tails of a real deployment. No automatic performance selection is enabled.

The [local archive](../artifacts/phase4/2026-10-26/local-3.tar.gz) contains the evaluation, decision trace, frozen oracle specification, Phase 3 predictions/calibration/evaluation, and [per-file hashes](../artifacts/phase4/2026-10-26/local-3/file_hashes.json). Archive SHA-256: `61ffa1eb1b911fc64bacfc58976ac6e7518a1204f65bb0b21e5295402affe502`. Generated `artifacts/` are ignored by Git, so preserve this archive when transferring the project.

## Run and view it now

From the repository root, use a fresh directory:

```bash
.venv/bin/adaptive-vlm evaluate-selector \
  --predictions artifacts/phase3/2026-10-26/clean-1/predictions.jsonl \
  --specification configs/oracle/phase4_local_holdout.json \
  --output-dir artifacts/phase4-replay
```

Open `artifacts/phase4-replay/evaluation.json` for gate flags and aggregate regret/agreement. Open `decisions.jsonl` for each request's complete candidate predictions, reasons, selected ID, measured winner and regret. In Python, construct `CandidateSnapshot` values from `CapabilityRegistry.statuses()` and `Predictor.predict(...)`, then call `select(tuple(snapshots), SelectionConstraints(...))`; inspect `decision.to_dict()` before any future dispatch. A caller must handle `no_feasible_candidate` and `sla_unmet` explicitly.

## When the second machine becomes available

1. Follow [separate cloud PC setup](../README.md#using-a-separate-cloud-pc). Verify host inventory, matching model contract, reachable dual-path server, remote capabilities and the same frozen manifest. Keep fixed-policy benchmark runs as controls. Declare the physical host pair and controlled network buckets in a **new** Phase 2 profile specification; run matching edge/cloud/distributed schedules with repetitions. The resulting profile carries each repetition into a condition block. Measure memory safety and host health before treating a candidate as available.
2. Freeze a **new** Phase 3 training/request/network holdout split and its profile SHA-256 before fitting. Run `calibrate-predictor` against that profile/split and inspect `evaluation.json`. G3 must pass its latency and ranking targets on the fresh holdouts. A failed trial's outcome-derived prediction cannot be used for a Phase 4 decision; collect a valid predecision prediction or mark the block unusable and investigate.
3. Before examining observed winners, copy the local Phase 4 JSON specification under a new name. Replace `predictions_sha256` with `sha256sum NEW_PREDICTIONS.jsonl`, the calibration ID, actual held-out IDs/sections, physical predecision availability and health, and `evidence_scope` with `physical-controlled`. Set `safety_evidence_verified=true` only after archiving the matching remote capability, health, and memory checks. Set an SLA, quality floor or memory budget only when supported by measurements; unknown edge memory with a budget excludes edge/split. Commit the frozen specification before running `evaluate-selector --predictions NEW_PREDICTIONS.jsonl --specification NEW_SPEC.json --output-dir NEW_DIR`.
4. Check every `decisions.jsonl` block for candidate coverage and unsafe exclusions. A full G4 assessment requires at least ten complete paired blocks, a network holdout, median regret ≤10%, and selector/predicted-oracle agreement. Repeated condition blocks produce p10/p90 oracle latency; the paired bootstrap provides uncertainty for aggregate regret and agreement. If G3/G4 fails, revise the model or policy and create a **new** split/campaign rather than retuning against the inspected holdout.

The same selector API and CLI remain valid for the second machine; only the physical campaign data, frozen split/specification and availability snapshot change. Phase 5 can later integrate a proven selector into adaptive execution with explicit decision/retry logs.

## Verification

Synthetic fixtures cover ties, nonfinite/negative predictions, invalid/missing profiles, unloaded policy, health, edge memory, SLA feasible/unmet, unknown quality, zero feasible candidates, outcome leakage, repeated blocks, physical gate conditions, and frozen-file hash rejection. The full suite passed with `159 passed`, zero skipped, including loopback integration. Ruff and strict mypy passed; all six archived file hashes were verified.
