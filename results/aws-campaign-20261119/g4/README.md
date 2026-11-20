# G4: the r2 selector against the measured oracle (evaluation cells)

**G4 passed (20 November 2026).** The rules come from the [G4 specification](../../../docs/predictor-v2/g4-specification.md). Its machine-readable copy, [g4_specification.json](g4_specification.json) (SHA-256 `b2242d7f…5761`), was frozen in commit `8af54cc`, before any evaluation cell was profiled.

The selector is the existing `scheduler.selector.select` over all five configurations, fed with the frozen `r2_component` predictions (G3). The evaluation cells E01–E05 had never been used for fitting or for any selection choice.

## How it was run

1. **Best static configuration**, chosen on the validation cells only ([best_static.json](best_static.json)): `cloud/raw_image`, with a median of 830 ms.
2. **Freeze:** `run_g4_r2.py freeze` pinned:
   - the frozen calibration (`16c20438…f05a`);
   - the G3 decision;
   - the best static choice;
   - the SHA-256 of the seven selection-code files.
3. **Evaluation profiles:**
   - **Source:** `derive_g2v2_tables.py profiles --splits evaluation --unseal-evaluation`, run from a worktree of `0948aa8`. That commit's `src/` equals the frozen campaign commit except the two exempt analysis modules, so the reference bundles verified.
   - **Output:** 44 sub-blocks, every one passing the heterogeneous correctness check, kept privately in `~/g2v2plus-r2/eval/`.
   - **A first attempt failed** before writing any profile: the worktree lacked the git-ignored dataset images. It was rerun after linking `data/raw`.
4. **Evaluation:** `run_g4_r2.py evaluate`, run once. It checked the specification hash, the code hashes and the calibration. It wrote [evaluation.json](evaluation/evaluation.json) and [decisions.jsonl](evaluation/decisions.jsonl): one record per request-cell, with every prediction, the decision, the observed latencies, the oracle and the regret of each policy.
5. **After the decision**, two more outputs:
   - official VQA scoring on the evaluation IDs ([official_vqa_evaluation.json](official_vqa_evaluation.json));
   - the cell-level oracle for the evaluation cells ([oracle_cells.csv](oracle_cells.csv), [oracle_cells.json](oracle_cells.json)).

## Gate

| Condition | Required | Result |
| --- | --- | --- |
| Median relative regret | ≤ 10% | **0.00%** (P90 0.00%, mean 0.06%, 95% CI 0.02–0.11%) |
| Complete blocks (cells) | ≥ 10 | **22 of 22** (630 of 630 request-cells complete) |
| Network holdout | every scored condition in E | **E01–E05 only** |
| No-feasible decisions, unsafe selections | 0 | **0, 0** |
| Decision = minimum of its own predictions | always | **yes** |

No selected prediction was extrapolated.

## Selector against the oracle and the static policies

Over 630 request-cells. The measured oracle chose `cloud/raw_image` in 491 and `edge/full` in 139.

| Policy | Top-1 agreement | Mean regret (95% CI) | Max regret | Mean latency | Selector − policy (95% CI) |
| --- | ---: | --- | ---: | ---: | --- |
| **Selector (five configurations)** | **98.1%** | **0.6 ms (0.2–1.1)** | 64 ms | **775 ms** | |
| Selector (four lossless) | 98.1% | 0.6 ms (0.2–1.1) | 64 ms | 775 ms | identical choices |
| Always cloud = best static | 77.9% | 43.2 ms (33.7–53.5) | 695 ms | 817 ms | **−42.6 ms (−53.1 to −33.1)** |
| Always edge | 22.1% | 531.8 ms (451–636) | 4,217 ms | 1,306 ms | −531.2 ms (−635.7 to −450.8) |
| Always int8 | 0.0% | 631.0 ms (597–661) | 1,243 ms | 1,405 ms | −630.4 ms (−660.3 to −596.9) |

**Choices.** The selector chose `cloud/raw_image` in 483 request-cells and `edge/full` in 147, and never a split configuration.

**Against the strongest static policy.** Always-cloud is also the best static configuration chosen on validation. The selector is 42.6 ms faster per request on average, or 5.2% of always-cloud's mean latency, and the paired CI excludes zero. Always-cloud's mean relative regret is 5.2% (CI 3.7–7.0%); the selector's is 0.06%.

**By load state:**

| Load state | Top-1 | Mean regret | Max regret | Selector choices |
| --- | ---: | ---: | ---: | --- |
| L0 / C0 | 98.8% | 0.4 ms | 33 ms | cloud 159, edge 6 |
| L0 / C2 | 94.0% | 2.1 ms | 64 ms | cloud 28, edge 122 |
| L1 / C0 | 100% | 0.0 ms | 0 ms | cloud 165 |
| L1 / C2 | 99.3% | 0.0 ms | 0.1 ms | cloud 131, edge 19 |

**By condition:** top-1 is 96.7–100% and mean regret 0.0–1.4 ms in every condition, E01–E05.

**The selector adapts per request, not only per cell.**
- At L0 / C0 it sends 6 small-image requests to the edge.
- At L1 / C2 it sends 19 to the edge, in cells where the cloud wins on average.
- In 7 of the 22 cells, the oracle itself picks different configurations for different requests. The selector follows those splits closely; for example, at E05-L1-C2 the oracle chose cloud 17 times and edge 13, and the selector made exactly the same choices.

The remaining errors are near-ties at heavy cloud load (E04-L0-C2: 7.2 ms mean regret, E01-L0-C2: 2.9 ms).

**Against the cell-level oracle.**
- In the cell-level oracle (medians, bootstrap CIs) the cloud wins 18 evaluation cells and the edge 4, with no ties.
- The selector's majority choice equals the winner in **22 of 22** cells.

## Quality on the evaluation IDs

Official VQA accuracy, with the pinned evaluator, over 30 evaluation IDs:
- **70.67% lowercased** (primary) and 47.33% unchanged, **identical for all five configurations**;
- 0 questions differ between any two configurations.

Parity holds on all three splits. Selecting a configuration therefore never changes the answer, and the selector's latency gains come at no cost in quality.

## Meaning and limits

- **What G4 shows:** on unseen networks and requests, the frozen predictor plus the selector comes within 0.6 ms of the per-request measured oracle on average. It also beats every static policy, including the best static one, with paired CIs that exclude zero.
- **The gain is concentrated:** under heavy cloud load at no edge load (L0 / C2), the selector moves most requests to the edge. Elsewhere it mostly confirms the cloud. This is the pattern the baseline report predicted.
- **These are offline replays** of measured campaign states. The load levels are inputs given by the experiment, not sensed. A pass does not enable live automatic selection by itself; sensing, switching overhead and changing conditions are phases 5 and 6 (G5).
- **Exposure:** the evaluation cells' aggregate winners were exposed on 19–20 November ([amendment A3](../../../docs/protocols/g2-v2-amendment-a3.md)). No choice in r2 or G4 used them, but the thesis discloses the exposure as a threat to validity.

## Files

| File | Content |
| --- | --- |
| [g4_specification.json](g4_specification.json) | Frozen specification (hashes of every input and the code) |
| [best_static.json](best_static.json) | Best static configuration from the validation cells |
| [evaluation/evaluation.json](evaluation/evaluation.json) | Gate, policies, paired differences and breakdowns |
| [evaluation/decisions.jsonl](evaluation/decisions.jsonl) | One record per request-cell |
| [official_vqa_evaluation.json](official_vqa_evaluation.json) | Official VQA accuracy and parity on the evaluation IDs |
| [oracle_cells.csv](oracle_cells.csv), [oracle_cells.json](oracle_cells.json) | Section 9 cell oracle for the evaluation cells |

The per-attempt evaluation oracle table (`oracle_attempts.jsonl`, SHA-256 `73c809e5…766d`) and the evaluation profiles stay private in `~/g2v2plus-r2/eval/`.
