# Physical correctness contract v1

This contract applies prospectively to physical runs using the pinned Moondream2 model, frozen dataset manifest, and three configurations in `physical_pair_v1.yaml`. Record the Git SHA, model artifact SHA-256, model fingerprint, dataset SHA-256, host, device placement, seed, and output token IDs for every compared request. A changed model or generation contract requires a new contract and campaign revision.

## Transport and decomposition

For each identical request and the same cloud text placement, compare `cloud/raw_image` with `distributed/projected/synchronous`. Require both to finish without protocol/model errors and to produce exactly identical token IDs and decoded text. Any mismatch fails the transport gate and requires investigation before performance data is accepted. Report bytes and projected tensor dtype/shape for every split request.

## Cross-placement generation

Compare `edge/full` with the cloud-backed paths separately. Report exact token equality and decoded-text equality per request; do not treat a cross-device token mismatch as a transport failure when the two cloud-backed paths match. For a divergent request, repeat each affected placement at least five times with the same input and record whether tokens are stable within each placement. An unstable placement fails the reproducibility gate.

For labeled short-answer tasks, compute the same frozen task score for every placement and report both per-request and aggregate score differences. A placement is task-quality equivalent on a frozen set only if its per-request normalized answer and aggregate task score match the reference. For unscored descriptive prompts, report text and token differences; they cannot establish task-quality equivalence without a separately frozen evaluation rule. When official VQA evaluation is used, apply the official scorer to all placements and report paired accuracy differences with uncertainty. Do not replace this with visual inspection or post hoc normalization.

## Decision rules

| Observation | Decision |
| --- | --- |
| Cloud raw and projected tokens match on every frozen request | Transport/decomposition PASS |
| Any cloud raw/projected token mismatch | Transport/decomposition FAIL |
| A placement changes tokens across identical repetitions | Reproducibility FAIL |
| Stable cross-placement token difference with equal frozen task scores | Exact-token FAIL; task-quality PASS on that scored set |
| Stable cross-placement difference on an unscored request | Exact-token FAIL; task-quality UNDETERMINED |
| Different model fingerprint, manifest hash, or generation contract | Comparison INVALID |

The 2026-11-02 edge CPU/cloud GPU synthetic campaign has transport/decomposition PASS (12/12) and strict three-path exact-token FAIL (11/12). `correctness-001` is a descriptive prompt; its task-quality status remains UNDETERMINED under this contract. The later clean edge GPU/cloud GPU campaign has strict three-path exact-token PASS (12/12) on the same frozen manifest. These are separate device placements and separate gate results. Freeze this contract before using it for the next physical campaign and obtain edge/coordinator review of the task-quality criteria.
