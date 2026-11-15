# G2 stage analysis

Read [stage_analysis.md](stage_analysis.md) for the complete findings and their relationship to the thesis. It contains the nine-condition bottleneck table, detailed encoder/projection/serialization/transfer/prefill/decode budgets, memory evidence, paired uncertainty, and adaptive-placement implications.

Useful files:

- [stage_summary.csv](stage_summary.csv): 27 mean stage budgets and end-to-end median/p95; server/exchange parent timers are separate and must not be summed with their children.
- [stage_shares.csv](stage_shares.csv): per-stage mean, p95, mean-latency share and measurement availability. Cloud-only vision/projection/image-prefill are inseparable in the archive and are explicitly marked.
- [placement_comparison.csv](placement_comparison.csv): condition winners, paired differences and 95% request-cluster bootstrap intervals, plus per-request winner counts.
- [request_stage_samples.csv](request_stage_samples.csv): every measured request's additive decomposition; zeros for unavailable/not-applicable categories are bookkeeping entries.
- [transfer_accounting.csv](transfer_accounting.csv): payload and protocol bytes, transfer residual, bytes/rate reference, and non-additive socket diagnostics.
- [memory_summary.csv](memory_summary.csv), [replicate_drift.csv](replicate_drift.csv), and [task_feature_associations.csv](task_feature_associations.csv): system/task evidence.
- [adaptive_implications.json](adaptive_implications.json) and [descriptive_crossover.json](descriptive_crossover.json): retrospective opportunity and descriptive threshold calculations; neither is an evaluated online adaptive policy.
- [VALIDATION.md](VALIDATION.md): checks and reproduction commands; [analysis_validation.json](analysis_validation.json): provenance, versions, seeds and timing closure.

Figures are supplied as PNG for quick viewing and SVG for thesis editing/export:

1. [Exclusive stage decomposition](stage_decomposition.png)
2. [Placement latency by bandwidth and RTT](placement_latency.png)
3. [Transfer and paired uncertainty](transfer_and_uncertainty.png)
4. [Memory and vision timing](memory_and_vision.png)

All original run results remain one directory above this analysis.
