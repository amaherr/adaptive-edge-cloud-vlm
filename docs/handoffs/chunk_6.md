# Chunk 6 handoff

The feasible same-host Chunk 6 gate is complete. One benchmark command runs
edge, cloud, or distributed against the same resolved model/data/generation
config and frozen image-hashed request manifest. It records warm-ups,
randomized measured repetitions, per-attempt raw JSONL including failures,
resumable plan/run metadata, and a summary. The comparison command audits
every measured pair and reports token/text mismatches and task quality.

With pinned CPU BF16 Moondream2, all three policies completed the 12-request
correctness manifest after one warm-up each. The three-way audit found exact
token and decoded-text agreement on all 12 measured pairs, with zero failures.
The synthetic fixture score was 3/9 scored prompts in every policy. A separate
fake-adapter check completed 100/100 measured requests after 10 warm-ups.

The [report](../reports/chunk_6_benchmark.md),
[requirements matrix](../requirements_traceability.md), and
[README benchmark guide](../../README.md#unified-benchmark-and-correctness-runs)
contain results and usage. Raw real-model runs and the comparison JSON are in
ignored `artifacts/chunk6/` working data; preserve or archive that directory
before changing machines. The official VQA v2 importer is ready, but no
public VQA files are present and its provisional in-run score is not official
VQA accuracy. The user confirmed there is no separate cloud host yet, so G4
physical-host correctness and all physical network experiments remain pending.

Chunk 7 can begin with reversible network-shaping tooling and a pilot plan,
but running or interpreting physical edge–cloud conditions requires the cloud
host inventory, address, access, and VQA validation data to be supplied.
