# Completed physical G2 baseline — 10–11 November 2026

All **8,100 measured requests succeeded**, plus 27 warmups. All nine controlled network blocks passed the user-approved `physical-heterogeneous-v2` criterion. The campaign completed at **2026-11-11T05:20:58+00:00** (07:20:58 Cairo), in **15.20 hours** including preflight. Source commit: `3c38be2a68d902dbbf286621a593832aa2149978`.

## Official VQA quality follow-up

Read [the official scoring and quality diagnosis](official_vqa/quality_report.md). All 81 run/repetition exports score **2.0%** with the untouched official evaluator on the saved raw answers (100 unique questions). All three paired placement score differences are **0 percentage points**; that floor-level agreement does not establish semantic equivalence. The report explains the 3% provisional score, verbose-answer mismatch and `Hotel` case-handling discrepancy. A clearly separate leading-yes/no diagnostic scores 32%; it does not replace the raw baseline.

See [paired quality differences](official_vqa/paired_quality_differences.csv), [per-question answers and scores](official_vqa/per_question_quality.csv), and [reproduction commands](official_vqa/REPRODUCE.md). Original inference artifacts and gate results are preserved.

## Stage-level analysis and thesis findings

Start with [the detailed stage analysis](analysis/stage_analysis.md). It explains the bottleneck in every network condition, includes four scientific figures and paired uncertainty, and relates the measured crossover and memory effects to adaptive placement. The analysis covers all 8,100 measured rows; the original evidence is unchanged.

Cloud-only wins N1–N4 and N7–N9 on mean latency; the projected split wins N5–N6. The fitted crossing near 39 Mb/s is descriptive, not a validated live switching threshold. A retrospective condition-wise choice saves about 3.5% against always-cloud on this equally weighted grid, before sensing/switching costs.

- [Stage budgets CSV](analysis/stage_summary.csv): all 27 network/placement budgets.
- [Placement comparison CSV](analysis/placement_comparison.csv): paired request-level confidence intervals and winner counts.
- [Analysis validation](analysis/analysis_validation.json): archive hashes, row counts, timing closure and software versions.
- [Reproduction script](analysis/analyze_stages.py): regenerate all tables and figures without a GPU or cloud connection.

## Original run evidence

- [summary.json](summary.json): completion, aggregate placement timings, counts and correctness outcomes.
- [block_summary.csv](block_summary.csv): latency, quality and framed-byte summaries for all 27 block/placement combinations.
- [evidence.tar.gz](evidence.tar.gz): original request journals, plans, manifests, profiles, GPU samples, pre/post network measurements, validation reports, preflight logs and hash-bound same-device correctness references.
- [archive_inventory.json](archive_inventory.json): SHA-256 and size of every archived file.
- [SHA256SUMS](SHA256SUMS): archive checksum.

Verify and restore from the repository root:

```bash
(cd results/physical-g2-20261110 && sha256sum -c SHA256SUMS)
tar -xzf results/physical-g2-20261110/evidence.tar.gz
```

The archive preserves the original repository-relative artifact paths and byte content. Recorded absolute paths identify the original hosts; using another machine requires explicit path mapping. Model weights, raw dataset images, executable caches, PID and lock files are excluded. The original complete working artifacts remain locally under `artifacts/unattended/site-preflight-20261110/`.

The approved criterion requires exact same-device path parity, stable per-placement reference outputs, repeatability, matching per-request task scores, exact transport and valid timing/network evidence. **Legacy cross-GPU exact-token gates remain false** and are preserved in every block report; differing wording is not represented as identical output. The provisional VQA scores are not an official VQA evaluation or a general accuracy claim. Byte accounting covers application frames, excluding TCP/IP/Ethernet overhead, retransmissions, probes and SSH. The edge model used WDDM shared GPU memory; success does not establish dedicated-VRAM feasibility.
