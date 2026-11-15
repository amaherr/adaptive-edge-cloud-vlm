# G2 v2 prompt calibration (15 November 2026)

**Selected template: P1, `{question} Answer with a single word or phrase.`**

This applies sections 3.3 and 3.4 of the [G2 v2 protocol](../../docs/protocols/g2-v2-protocol.md). The rule and the four candidates were committed in `5c5e351` before any calibration answer was generated. The calibration inputs (the 200-question set and the four manifests) were frozen in `d5662f0`. All runs used that commit's inference code.

## Results

All runs: 200 VQA v2 val2014 questions; edge laptop (RTX 3050, 4 GiB); swap residency; BF16; greedy decoding; one warmup and one measured repetition. Answers were exported unchanged and scored with the pinned official evaluator (GT-Vision-Lab/VQA `a013f00`, PyPy 2.7.18 / 7.3.15).

| Template | Eligible | Official accuracy | Yes/no | Number | Other | Mean output tokens | 32-token cap hits | Failed | Median latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 `{question}` | No (G2 reference) | 1.0% | 0.0% | 6.9% | 0.0% | 12.28 | 13 | 0 | 1,534 ms |
| **P1** `… Answer with a single word or phrase.` | Yes | **56.7%** | 50.1% | 57.9% | 61.0% | **1.47** | 0 | 0 | 1,312 ms |
| P2 `… Answer in one or two words.` | Yes | 57.2% | 50.1% | 57.9% | 62.0% | 1.50 | 0 | 0 | 1,279 ms |
| P3 `… Respond with only the answer, in as few words as possible.` | Yes | 52.45% | 50.0% | 61.4% | 51.6% | 1.81 | 0 | 0 | 1,315 ms |

**How the rule was applied:**
1. The best eligible accuracy is 57.2% (P2), which is above the 50% floor.
2. P1 and P2 are within 1.0 percentage point of it.
3. Neither hits the token cap, so the next tie-break, lower mean output tokens, picks **P1** (1.47 against 1.50).

The full record, with hashes, is [selection.json](selection.json). Predictions and official outputs are stored as `<template>.predictions.json` and `<template>.official.json`. Run plans, records and summaries are in `runs/`.

## Diagnostic: the evaluator's case handling

Moondream answers "Yes" and "No" with a capital letter. When all ten human answers agree (for example ten times "yes"), the pinned evaluator compares case-sensitively and scores "Yes" as wrong. It lowercases answers only when the references disagree. G2's [quality report](../physical-g2-20261110/official_vqa/quality_report.md) identified the same evaluator branch. This is why yes/no accuracy sits near 50% for every template.

Lowercasing the same saved answers, a diagnostic that was **not** part of the selection rule, gives:

| Template | Accuracy | Yes/no | Number | Other |
| --- | ---: | ---: | ---: | ---: |
| P1 | 79.2% | 93.8% | 57.9% | 75.0% |
| P2 | 78.7% | 93.8% | 57.9% | 74.0% |
| P3 | 74.45% | 90.9% | 61.4% | 66.6% |

P1 ranks first here too, so the selection doesn't depend on case handling. 79.2% is close to Moondream2's published VQAv2 accuracy. These files are in [diagnostic_lowercase/](diagnostic_lowercase/). How G2 v2 quality is normalized before scoring is a section 9 decision in the protocol, which is still a draft.

## Incident: CUDA failure during the first P3 run

The first P3 run failed after 124 successful requests with `CUDA error: unknown error`. The CUDA context then stayed broken, so the next two requests also failed, and the runner stopped after three consecutive failures. No GPU driver reset was logged in Windows. Its records are kept in [runs/p3-cuda-unknown-error/](runs/p3-cuda-unknown-error/).

The process crash produced a 5.4 GB WSL core dump on C:. That filled the drive and also stopped the local MongoDB service. The dump was moved to `artifacts/prompt-calibration/crash/` on D:.

No answer had been scored at that point, so P3 was rerun in full from a fresh run directory. Core dumps were disabled for the rerun (`ulimit -c 1`; WSL's piped crash handler ignores `ulimit -c 0`), and it completed 200/200. Across the four runs that makes one CUDA context failure in about 730 swap-residency requests.

This matters for the unattended campaign. A broken CUDA context doesn't recover inside the process, so the campaign runner has to restart the edge process and resume. Core dumps have to be disabled so a crash can't fill the system drive.

## Provenance notes

- The run manifests record `git_commit` `d5662f0` with `git_dirty: true`. The uncommitted files at the time were [score_prompt_calibration.py](../../scripts/vqa/score_prompt_calibration.py) and `configs/experiments/physical_vqa_v2.yaml`. Neither is used during inference.
- Source hashes: questions `f34d9c99…44811`, annotations `0564650e…31c4bf` (the same files G2 used).
