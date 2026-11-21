# Reproducing the thesis results

Every table and figure in the thesis can be regenerated from the archived data with the commands below. None of them needs a GPU or AWS. Measurements came from two AWS campaigns:
- **G2 v2+:** 18–19 November 2026;
- **G5:** 20–21 November 2026.

What follows covers the analysis of those measurements, not rerunning the campaigns. The campaign runbooks and grids are in [runbooks/](runbooks/) and [configs/profiles/](../configs/profiles/).

## 1. Source and identities

Each result is pinned by a git tag, and every tag message records the image digest or the protocol and specification hashes. To check out a result's exact source: `git checkout <tag>`.

| Tag | What it pins |
| --- | --- |
| `g2v2plus-campaign` | G2 v2+ campaign source (`d5e116c`), image `sha256:4049c069…`, protocol `a972ecdc…` |
| `g3-r2-specification`, `g3-r2-candidate`, `g3-passed` | Predictor r2: specification before fitting, frozen candidate with sealed predictions, G3 decision |
| `g4-specification`, `g4-passed` | G4 specification frozen before the evaluation cells were opened; G4 result |
| `g5-campaign`, `g5-freeze`, `g5-passed` | G5 campaign source (`6cf005f`, image `sha256:2cd3bd95…`), freeze record, G5 result |
| `final-analysis` | Final tables and figures |

## 2. Environments

- **`.venv`:** Python 3.12 with the package installed in editable mode (`pip install -e .[dev]`). It runs everything below.
- **The official VQA evaluator:**
  - GT-Vision-Lab/VQA `a013f00`, under PyPy 2.7 (`scripts/vqa/run_official_vqa_eval.py` documents how);
  - the evaluator checkout lives in `artifacts/official-vqa-g2/`.
- **Moondream's pinned tokenizer and Pillow** (`requirements/model-cu128.lock`) are needed only to rebuild the task features.

## 3. Archived data

| Data | Local path (git-ignored) | Integrity |
| --- | --- | --- |
| The whole artifact bucket, including both campaigns' checkpoints and the final copies of both hosts' data volumes | `artifacts/aws-s3-mirror/` | [archive inventory](../results/archive-20261121/README.md): every object checked against S3 by key, size and MD5 |
| G2 v2+ campaign folder (edge data volume) | `artifacts/aws-s3-mirror/cloud-v1/edge/ebs-final/g2v2plus/campaign/` | [SHA256SUMS](../results/aws-campaign-20261119/SHA256SUMS) (5,921 files) |
| G5 campaign checkpoint | `artifacts/g5-aws/campaign/` | [SHA256SUMS](../results/g5-aws-20261120/campaign/SHA256SUMS) (2,721 files) |
| Predictor partitions (training rows, validation inputs, validation targets) | `~/g2v2plus-r2/` (WSL) | hashes in [partition_audit.json](../results/aws-campaign-20261119/predictor_r2/candidate/partition_audit.json) |
| G4 evaluation profiles | `~/g2v2plus-r2/eval/` (WSL) | rebuilt by the command in section 4 |

**Rebuilding derivations from the G2 v2+ campaign.** Re-derive profiles, partitions or oracles from a worktree of `0948aa8` (`git worktree add ../frozen 0948aa8`). Its `src/` matches the campaign commit except two hash-exempt analysis modules, so the reference bundles verify. Later source changes alter the verifier's source hash.

**Container paths.** The campaign plans store container paths, so map `/app` to the repository and `/artifacts` to `artifacts/aws-s3-mirror/cloud-v1/edge/ebs-final`, for example with links in WSL.

## 4. Commands per result

**Campaign checks and quality (G2 v2+):**

```bash
.venv/bin/python scripts/campaign/check_campaign_correctness.py ...   # see results/aws-campaign-20261119/derived/README.md
.venv/bin/python scripts/vqa/score_campaign_vqa.py ...                 # training and validation IDs
.venv/bin/python scripts/campaign/derive_g2v2_tables.py profiles --splits training ...
.venv/bin/python scripts/campaign/derive_g2v2_tables.py oracle ...
```

The full argument lists are in the [derived checks](../results/aws-campaign-20261119/derived/README.md).

**G2 v2+ baseline report**, which writes every table and figure in `results/aws-campaign-20261119/baseline/` (about 50 s):

```bash
.venv/bin/python scripts/campaign/report_g2v2plus_baseline.py all \
  --campaign artifacts/aws-s3-mirror/cloud-v1/edge/ebs-final/g2v2plus/campaign \
  --oracle artifacts/g2v2plus-aws-campaign-derived/oracle-training \
  --output results/aws-campaign-20261119/baseline
```

**Predictor r2 and G3.** Fit and review reproduce the calibration hashes in the [r2 record](../results/aws-campaign-20261119/predictor_r2/README.md):

```bash
.venv/bin/python scripts/prediction/run_predictor_r2.py fit --partitions ~/g2v2plus-r2/partitions --output <new dir>
.venv/bin/python scripts/prediction/run_predictor_r2.py review --fit <new dir>
```

`validate` refuses to run where a decision already exists. To re-score, give it a new `--output`. The decision it reaches is fixed by the committed predictions and targets.

**G4:**

```bash
.venv/bin/python scripts/prediction/run_g4_r2.py evaluate \
  --specification results/aws-campaign-20261119/g4/g4_specification.json \
  --specification-sha256 b2242d7fbbfab72080ec390bc2096e7c2cf69620266700baeb08006dee025761 \
  --calibration results/aws-campaign-20261119/predictor_r2/candidate/calibration_r2_component.json \
  --campaign artifacts/aws-s3-mirror/cloud-v1/edge/ebs-final/g2v2plus/campaign \
  --profiles ~/g2v2plus-r2/eval/profiles --features artifacts/g2v2plus-aws-campaign-derived/task_features \
  --output <new dir>
```

The evaluation profiles come from the `0948aa8` worktree:

```bash
PYTHONPATH=<worktree>/src .venv/bin/python scripts/campaign/derive_g2v2_tables.py profiles --splits evaluation --unseal-evaluation ...
```

**G5:**

```bash
.venv/bin/python scripts/campaign/analyze_g5.py --campaign artifacts/g5-aws/campaign \
  --calibration configs/g5/calibration_r2_component.json \
  --features configs/g5/task_features-short.json --output <new dir>
```

**Final analysis and thesis figures:**

```bash
.venv/bin/python scripts/campaign/final_analysis.py g5 --campaign artifacts/g5-aws/campaign --output <dir>
.venv/bin/python scripts/campaign/final_analysis.py figures --campaign artifacts/g5-aws/campaign \
  --partitions ~/g2v2plus-r2/partitions \
  --predictions results/aws-campaign-20261119/predictor_r2/candidate/validation_predictions.jsonl --output <dir>
```

The six supplementary figures (load effects, static regret, G3 error distribution, G4 regret by load state, bytes and latency, network verification):

```bash
.venv/bin/python scripts/campaign/thesis_figures.py   --campaign artifacts/aws-s3-mirror/cloud-v1/edge/ebs-final/g2v2plus/campaign   --partitions ~/g2v2plus-r2/partitions --output results/final-analysis-20261121
```

**Tests:** `.venv/bin/python -m pytest -q`; they need no GPU and no data.

## 5. Protocol and amendments

- **Protocol:** [g2-v2-protocol.md](protocols/g2-v2-protocol.md), pinned by SHA-256.
- **Amendments,** in order:
  - A2a and A2b (inside the protocol file, before the campaign);
  - [A3](protocols/g2-v2-amendment-a3.md): the V/E exposure;
  - [A4](protocols/g2-v2-amendment-a4.md): G5;
  - [A4a](protocols/g2-v2-amendment-a4a.md): the edge-probe shape;
  - [A4b](protocols/g2-v2-amendment-a4b.md): background cloud sensing.
- **Specifications:** [r2](predictor-v2/r2-specification.md) and [G4](predictor-v2/g4-specification.md).

Every amendment was written before the data it governs.
