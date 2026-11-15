# Reproduce official scoring

The evaluation uses the **unmodified** upstream source, not a Python 3 rewrite. The primary predictions are exact saved `output_text` values. The existing adapter `scripts/run_official_vqa_eval.py` records the actual upstream source/input hashes in every result. No GPU, model loading or university cloud connection is needed.

Required inputs are the committed evidence archive, the frozen manifest, and the same 100 image paths (the existing manifest exporter validates their presence). Keep original questions and annotations in ignored data/artifact directories. The official source JSONs and the runtime are not redistributed in this results directory.

1. Fetch [official VQA validation annotations](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip) and [questions](https://visualqa.org/download.html). The annotation ZIP SHA-256 is `0caae7c8d1dafd852727f5ac046bc1efca9b72026bd6ffa34fc489f3a7b3291e`. Extracted annotation JSON must hash to `0564650e712b34c246e4ad142894692bf2e45dee8e9cc6ebb2cedbe85731c4bf`; questions JSON to `f34d9c9909dea76700d361baddd63dd47b4494945097459b542d9fe840244811`.
2. Fetch the pinned evaluator checkout:

```bash
git clone https://github.com/GT-Vision-Lab/VQA.git artifacts/official-vqa-g2/VQA
git -C artifacts/official-vqa-g2/VQA checkout a013f0043c1e2cdc995922dfe257f7149aa9af06
```

3. Use a Python 2.7 interpreter. This run used portable [PyPy 2.7.18 / 7.3.15](https://downloads.python.org/pypy/pypy2.7-v7.3.15-linux64.tar.bz2), extracted under `/tmp`, with archive SHA-256 `e857553bdc4f25ba9670a5c173a057a9ff71262d5c5da73a6ddef9d7dc5d4f5e`, verified against [PyPy's official checksum page](https://pypy.org/checksums.html). No system Python or existing virtual environment was replaced.
4. From the repository root, with original annotations stored at the shown path:

```bash
.venv/bin/python results/physical-g2-20261110/official_vqa/score_saved_predictions.py \
  --python2 /tmp/pypy2.7-v7.3.15-linux64/bin/pypy \
  --vqa-root artifacts/official-vqa-g2/VQA \
  --annotations artifacts/official-vqa-g2/v2_mscoco_val2014_annotations.json \
  --questions data/raw/vqa/v2_OpenEnded_mscoco_val2014_questions.json \
  --work-dir artifacts/official-vqa-g2/scoring
```

The orchestrator verifies source/manifest hashes, creates a matching official annotation/question subset, checks prompt/reference/image-ID alignment, validates exported plans, and evaluates every block/placement/repetition. Existing outputs are reused only after checking their input/evaluator hashes. The leading-yes/no and lowercase-only files are generated under explicitly diagnostic names; the primary raw files are not rewritten.

Then generate the paired report, tables and figure using Python 3 with NumPy/Matplotlib:

```bash
MPLCONFIGDIR=/tmp/g2-matplotlib /usr/bin/python3 \
  results/physical-g2-20261110/official_vqa/analyze_quality.py
```

The analysis uses the ignored subset at `artifacts/official-vqa-g2/scoring/subset/annotations.json`; the scoring command creates it. It verifies every exported prediction against the archived raw answer, all 81 official result hashes, the 8,100 provisional scores, within-placement repeatability, and per-question score stability. The two original subset metadata hashes are committed in `subset_provenance.json`.

Synthetic upstream regression checks (actual unmodified evaluator, not an imitation):

```bash
/tmp/pypy2.7-v7.3.15-linux64/bin/pypy \
  results/physical-g2-20261110/official_vqa/check_upstream_cases.py \
  --vqa-root artifacts/official-vqa-g2/VQA
```

These cover the unanimous-reference case-handling branch, mixed-reference lowercasing/digits, sentence mismatch, and leave-one-annotator consensus scores of 30/60/90/100%. All nine passed.

Exporter metadata retains `pending_official_vqa_evaluator` as its export-time metric label; `evaluation_index.json` links each prediction's SHA to its completed official evaluation. This does not mean the evaluation is still pending.
