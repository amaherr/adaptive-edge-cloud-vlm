# Quality validation

- The pinned upstream evaluator sources were executed unchanged in a checksum-verified portable Python 2.7 interpreter. Their SHA-256 values are recorded in all 81 official outputs and in `evaluator_provenance.json`.
- Original official question/annotation JSON hashes match the committed dataset-freeze protocol. All 100 prompt strings, question/image IDs and ten human answer multisets align with the frozen manifest.
- Every primary exported answer matches the preserved raw output text. All 8,100 measured predictions were evaluated; 27 warmups were excluded. No failures were removed. The 81 evaluations retain all 100 question IDs each.
- The provisional formula was reproduced for all 8,100 attempts. Text/tokens and official score vectors are stable within each placement across all 27 block/repetition combinations.
- All observed per-question placement score differences are zero, while raw text differences remain 19/18/16 per placement pair (26 IDs in their union).
- Nine synthetic cases passed against the actual official evaluator: the unanimous-reference case branch, mixed-reference normalization, number conversion, sentence mismatch, and leave-one-human consensus behavior.
- Five independent published-output tests passed, checking counts, the `Hotel` discrepancy, score/text distinction, preservation of primary answers and reference-free diagnostic separation, and score summaries.
- The existing five stage-analysis tests passed again after adding the official-quality cross-reference. Repository Ruff checks passed. Generated reports/figures were reproduced and local links checked; original G2 archive bytes and archived quality values remain unchanged.

Independent quality-output checks:

```bash
/usr/bin/python3 -m unittest discover \
  -s results/physical-g2-20261110/official_vqa \
  -p 'test_quality_outputs.py' -v
```

See `REPRODUCE.md` for official evaluator execution and `quality_validation.json` for machine-readable checks. No inference or model/prompt changes were made.

Python 2's native indented JSON includes spaces before some newlines. Those evaluator/inspection output bytes are preserved with their hashes. A path-scoped `.gitattributes` rule permits that generated JSON whitespace rather than reserializing the evidence; source, CSV and Markdown whitespace checks remain active.
