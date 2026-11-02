# Physical CPU reference: host-specific divergence

## Experiment

On 2026-11-02, the cloud host ran the pinned Moondream2 model with both vision and text on CPU. The runner was `scripts/physical_cpu_reference.py` at Git commit `e520538`; the working tree was clean for the full 12-request run. It used the unchanged `physical_pair_v1.yaml` model contract, seed 42, offline verified cache, and frozen correctness manifest. It compared its tokens with the archived physical report from edge commit `ee29a81`.

Raw results are in ignored local artifacts:

- `artifacts/phase1/cpu-reference-v1/single.json`
- `artifacts/phase1/cpu-reference-v1/repeat-001.json`
- `artifacts/phase1/cpu-reference-v1/full.json`
- `artifacts/phase1/physical-v1/phase1_smoke.json` (baseline)

The model contract fingerprint is `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`; the model artifact SHA-256 is `70a7d94c0c8349eb58ed2d9e636ef2d0916960f321ecabeac6354b8ba3d7403f`. The dataset manifest SHA-256 is `c5bc9df4a010dcefbcbb7fd60c9dd1e47dc0e0cf798acff579a1a004006ce8d1`.

## Result

| Request set | Cloud CPU vs edge CPU | Cloud CPU vs cloud GPU raw | Cloud CPU vs cloud GPU projected |
| --- | ---: | ---: | ---: |
| All 12 requests | 11/12 exact | 12/12 exact | 12/12 exact |
| `correctness-001` | Diverges at generated token 20 | Exact | Exact |

The cloud CPU generated the same `correctness-001` token sequence in five consecutive runs. All five matched both archived cloud GPU paths and differed from the archived edge CPU path. The initial single CPU run matched as well. This disproves the simple explanation that CPU placement alone causes the difference. The result is consistent with a host-specific backend, software, or CPU behavior, but the cause is not yet isolated.

## Gate decision and next checks

P1 remains **open**. The transport/decomposition result is exact for the 12 archived requests, but cross-host token equality fails for one request. Collect edge package versions and repeat `correctness-001` on the edge CPU several times. Compare preprocessing and projected tensor bytes across hosts if versions and repeatability do not explain the mismatch. Do not change the frozen model contract or relax the correctness gate based on this diagnostic result.
