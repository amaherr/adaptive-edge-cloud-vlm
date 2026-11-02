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

Both hosts report the same package versions: PyTorch `2.7.1+cu128`, Transformers `4.52.4`, tokenizers `0.21.4`, Pillow `11.2.1`, and NumPy `2.4.6`.

A further diagnostic sent the edge CPU's projected BF16 tensor to a temporary cloud CPU text server, using the same smoke image and prompt. The 32 generated tokens matched the cloud CPU and GPU results exactly and differed from edge-local CPU `edge/full` at token 20. The raw edge output is archived at `artifacts/phase1/cpu-reference-v1/split_edge_vision_cloud_cpu.txt`. This narrows the mismatch to edge-local text inference or another host-specific runtime effect after the projection boundary. The cloud GPU server was restored after this diagnostic.

Three additional edge-local CPU `edge/full` runs returned identical 32-token sequences, all matching the earlier corrected smoke run and the archived physical campaign's edge row. Their raw output is archived at `artifacts/phase1/cpu-reference-v1/edge_repeat.txt`. Thus five observed edge CPU runs agree, while six observed cloud CPU runs (one initial run plus five repeats) agree on the other sequence. The observed divergence is repeatable, not random.

## Gate decision and next checks

P1 diagnostic isolation **passes**: transport/decomposition is exact for all 12 archived requests, each CPU placement is internally repeatable for the divergent request, and the cross-host difference is localized to edge-local text inference or another host-specific effect after the projection boundary. The exact low-level kernel or instruction responsible is unproven. The original strict three-path token-equivalence gate remains **failed** at 11/12. The next phase must define and freeze separate transport and cross-placement correctness criteria before physical profiling. Do not change the frozen model contract or silently relax the old gate.
