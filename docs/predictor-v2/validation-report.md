# Predictor V2 r1 validation report

**G3 FAILED. No selected variant, approved model freeze, final evaluation or live deployment.** AWS and Vultr are excluded by explicit user instruction. All three physical G2 placements remain in the audit and gate denominators. The physical baseline itself completed successfully; this gate concerns the predictor and conservative resource policy.

## Identity and data audit

Campaign: `physical-g2-20261110`; host pair `edge-rtx3050-cloud-rtx2070super-v1`. Edge: RTX 3050 Laptop, 4096 MiB dedicated VRAM, driver 610.62; university PC: RTX 2070 SUPER, 8192 MiB, driver 610.60. Hardware configurations record PyTorch 2.7.1+cu128 on WSL; the cloud GPU uses BF16 emulation. These are recorded campaign identities, not claims of a new hardware probe. Docker image and Terraform instance identity are not applicable to this native physical campaign.

Model revision `9a7d4024050840e001defacec2b00727e89149e6`, weights SHA `70a7d94c0c8349eb58ed2d9e636ef2d0916960f321ecabeac6354b8ba3d7403f`, tokenizer SHA `0512fdcac4a5f9e7746cbefce4a468dc93bf0f93f11e701b84f8b31bff199e9c`. Native BF16, greedy decoding, batch 1, maximum 32 new tokens. Model contract SHA `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`; dataset manifest SHA `4e62826280e49359e8cf98d85154000c1583d63474be83a2e9a429da9bde72c9`.

Fit code commit: `3d702928ee0f598958c2e1d6e6454a5bfd6a7199`. Review/validation code commit: `7924ee2` (full identity in source receipts). Both ran with clean source trees. The source archive SHA is `51201e91cafb3864e0cb70e7aa3ea8aff0408d7ea3ce87773d493fe0a76e34ff`.

Training used exactly 60 frozen request IDs × N1/N3/N5/N6/N8 × three placements × three repeats: **2700 attempted, 2700 successful, zero failed**. Validation used exactly 20 different frozen IDs × N2/N7 × three placements × three repeats: **360 attempted, 360 successful, zero failed**. Every validation table cell therefore has 60 successful measured trials and 20 independent requests. **4680 crossed attempts were excluded.** The other 20 IDs × N4/N9 comprise 360 protected slots, not read or materialized by V2. Warmups were excluded. Exact IDs, block assignments and hashes are in `fit_dataset_manifest.json` and the copied split.

Earlier G2 analysis already inspected N4/N9, so those measurements are not pristine confirmatory data. This pipeline's isolation does not undo that history. The source archive is immutable; validation predictions were sealed before the 360 outcome targets were read, and the Linux target store was resealed afterward. DrvFS did not enforce permission bits; the corrected store is on Linux `/tmp`, and its path is provenance rather than a portable archive dependency.

## Predictor and training

`CalibrationV2`, `PredictionStateV2`, `PredictionV2` and `PredictorV2` in `src/adaptive_vlm/prediction/v2/predictor.py` implement schema 2 alongside unchanged V1 calibration support. `prediction/v2/dataset/` isolates rows/features; `prediction/v2/calibration/` fits them; `prediction/v2/evaluation/` seals, scores and applies the one-shot gate. `selector.py` honors the explicit live eligibility flag. The registry retains all three policies. V2-B was optional and is not implemented or selected.

Total latency is the sum of named non-overlapping compute stages, predicted decode, and (for remote placements) analytical transport. Compute stage equations are `intercept_ms + sum(slope_ms * feature / training_scale)`. Full coefficients and scales are preserved in the calibration, not approximated from validation.

- Edge/full: edge nondecode + predicted decode + client residual. Nondecode includes the edge vision/projector and language prefill.
- Cloud/raw image: client image read + client residual + cloud nondecode including vision/prefill + predicted decode + transport.
- Distributed/projected/synchronous: edge vision/projector + tensor serialization + client residual + cloud nondecode including deserialization/prefill + predicted decode + transport. Server deserialization appears once. Stages lacking separately identifiable source measurements remain combined; no invented prefill/vision split is fitted.

Transport milliseconds = `beta0 + beta_rtt * RTT_ms + 8000 * request_wire_bytes / uplink_bps + 8000 * response_wire_bytes / downlink_bps`. Request framing and response size are training-only priors; raw image bytes are known input metadata. The projected tensor is 2,985,984 bytes before framing. This is application byte accounting, not a claim to model TCP retransmission wire captures. Decode milliseconds = training median ms/token × training median output length; current actual output length is never an input.

| Placement | Decode ms/token | Token prior | Interval half-width ms | Transport intercept ms | RTT coefficient |
|---|---:|---:|---:|---:|---:|
| cloud/raw_image | 21.151309 | 11 | 679.567 | 40.290510 | 1.732279 |
| distributed/projected/synchronous | 21.303086 | 11 | 995.790 | 129.497760 | 0.625220 |
| edge/full | 55.674601 | 11 | 5814.914 | 0.000000 | 0.000000 |

Compute fitting uses nonnegative low-dimensional regression with fixed Ridge alpha 1e-6, per-request median targets, image pixels and prompt tokens; constant stages use medians. Network correction uses only five independent training-block median signed residuals; it does not clip signed residuals or treat 2700 trials as independent network states. All priors, scales, memory requirements and uncertainty come from training alone. Five request-grouped CV folds and five leave-training-block-out diagnostics were saved. The empirical interval half-width is the 95th percentile of each training request's maximum OOF absolute error, doubled for extrapolation; it is not a probabilistic guarantee. OOF/LOBO rows are diagnostic forecasts, not extra independent requests.

## Single validation pass

The gates require median APE ≤15%, P90 APE ≤30%, complete required coverage, acceptable ranking and conservative feasibility for all three placements. All 1080 candidate forecasts were sealed before outcomes. There was no validation refit or feature/hyperparameter revision.

| Model | Block | Placement | Valid requests | Median APE | P90 APE | MAE ms | Bias ms | Interval coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|
| constant_median | N2 | cloud/raw_image | 20/20 | 11.037% | 72.607% | 436.69 | 144.01 | 100.000% |
| constant_median | N2 | distributed/projected/synchronous | 20/20 | 37.206% | 38.418% | 2524.89 | -2524.89 | 100.000% |
| constant_median | N2 | edge/full | 0/20 | — | — | — | — | — |
| constant_median | N7 | cloud/raw_image | 20/20 | 7.687% | 81.365% | 431.24 | 277.93 | 100.000% |
| constant_median | N7 | distributed/projected/synchronous | 20/20 | 2.559% | 10.807% | 182.15 | 35.79 | 100.000% |
| constant_median | N7 | edge/full | 0/20 | — | — | — | — | — |
| v1_recalibrated | N2 | cloud/raw_image | 19/20 | 6.333% | 72.506% | 363.69 | 292.34 | 84.211% |
| v1_recalibrated | N2 | distributed/projected/synchronous | 19/20 | 2.087% | 5.244% | 183.53 | -66.40 | 100.000% |
| v1_recalibrated | N2 | edge/full | 0/20 | — | — | — | — | — |
| v1_recalibrated | N7 | cloud/raw_image | 0/20 | — | — | — | — | — |
| v1_recalibrated | N7 | distributed/projected/synchronous | 0/20 | — | — | — | — | — |
| v1_recalibrated | N7 | edge/full | 0/20 | — | — | — | — | — |
| v2_a | N2 | cloud/raw_image | 20/20 | 6.153% | 30.205% | 203.33 | 80.30 | 95.000% |
| v2_a | N2 | distributed/projected/synchronous | 20/20 | 1.985% | 3.763% | 152.89 | -77.95 | 100.000% |
| v2_a | N2 | edge/full | 0/20 | — | — | — | — | — |
| v2_a | N7 | cloud/raw_image | 20/20 | 7.383% | 33.076% | 218.58 | 49.39 | 100.000% |
| v2_a | N7 | distributed/projected/synchronous | 20/20 | 2.195% | 7.409% | 158.43 | 55.25 | 100.000% |
| v2_a | N7 | edge/full | 0/20 | — | — | — | — | — |

APE and bias above use per-request medians across three repeats; repeat-secondary metrics, diagnostic edge errors and support labels are in `validation_summary.csv` and the full JSON report. Blank errors mean no valid predictions; they do not mean zero error. Signed bias is predicted minus observed. The finite empirical intervals can cover many observations while point estimates still fail the error gate.

V2-A cloud P90 errors are **30.2054797% on N2** and **33.0760471% on N7**: both exceed 30%, including the marginal N2 miss. Distributed meets both error targets on both blocks. Edge/full has zero eligible coverage; diagnostic P90 errors are about 172.76% and 180.45%. These estimates remain visible rather than disappearing from the audit.

V2-A fastest-feasible ranking matches **40/40** request/block conditions, as does constant median. Both always select cloud/raw_image here. Recalibrated V1 ranks **19/40** scheduled conditions, all 19 matching (100% among comparable conditions), with 21 unranked due support restrictions. V1's N7 coverage is zero and one N2 request is outside its support. Its 19/19 comparable agreement must not be presented as complete coverage. Ranking and paired bootstrap evidence are preserved in `candidate_model_comparison.json`; request-cluster bootstrap uses seed 1707 and 2000 resamples. There are only 20 independent validation IDs and two held-out network blocks, not 360 independent experimental units.

## Feasibility, extrapolation and anomalies

Training peak reserved memory plus 10% headroom requires **5027 MiB** for edge/full against 4096 MiB dedicated VRAM. Successful WDDM shared-memory execution is not proof of dedicated-memory safety. Distributed requires 1392.6 MiB edge and 5786 MiB cloud; cloud/raw_image requires 5786 MiB cloud. Thus V2-A covers 240/360 scheduled trials (66.67%) and 240/240 trials in the conservative two-placement resource domain. No shared-memory waiver was authorized, and the all-three-placement coverage gate remains failed. Zero known unsafe feasibility claims under this policy is not an unattended execution guarantee.

No failed attempts, OOMs or timeouts occurred in these selected training/validation rows. Data validators reject nonfinite/negative stage accounting, inconsistent identities and incomplete/duplicate selected schedules; failures cannot silently be imputed as zero. This does not certify excluded rows or future executions. Baseline shared-memory behavior and noisy training network corrections remain limitations.

Training pre-block measured RTT support is 24.068–55.383 ms; N7 is lower-RTT physics extrapolation. V2 labels distances and doubles intervals; actual strict V1 instead rejects departures. Engineering limits are 1–200 Mb/s and RTT 0–250 ms. The offline study holds the independent pre-block probe constant for the controlled block, with an explicitly declared four-hour maximum age. It never uses post-block probes as features. Live freshness is separately limited to 30 seconds and live extrapolation is disabled. No saved validation forecast is live-selection eligible. N9 was intended to probe the high-RTT boundary, but it is not scored here, is not permission to tune, and is already historically inspected.

Measured V2-A `predict()` overhead: median **0.01622 ms**, P95 **0.02565 ms**; this excludes sensing, image tokenization, loading and switching. It is not end-to-end scheduler overhead.

## Gate and thesis implications

**G3 FAILED**; selected variant is null. V2-A fails cloud numerical tails and all-three-placement coverage. V1 fails numerical, coverage and scheduled ranking gates. No `model_freeze_marker.json` is issued. The existing fits are immutable research candidates, not approved deployable models.

Distributed transport is predictable enough to support a lightweight analytical predictor on this hardware/network range. Cloud compute tails and shared-memory edge behavior remain weak points. The 40/40 ranking result does not demonstrate adaptive improvement over a static cloud policy, because the constant baseline also achieves it and the validation states do not change the winner. These data alone do not establish useful adaptation, SLA guarantees or generalization to other machines.

Original official VQA quality is only 2% under the frozen prompt; latency prediction does not repair that task-quality problem. A new prompt/model contract would need its own validated quality and latency measurements. Correctness approval for the original implementation is distinct from useful answer quality.

A later separately frozen revision could investigate crop/vision work features, output-length uncertainty and actual shared-memory/residency state, with fresh validation and network conditions where the feasible winner changes. These are proposed research directions, not implemented or validated fixes. Do not tune r1 using N2/N7 or N4/N9, relax the gate afterward, or enable G4 automatic selection. G4 measured-oracle regret/SLA analysis remains separate and this gate blocks a production handoff.

## Evidence and reproduction

Final CPU checks: **274 passed in 96.87 seconds**, **83.28% branch-aware coverage** against the unchanged 80% threshold. Ruff passed; strict mypy passed for 69 source files. Source/check receipts and full generated predictions are preserved in `evidence.tar.gz`; the original physical source archive remains separate.

The following are the actual commands used on this revision (run from the repository root with its prepared environments):

```bash
.venv-model/bin/python scripts/prediction/prepare_predictor_v2_data.py --output artifacts/phase3/physical-v2/data-r3
.venv/bin/python scripts/prediction/run_predictor_v2.py fit --data-bundle artifacts/phase3/physical-v2/data-r3 --output-dir artifacts/phase3/physical-v2/fit-r1
.venv/bin/python scripts/prediction/run_predictor_v2.py review --data-bundle artifacts/phase3/physical-v2/data-r3 --fit-dir artifacts/phase3/physical-v2/fit-r1 --output-dir artifacts/phase3/physical-v2/training-review-r1
.venv/bin/python scripts/prediction/run_predictor_v2.py validate --data-bundle artifacts/phase3/physical-v2/data-r3 --fit-dir artifacts/phase3/physical-v2/fit-r1 --output-dir artifacts/phase3/physical-v2/validation-r1
.venv/bin/python -m pytest --cov=adaptive_vlm --cov-report=term-missing --cov-report=json:artifacts/phase3/physical-v2/coverage-v2-final.json --junitxml=artifacts/phase3/physical-v2/tests-v2-final.xml -q -rs
.venv/bin/python -m ruff check . --no-cache
.venv/bin/python -m mypy --cache-dir /tmp/predictor-v2-mypy
```

The CLI requires a clean source tree and refuses an existing validation decision. These commands document the completed pass, not an instruction to run selection again. Source receipts pin the code used, independent of the later documentation commit. The feature-preparation command needs original input images and pinned tokenizer cache; the saved input-feature JSON is already committed. If the Linux validation store is lost across WSL restart, data preparation can recreate source-identical partitions, but it must not substitute inputs or authorize another selection.

To restore inspectable evidence and verify saved file bytes without fitting or reading final outcomes:

```bash
cd results/physical-g2-20261110/predictor_v2/r1
sha256sum -c SHA256SUMS
mkdir -p /tmp/physical-v2-r1-evidence
tar -xzf evidence.tar.gz -C /tmp/physical-v2-r1-evidence
```

Original `file_hashes.json` receipts cover files in the complete archive, including larger files omitted from the expanded summary directories. Verify those receipts after extraction. No image dataset, model weights, credentials, protected final outcomes or Linux sealed raw target store are added to this derived evidence archive. No AWS/Vultr provisioning, experiment, backup or artifact upload was performed.

Candidate V2-A calibration: `fit-r1/calibration_v2_a.json` in the result directory. Its canonical calibration ID is `2b0789184034b6a6b5292dbd95450ccf5eda847f0ac3230447a9d4efc0456761`; its exact file SHA-256 is `3191e67f19fe471347d5affbc6796ee81164213e36b398b3539d4a263fc55b0b`. These are different hash domains. Load through `CalibrationV2.load` and `PredictorV2` only as a research artifact; invalid/diagnostic estimates must never become execution choices. Gate decision canonical SHA is `48eb4d3f4adfa7f857ebeb6fa3f2d82f2c4a85a77d9ac88480836835dc7bca0d`. File-byte hashes for every report, fit and archive are in `SHA256SUMS`.

## Subsequent training-only diagnosis

The [stage diagnosis](../../results/physical-g2-20261110/predictor_v2/r1/stage_diagnosis/README.md) decomposes saved OOF/LOBO errors and identifies crop-dependent vision work, output-length uncertainty and unexplained block-dependent compute shifts. It does not change r1 coefficients, validation decisions or eligibility. The next feature candidate is crop-aware vision modeling; resource-state causes require targeted measurements.
