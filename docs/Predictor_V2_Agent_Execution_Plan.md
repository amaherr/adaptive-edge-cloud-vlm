# Predictor V2 — Agent Execution Specification

**Project:** Adaptive Edge–Cloud VLM Inference  
**Thesis:** Distributed Vision–Language Inference Across Edge and Cloud  
**Milestone:** G3 — Predictor V2 calibration and validation  
**Status:** Implementation plan; **not** a claim that G3 has passed  
**Target repository:** [amaherr/adaptive-edge-cloud-vlm](https://github.com/amaherr/adaptive-edge-cloud-vlm)  
**Primary deliverable:** A versioned, interpretable latency predictor, fitted **exclusively** on the frozen 60 training requests and five training network blocks, selected on the separate validation slice, and frozen before final evaluation.

> **Agent directive:** Implement this specification in the existing repository. Work in ordered phases, run the relevant tests at every gate, preserve existing functional interfaces, and produce a reproducible handoff with links to commits and artifacts. Do not mark a gate as passed without inspecting its actual evidence. Do not fabricate results, path names, successful runs, measurements, or commits. If raw Gate 2 artifacts are missing, implement and test with synthetic fixtures, then stop at the data-dependent gate and report exactly what remains.

---

## 0. Non-negotiable experimental rules

1. **Use the already predeclared 60/20/20 request split.** Do not resample, rebalance, shuffle assignments, or create a convenient new split.
2. **Fit only on** `training_request_ids × {N1,N3,N5,N6,N8}`. Training includes the three fixed configurations and the three planned repetitions per condition. Reject additional rows even if they exist in the Gate 2 archive.
3. **Validate only on** `validation_request_ids × {N2,N7}`. Validation outcomes may be used to compare the *predeclared* candidate models and select one, never to estimate final predictor coefficients.
4. **Protect the final test:** `evaluation_request_ids × {N4,N9}` is not available to fitting, feature scaling, hyperparameter choice, uncertainty calibration, model selection, threshold choice, debugging based on outcome, or iterative model improvement. After the model and decision rule are frozen, a single reporting/evaluation pass may be executed. **Never tune on N4 or N9.**
5. **All other crossed combinations are excluded** from these three datasets. In particular, do not borrow training-request results from N2/N7/N4/N9, or validation/evaluation-request results from training network blocks.
6. **No target leakage:** For a trial being predicted, never use its measured total latency, measured stage timings, realized generated-output length, actual response bytes, OOM outcome, post-execution GPU memory, or measured winner as *input features*. These variables may be target measurements or training-derived aggregates in the appropriate permitted slice.
7. **Keep hardware and model identity pinned:** The cloud virtual edge/cloud pair, immutable image digest, model contract, tokenizer, BF16 precision, batch size 1, `max_new_tokens=32`, network measurement methodology and transport protocol must match the profiled campaign. Do not reuse physical-host or local-CPU coefficients to fit the cloud-hosted pair.
8. **Treat rows as dependent:** Three repetitions and five network blocks do not turn 60 independent requests into 2,700 independent requests. Use request-group and network-block aware uncertainty and validation, and do not imply overly precise confidence intervals.
9. **Preserve infeasibility and failures:** Failed/OOM/timeout measured trials remain in coverage/feasibility accounting. Do not impute successful latency for them or silently remove them from the denominator. Exclude non-success observations only from ordinary supervised latency regression.
10. **Never activate an unvalidated placement or internal split.** The executable candidate set remains the three already supported policies unless a separately documented capability gate has passed.
11. **No hidden use of held-out data:** Read frozen evaluation ID metadata to enforce exclusions, but do not inspect final-test measured outcome fields until the frozen model-evaluation phase. If a shared profile file is required for provenance checks, isolate its final-test rows in a sealed/read-prohibited partition and avoid computing outcome statistics from them.
12. **No broad refit after validation:** When V2-A or V2-B is chosen, freeze its training-derived coefficients. Do not incorporate validation requests into a new full-data fit for the reported final test.

### Frozen request split

| Partition | Unique requests | Permitted use |
|---|---:|---|
| Training | **60** | Fit costs, transport coefficients, feature transforms, output-length estimate, residual correction, training-only uncertainty |
| Validation | **20** | Compare predetermined models, test N2/N7 generalization, choose frozen V2 variant |
| Final evaluation | **20** | One-time evaluation after freeze; never tuning |

### Frozen network block split

The following values are **nominal shaping settings**; use independent **measured** application-relevant goodput and RTT as runtime predictor inputs. Verify both directions and align probes with trial times.

| Block | Nominal egress rate / host | Added nominal RTT | Split |
|---|---:|---:|---|
| N1 | 2 Mbps | 20 ms | **Training** |
| N2 | 5 Mbps | 20 ms | **Validation** |
| N3 | 10 Mbps | 20 ms | **Training** |
| N4 | 25 Mbps | 20 ms | **Final evaluation — protected** |
| N5 | 50 Mbps | 20 ms | **Training** |
| N6 | 100 Mbps | 20 ms | **Training** |
| N7 | 10 Mbps | 5 ms | **Validation** |
| N8 | 10 Mbps | 50 ms | **Training** |
| N9 | 10 Mbps | 100 ms | **Final evaluation — protected** |

Expected planned measured attempts, assuming all three placements and three repetitions:

- **Fit:** `60 × 5 × 3 × 3 = 2,700` attempts.
- **Validation:** `20 × 2 × 3 × 3 = 360` attempts.
- **Final evaluation:** `20 × 2 × 3 × 3 = 360` attempts.
- **Entire Gate 2 campaign:** `100 × 9 × 3 × 3 = 8,100` attempts, **plus 27 warm-ups**.
- **All remaining cross-split attempts:** `8,100 − 2,700 − 360 − 360 = 4,680` are **excluded** from G3 fit, validation and final-test scoring. Do not use them as extra data.

Expected counts are schedule counts, **not** a statement that all attempts succeeded. Preserve unsuccessful attempts and report coverage.

### Existing split/protocol references

- `configs/evaluation/physical_vqa_split_v1.json` already lists the exact request IDs, network partition, task (`vqa_short_answer`), 32-token cap, 60,000 ms SLA, seed `1707`, 2,000 bootstrap resamples, and 95% confidence level. **Use it as the authoritative split assignment reference**; do not pretend the physical environment profile coefficients apply to the new cloud pair.
- Prefer the frozen cloud-specific equivalent (if already committed), e.g. `configs/evaluation/cloud_vqa_split_v1.json`, and verify that its assignments exactly match the frozen plan. If that file differs, stop and document the discrepancy before fitting.
- `docs/Gate2_Cloud_Migration_Roadmap_v2.md` documents the cloud-emulation environment, checkpoint discipline and G3 handoff. The underlying experiment is **distributed inference on two distinct cloud VMs under controlled network emulation**, **not** physical Jetson/mobile deployment or a real 5G radio link.

---

## 1. Technical objective and executable configurations

Predict, **before execution**, the steady-state fixed-policy end-to-end latency (milliseconds) of each of the existing placements for an image/question request and measured network state:

| Stable configuration ID | Edge computation | Cloud computation | Wire transfer |
|---|---|---|---|
| `edge/full` | Full Moondream2 VLM | None | None on inference data link |
| `cloud/raw_image` | Image read / framed request | Full Moondream2 VLM | Encoded source image + prompt, then response |
| `distributed/projected/synchronous` | Vision encoder + projector | Image/prompt prefill + language decoding | BF16 projected vision tensor + prompt, then response |

The current split boundary emits a BF16 `[729, 2048]` tensor: `729 × 2048 × 2 = 2,985,984` **raw payload bytes** before protocol framing. Do not mistake raw payload bytes for full request-wire bytes. All other model and token-generation properties remain pinned.

**Success condition:** Predict each feasible policy's latency and named non-overlapping components; return a typed invalid reason on unsupported configuration, unavailable hardware, model mismatch, insufficient memory, unhealthy server, unknown network state or prohibited extrapolation; rank feasible configurations correctly against a held-out measured oracle without reading the oracle during prediction.

**Out of scope for G3:** Training or finetuning Moondream2, changing token generation policy, developing new internal model split points, optimizing dynamic handoff overhead, changing the 60/20/20 split, or claiming physical-device generalizability. Selector regret/SLA analysis is primarily G4; G3 should expose the required ranking evidence.

---

## 2. Audit the repository before editing

Read the actual current code before modifying it; the following files were present in the inspected repository:

| File | Existing function | Action |
|---|---|---|
| `src/adaptive_vlm/scheduler/predictor.py` | `PredictionState`, `PredictionTask`, `NetworkFit`, `CandidateFit`, `Calibration`, `Prediction`, `Predictor.predict()` | Extend with versioned V2 or parallel V2 classes, preserving V1 behavior |
| `src/adaptive_vlm/benchmark/calibrate.py` | `load_profile_and_split()`, `_network_fit()`, `_candidate_fit()`, `fit_calibration()` | Strict three-way split loader + V2 fitting |
| `src/adaptive_vlm/benchmark/predictor_eval.py` | Predictions, holdout errors and ranking | Validation/test partition isolation and richer reports |
| `src/adaptive_vlm/scheduler/configuration.py` | Three supported `CONFIGURATIONS` | Do not alter registry without independent capability gate |
| `src/adaptive_vlm/scheduler/selector.py` | Feasibility-aware selector | Maintain compatible return contract; integration test only |
| `docs/phase3_predictor_schema.md` | V1 profiling/calibration definitions | Add V2 specification; retain original evidence |
| `docs/phase2_profile_schema.md` | Timing boundaries, null meanings, units | Reuse precise stage semantics |
| `docs/phase3_handoff.md` | Prior local CPU/loopback result | Keep as historical diagnostic; do not reuse as cloud calibration |

**Important discovered existing-code limitations to address:**

1. V1 splits data into `training_request_ids`, `holdout_request_ids` and a single `holdout_network_bucket_ids` set. V2 requires **three request partitions and three network partitions**, with a positive allowlist for the fit slice and explicit validation/final isolation.
2. V1 currently uses training medians for most compute costs and an ideal serialized-byte transfer formula with fitted RTT/intercept. This is the starting baseline, not the end state.
3. V1 rejects out-of-range network states. N7 nominal RTT = **5 ms** lies below the lowest nominal training RTT (20 ms), while N9 nominal RTT = **100 ms** lies above the highest (50 ms). V2 needs **predeclared and explicitly labeled** physics-based extrapolation, not accidental out-of-coverage or hidden tuning.
4. V1 predicts response size from a training median; keep that pre-execution-only design. It must not use a held-out realized output length or actual response bytes.
5. Prior local V1 ranking scored **0/2** fastest-choice matches versus **2/2** for constant median on its tiny CPU/loopback request holdout. The old calibration is diagnostic only and cannot satisfy cloud G3.

### Preflight commands (illustrative; adjust to current repository)

```bash
git status --short
git rev-parse HEAD
python -m pytest -q
python -m ruff check .
python -m mypy src
```

Record actual invocations, software environment, baseline test results, model hash, artifact hash, branch and commit SHA. Use a new branch, such as `feature/predictor-v2-cloud-g3`, and avoid editing any raw experimental logs.

**Gate G3.0 — baseline inspected**

- [ ] Actual source/API schema reviewed, including nested `profile.jsonl` fields and all currently registered configurations.
- [ ] Frozen split assignments and network ID membership match source manifest exactly.
- [ ] Baseline tests documented; Git status and provenance recorded.
- [ ] No production/raw artifact files overwritten.

---

## 3. Deliverables and planned file layout

Reuse the repo's current conventions where they exist. Suggested additions (not a license to invent unverified existing paths):

```text
configs/predictor/
    cloud_vqa_v2.json                  # Frozen V2 spec, source manifest hashes and gates

src/adaptive_vlm/scheduler/
    predictor.py                       # Versioned V2 prediction API / compatibility
    predictor_v2.py                    # Optional isolated implementation if cleaner

src/adaptive_vlm/benchmark/
    calibrate.py                       # Keep V1; add a V2 fit route or companion
    calibrate_v2.py                    # Optional strict data loader + fit pipeline
    predictor_eval.py                  # Keep V1; V2 validation/test adapter
    predictor_eval_v2.py               # Optional separated evaluator

scripts/
    validate_predictor_v2_dataset.py   # Offline split + provenance assertions
    run_predictor_v2.py                # Thin CLI only if no existing CLI extension fits

tests/
    ...                                # Existing organization, add V2 unit/integration tests

docs/
    phase3_cloud_predictor_v2.md       # Equations, features, units, protocol and results
    phase3_cloud_g3_handoff.md         # Evidence-based, post-implementation handoff

artifacts/phase3/cloud-v2/              # Generated, gitignored, S3-archived
    source_manifest.json
    fit_dataset_manifest.json
    calibration.json
    evaluation_validation.json
    predictions_validation.jsonl
    training_diagnostics.json
    coverage_report.json
    run_environment.json
    file_hashes.json
    ...
```

If existing commands can be safely extended (e.g. `adaptive-vlm calibrate-predictor --predictor-version 2`), prefer a consistent CLI over duplicating unnecessary wrappers. Ensure V1 calibration files remain loadable, and V2 has a distinct schema version / calibration ID.

**No final-test prediction/report artifact should exist before the freeze marker** and explicit G3 validation approval.

---

## 4. Phase A — strict data isolation and feature preparation

### A1. Create a positive-allowlist loader

Load only:

```python
TRAIN_BLOCKS = frozenset({"N1", "N3", "N5", "N6", "N8"})
VAL_BLOCKS = frozenset({"N2", "N7"})
TEST_BLOCKS = frozenset({"N4", "N9"})

# These must be loaded from the frozen, hash-verified split manifest,
# not constructed by new random sampling.
TRAIN_IDS = frozenset(split["training_request_ids"])
VAL_IDS = frozenset(split["validation_request_ids"])
TEST_IDS = frozenset(split["evaluation_request_ids"])

assert (len(TRAIN_IDS), len(VAL_IDS), len(TEST_IDS)) == (60, 20, 20)
assert not (TRAIN_IDS & VAL_IDS or TRAIN_IDS & TEST_IDS or VAL_IDS & TEST_IDS)
assert not (TRAIN_BLOCKS & VAL_BLOCKS or TRAIN_BLOCKS & TEST_BLOCKS or VAL_BLOCKS & TEST_BLOCKS)

# Each row must belong to precisely the intended crossed partition.
def partition_for(request_id: str, network_block_id: str) -> str:
    if request_id in TRAIN_IDS and network_block_id in TRAIN_BLOCKS:
        return "train"
    if request_id in VAL_IDS and network_block_id in VAL_BLOCKS:
        return "validation"
    if request_id in TEST_IDS and network_block_id in TEST_BLOCKS:
        return "final_test_protected"
    return "excluded"
```

- Prefer physically separate, content-hashed training and validation input artifacts; final-test measurements are not loaded by the normal fitting/validation path.
- Check the actual CSV/JSONL row schema before writing the parser: normalize `request_id`, `configuration_id`, `network_state.bucket_id`, `repetition`, `trial_id`, `warmup`, `status`, `observed_end_to_end_ms`, `fields`, `raw_metrics`, `host_pair_id`, `model_contract_sha256`, `dataset_manifest_sha256`, and placement hashes.
- Enforce exact unique trial key `(request_id, network_block_id, configuration_id, repetition)` (or the frozen schedule's canonical equivalent).
- Reject duplicates, wrong request IDs, wrong block IDs, unrecognized placements, incompatible host/model/dataset/config hashes, mismatched schedule, invalid units, mixed precision or generation budgets.
- Preserve failure records with typed `failure_reason`. Exclude `warmup=true` from measured denominators.
- Validate attempt counts against 2,700/360/360 **only if that campaign was completed**; otherwise generate a coverage failure and stop, not synthetic success rows.
- Read network probe evidence (`iperf3`, TCP echo, block validation/measurement manifest) and align measurements to the actual trial time. Do not substitute nominal shaping rate or added RTT for measured application goodput/RTT without an explicit labeled diagnostic mode.
- Exclude invalid/negative stage accounting residuals from coefficient fitting only under a **predeclared quality rule** and report exclusions. Never silently clip residuals or stage durations to zero.

**Absolute requirement:** All feature scaling, encoding, missing-data strategies, output-length priors, baseline medians and fitted coefficients are estimated on the train partition only. If any model uses fitted normalization, freeze its training-fitted scaler as part of `calibration.json`.

### A2. Construct only pre-execution features

| Feature | Units | Inference-time source | Role |
|---|---|---|---|
| `uplink_bps` | bits/s | Time-aligned probe / bounded live monitor | Request transfer model |
| `downlink_bps` | bits/s | Time-aligned probe / bounded live monitor | Response transfer model |
| `rtt_ms` | ms | Time-aligned TCP echo probe | Protocol/connection term |
| `image_bytes` | bytes | Source image before inference | Cloud request size |
| `image_width`, `image_height` | pixels | Input metadata | Optional preprocessing model and coverage |
| `prompt_tokens` | tokens | Pinned tokenizer/template; computed before selection | Cloud/edge prompt prefill cost |
| `task_bucket` | category | Frozen manifest | Coverage; only `vqa_short_answer` supported by this campaign |
| `requested_max_new_tokens` | tokens | Fixed model contract (32) | Contract and budget |
| `expected_output_tokens` | tokens | Training-derived prior or independently pre-execution estimator | Decode cost |
| `request_wire_bytes` | bytes | Predict from input size + training-fitted protocol overhead | Communication |
| `response_wire_bytes` | bytes | Training-only predicted median/size function | Communication |
| `edge/cloud memory budget` | MiB | Capability inventory + current pre-decision available resources | Feasibility check |
| `host_pair_id`, contract/profile hashes | identifiers | Frozen deployment/provenance | Compatibility check |

For this campaign, **task type and 32-token budget are constants**, not learnable diverse categorical or generation-range signals. Avoid adding more parameters than observable variation supports. Image dimensions may be treated as coverage descriptors unless training variation and cross-validation support a stable coefficient.

### A3. Explicitly forbid leakage

Before prediction of a validation or test trial, assert that the `PredictionTask`/`PredictionState` snapshot does **not** contain:

- `actual_output_tokens`, `decode_steps`, `output_token_hash` or actual response bytes for the current trial.
- Current trial `edge_total_ms`, `cloud_decode_ms`, `network_exchange_ms`, stage timings, request E2E latency or result status.
- A resource-feasibility flag derived from the current trial's success/OOM outcome.
- A condition-specific correction computed from any validation or protected final outcome.

Use training median output length as the default estimate. If a different pre-execution output estimator is considered, it must be fitted on training requests and validated against the training-only model before touching validation; its cost must be included in a future runtime-overhead analysis.

**Gate G3.A — data contract**

- [ ] Exact intended 2,700/360 schedule slots classified (or incomplete campaign reported) with missing/error counts.
- [ ] No fitting row comes from N2, N4, N7, N9 or from the validation/final request IDs.
- [ ] All other 4,680 cross-partition attempts are inaccessible to fitting and validation metrics.
- [ ] Provenance, hashes and units verified; all exclusions recorded.
- [ ] Pre-inference feature schema passes leakage tests.

---

## 5. Phase B — define an interpretable component model

For candidate configuration `c`, input `x`, network state `n`, and current capability state `r`:

```text
prediction(c, x, n) = compute(c, x)
                    + transfer(c, x, n)
                    + protocol_and_runtime_overhead(c, x, n)
```

All durations are **milliseconds**. Each component must be named, non-overlapping, testable, and sum to total within numerical tolerance. A fixed-policy Phase 2 target is steady-state `observed_end_to_end_ms` around `policy.infer`; actual future adaptive selection/monitor overhead belongs in the G4/G5 runtime measurement and must not be misrepresented as already included.

### B1. Communication equation

With measured uplink/downlink goodput in **bits/second** and complete wire-byte estimates:

\[
T_{\mathrm{comm},c} = \beta_{0,c} + \beta_{r,c}\,RTT_{ms}
    + \frac{8000\,S_{up,c}}{B_{up,bps}}
    + \frac{8000\,S_{down,c}}{B_{down,bps}}
\]

- `beta_0` = training-fitted protocol/connection/framing residual (ms).
- `beta_r` = nonnegative RTT sensitivity, fitted only on training RTT conditions; **do not force it to 1** without measurement.
- `S_up` = predicted full request frame bytes for each placement (source image + prompt/framing, or BF16 projected tensor + prompt/framing).
- `S_down` = training-fitted response-wire bytes estimate, **not** realized held-out response bytes.
- `B_up`, `B_down` = measured goodput, not advertised interface speed or nominal `tc` rate.
- `8000` converts bytes / bits/s into milliseconds.
- For `edge/full`, **communication cost = 0** for the remote inference link.

The current code's `transfer_ms(bytes, goodput_bps)` is the right units helper. Use a small, preferably nonnegative/monotonic fit around the analytical serialization terms if training residuals show systematic link inefficiency. Do **not** fit a rich network black-box from only five network blocks.

**Networking caveat:** The measured `network_exchange_ms` already includes `server_processing_ms`. Use `transport_residual_ms = network_exchange_ms - server_processing_ms` (after validating timestamps and nonnegative accounting), or sum only explicitly disjoint transport stages. Never double count server processing or queue time.

### B2. Computation equation

Use measured **training** stage components and a low-dimensional fit:

**Edge-only**

```text
T_edge = estimated_edge_nondecode_ms(x)
       + estimated_edge_decode_ms_per_token * expected_output_tokens(x)
       + measured_training_client_residual_ms
```

**Cloud-only**

```text
T_cloud = estimated_client_image_read_ms(x)
        + estimated_cloud_nondecode_ms(x)
        + estimated_cloud_decode_ms_per_token * expected_output_tokens(x)
        + predicted_transport_ms(x, n)
        + estimated_training_client_residual_ms
```

**Distributed synchronous**

```text
T_split = estimated_edge_vision_and_projector_ms(x)
        + estimated_tensor_serialization_ms(x)
        + estimated_cloud_text_prefill_and_other_nondecode_ms(x)
        + estimated_cloud_decode_ms_per_token * expected_output_tokens(x)
        + predicted_transport_ms(x, n)
        + estimated_training_client_residual_ms
```

Take care to account for tensor deserialization and other server work **once** inside the server non-decode term, consistent with the exact Phase 2 timing schema. For all-cloud, model's combined image encode/prefill timing may not be separately measurable; use an appropriately named combined component instead of inventing a decomposition. For edge, respect that `edge_total_ms` includes full model work; subtract decode **only once** when forming a non-decode training target.

**Initial fit strategy:** Robust training medians for near-constant stages plus at most a few low-dimensional request-specific slopes (e.g. prompt token count, encoded image bytes or pixels if actually predictive). Fit decode cost from training `decode_ms / generated_tokens` on successful training cases with positive output length. Use 32 as the declared generation maximum; never substitute 32 for the actual predicted expected length unless 32 was predeclared as the estimate.

### B3. Complexity budget and candidate variants

Implement **three named comparison models**:

| Label | Exact role | Fitting rule |
|---|---|---|
| `v1_recalibrated` | Existing component-median baseline | Refit on the new permitted cloud training slice; no physical/loopback coefficients |
| `v2_a` | **Primary** analytical transmission + request-aware, low-dimensional measured computation model | Training-only stage medians/slopes and nonnegative RTT coefficient |
| `v2_b` | Optional `v2_a` with a regularized **linear** residual correction | Train-only Ridge fit on residuals, tiny feature set; select only if validation improves |

Keep a **constant-median per-configuration** baseline as well. Do not introduce random forests, gradient boosting or a neural network in this G3 milestone.

If implementing Ridge, standardization, intercept regularization, and feature selection must use training-only information. Predeclare a small alpha grid or a single fixed alpha **before validation**, then select only within the declared comparison rules. Avoid repeated manual hyperparameter iteration against N2/N7.

**Gate G3.B — model behavior**

- [ ] Component sum equals reported prediction with no double counting.
- [ ] Network transfer increases with bytes, decreases with bandwidth, and has nonnegative RTT sensitivity.
- [ ] All three measured placements have separate trained computation profiles and correct wire format.
- [ ] No negative or NaN component/total latencies; nonfinite inputs produce typed invalid predictions.
- [ ] All fitted terms, feature weights, intercepts, scales, training source hashes, units and versions are recorded.

---

## 6. Phase C — resource feasibility and coverage policy

Keep **prediction validity**, **execution feasibility** and **quality feasibility** distinct:

1. **Execution:** candidate exists in the registered capability set, server is healthy/reachable, model/placement/serialization contract hashes match, required edge/cloud components are actually loaded, and there is no unsupported split.
2. **Memory:** compute conservative candidate-specific edge and cloud requirements from **training measurements** or a separately approved hardware capability preflight. Include a **predeclared headroom margin**, use consistent allocated/reserved/total/incremental semantics, and ensure the memory budget uses pre-decision observations. If the requirement cannot be established, mark it unknown and exclude rather than guessing.
3. **Quality:** current frozen three-policy equivalence/correctness gate remains required; quality must not be inferred from a single future prediction.
4. **Coverage:** request/token/network features must be supported and the host/model/task identities match the training regime.
5. **No feasible candidate:** return a typed failure. Do not execute a known-infeasible placement or select by hidden observed latency.

### C1. Three network-coverage labels

The existing V1 strict min/max rejection should evolve into:

| Coverage status | Rule | Runtime behavior |
|---|---|---|
| `interpolation` | Features inside measured training support | Predict normally |
| `physics_extrapolation` | Outside sampled RTT/goodput bounds but **inside an independently predeclared, physically justified operational envelope** | Compute prediction, attach coverage warning and larger/explicit uncertainty; selection eligibility according to frozen safety policy |
| `unsupported` | Invalid/missing/stale measurements, changed host/model, or outside approved operational envelope | `valid=false`, named reason; exclude from selection |

**Crucial:** N7 at nominal 5 ms RTT is **lower than training support**, so it tests extrapolation rather than straightforward RTT interpolation. N9 at nominal 100 ms RTT tests **high-RTT extrapolation** and must remain untouched until after freeze. Do not make the policy succeed on N7/N9 by estimating parameters from those outcomes.

Before running validation, independently predeclare (and justify from protocol limits and engineering requirements) the permitted RTT and goodput extrapolation envelope, numerical uncertainty/safety rules and whether extrapolated candidates are eligible for ranking. It is acceptable to produce an **offline research prediction** with `physics_extrapolation` but exclude it from live automatic selection if safe extrapolation cannot be justified. Always separate these two counts in reporting.

For measured values beyond training min/max, report exact distance beyond support. Do not silently clamp bandwidth or RTT to the nearest observed endpoint; that would conceal extrapolation.

**Gate G3.C — safety**

- [ ] Known OOM / unavailable execution paths excluded with explicit reasons.
- [ ] Edge **and** cloud memory feasibility considered where relevant.
- [ ] RTT/goodput interpolation, extrapolation and unsupported cases differentiated.
- [ ] N7 extrapolation is a frozen experimental test of model transfer, not a source of calibration.
- [ ] Predictor and selector handle an empty feasible set deterministically.

---

## 7. Phase D — training and internal diagnostics

### D1. Train only on positive-allowlist fit rows

1. Read/hash the frozen fit manifest and the permitted 2,700 planned trial keys. Check for missing, duplicate and unexpected records.
2. Validate timestamps and independent network probe measurements; reject inconsistent or ungrounded network metrics.
3. Summarize failure/OOM/timeout rates per configuration and training network block, retaining all failures in denominators.
4. Estimate training-only task feature ranges and pre-execution output-token prior.
5. Fit each candidate's robust computation components and communication intercept/RTT coefficient.
6. Fit optional V2-B residual correction using **only training** samples, with predeclared parameter choices.
7. Use **group-aware out-of-fold training residuals** (request grouping, with optional network leave-one-block-out stress checks) for a transparent uncertainty estimate; do not compute overly narrow intervals from in-sample residuals as if independent.
8. Write immutable training diagnostics and a versioned calibration record with SHA-256 fingerprint of source training rows, model/hardware contract and split specification.

### D2. Internal, training-only robustness checks

- **Grouped request CV:** keep all repetitions and network observations for a request in the same fold. Test that fitting one fold's held-out request cannot access that request's target/realized tokens.
- **Leave-one-training-network-block-out:** use each of N1, N3, N5, N6, N8 once as a **diagnostic** excluded block; record how well the physical model extrapolates/interpolates to that omitted training block. Do not call these estimates final-test results.
- **Per-placement residual analysis:** signed bias vs nominal block, measured goodput, RTT, prompt tokens, image size, expected tokens and measured output length (the latter **only for post-hoc training diagnostics**).
- **Stability checks:** coefficient signs, repeat variance, hardware/network trace alignment, stage accounting anomalies, sensitivity to small training subsets.

**Training gate:** If the primary model has unstable/unphysical coefficients, keep a simpler component model. Never “rescue” an unstable training fit using final-test data.

---

## 8. Phase E — isolated validation on 20 requests × N2/N7

Validation consists of the **360 planned measured attempts** for the 20 validation requests, two validation network blocks, three placements and three repetitions. Generate predictions **before accessing corresponding measured results** in the validation evaluator; preserve the task/system input snapshot used for each prediction.

### E1. Required metrics

| Metric | Definition | Decision rule |
|---|---|---|
| Median APE | `median(abs(pred - observed) / observed)` over successful, valid predictions | **≤ 15%** target per configuration |
| P90 APE | 90th percentile absolute percentage error | **≤ 30%** target per configuration |
| MAE (ms) | Mean absolute error | Report per placement/block |
| Signed bias (ms and %) | Mean(predicted − measured) | Report; inspect systematic underestimation |
| Top-1 agreement | Predicted fastest **feasible** configuration equals observed fastest feasible configuration | At least constant-median baseline on comparable paired conditions |
| Prediction coverage | Valid predictions / eligible scheduled prediction attempts | Report with reasons; cannot cherry-pick only valid cases |
| Failure rate | OOM/timeout/error attempts / all scheduled measured attempts | Report separately |
| Unsafe feasibility choices | Known infeasible candidate incorrectly declared eligible | **0** known unsafe choices |
| Interval coverage | Observed total inside training-calibrated interval | Report; do not claim probabilistic guarantees without calibration evidence |
| Prediction latency | Wall-clock model evaluation overhead | Report separately from fixed-policy E2E |

Retain existing predeclared **15% median APE** and **30% P90 APE** targets from the Phase 3 plan. A gate is not passed if all hard cases become invalid and the report only scores easy successes. Coverage and exclusions must be visible.

### E2. Mandatory disaggregations

- `N2` separate from `N7`; **N7 is lower-RTT extrapolation** relative to nominal training conditions.
- Each of `edge/full`, `cloud/raw_image`, `distributed/projected/synchronous` separately.
- Failure/invalid/unsafe rates separately from numerical accuracy.
- Request-level summary using the three repetitions, plus repetition-level diagnostic results.
- `interpolation` versus `physics_extrapolation` versus `unsupported` counts and errors.

### E3. Ranking methodology

For a matched `(request_id, network_block_id, repetition)` comparison, form each candidate prediction strictly from its **pre-execution** snapshot. The retrospective observed fastest *feasible* candidate uses measured latencies **only within the evaluator after predictions are sealed**. Use stable configuration ID for deterministic exact ties and document incomplete triples separately.

**Primary reporting:** Aggregate repeat-level latencies per `(request_id, network_block_id, candidate)` (e.g. median of successful repetitions with transparent failure policy) and rank on independent **request × block** conditions. Keep repeat-level agreement as secondary to avoid pseudo-replication. For uncertainty, resample at least by request group and report the limited number of independent network blocks; avoid naive trial-wise bootstrap confidence claims.

### E4. Predeclared selection between V1, V2-A and V2-B

1. Reject models with unsafe feasibility choices or unjustified, excessive invalid coverage.
2. Require numerical-error gate for the intended supported validation domain.
3. Compare top-1 fastest-feasible agreement against the recalibrated V1 and constant-median baselines.
4. Prefer stronger ranking (the downstream decision objective), use latency error as the secondary criterion, and prefer the **simpler** model if the apparent improvement is too small or unstable.
5. Record the chosen model and rejected models in an immutable validation decision artifact. **No repeated changes driven by N2/N7 after this one declared selection.** For an unsuccessful gate, report G3 blocked and specify a **new, separately frozen revision** for any future attempt.

The existing local V1 diagnostic had low numerical APE but missed fastest-choice ranking. Do not repeat that mistake: **accurate total milliseconds are insufficient if their errors change the placement decision**.

**Gate G3.E — validation**

- [ ] Predictions generated using only pre-inference features and frozen fit parameters.
- [ ] N2 and N7 evaluated separately, including explicit extrapolation assessment.
- [ ] All metrics and failure/coverage denominators reported.
- [ ] Baseline comparisons (recalibrated V1, constant median, V2-A, optional V2-B) complete.
- [ ] One frozen model choice documented; no N4/N9 outcomes read.
- [ ] If numerical/ranking/safety gates fail, mark G3 **failed/blocked**, not passed.

---

## 9. Phase F — compatibility, test matrix and CI

Create dependency-light synthetic fixtures to run the tests without AWS GPUs; separately run real Gate 2 data integration tests when raw artifacts are available.

### Required automated tests

**Split integrity and provenance**

- Exact 60/20/20 membership and 5/2/2 network partition.
- 2,700/360/360 planned slots with uniqueness and no unexpected crossed rows.
- Fail hard if a fit row contains `N2`, `N4`, `N7` or `N9` or a non-training request ID.
- Fail on duplicate trial key, mixed host pair, model, tokenizer, model artifact, dataset hash, precision, image manifest or generation budget.
- Inject an extreme final-test observed latency; fitted calibration hash/coefficients must remain **identical**. Even better, test training paths have **no permission to read the sealed final artifact**.
- Warm-ups excluded from measured metrics; failures retained in counts but not regression fit.

**Feature and leakage behavior**

- Hide/change held-out `actual_output_tokens`, `decode_ms`, measured total, actual response bytes: same pre-execution prediction.
- Prompt count uses pinned tokenizer/template, not character count.
- Train-fitted scaler and output prior unaffected by changes in validation/final rows.

**Physics/component arithmetic**

- `transfer_ms(1_000_000 bytes, 8_000_000 bps) == 1000 ms`.
- Larger bytes imply no smaller pure serialization cost; larger bandwidth implies no larger pure serialization cost.
- RTT slope nonnegative; no accidental extra full RTT without demonstrated protocol semantics.
- Edge policy has no inference-network transfer term.
- Distributed BF16 payload is **2,985,984 bytes before framing** under pinned model.
- `network_exchange_ms` and `server_processing_ms` are never added as overlapping totals.
- Named components sum to reported total, with finite nonnegative predictions.

**Coverage, hardware, and failures**

- `N7`-like lower-RTT test results in declared `physics_extrapolation` or explicit unsupported reason, never mislabeled interpolation.
- High-RTT synthetic test exercises extrapolation mechanics **without using N9 outcome data**.
- Missing/stale network probe, zero/NaN goodput, changed model/host ID and out-of-envelope inputs fail with typed reasons.
- Edge/cloud GPU budget and capability mismatch exclude unsafe placements; no feasible config yields deterministic failure.
- Unknown memory requirement is not silently treated as zero.

**Determinism and integration**

- Same calibration/source/state/task yields the exact same prediction and component breakdown.
- V1 calibration loading remains supported; V2 has its own schema/version/hash.
- Predictor's public API remains compatible with scheduler/selector integration.
- All three policies are correctly keyed by stable configuration IDs.
- Gate report cannot pass when ranking sample set is empty or consists entirely of invalid predictions.

Run relevant test suite, lint, type checking, and static checks; preserve exact command outputs in the handoff. Any GPU-dependent tests must be clearly labeled as skipped/unavailable rather than silently reported as passed.

**Gate G3.F — code quality**

- [ ] Unit/integration test matrix complete and passing.
- [ ] Static/lint/type checks passing under repository's actual configuration.
- [ ] Deterministic serialization, model loading and provenance checks working.
- [ ] Existing V1/selector/fixed-inference behavior unbroken.

---

## 10. Phase G — freeze the final predictor, then protect final evaluation

### G1. Freeze marker **before** final-test access

Write a content-addressed marker recording at least:

```text
frozen_request_split_sha256
frozen_network_profile_sha256
training_rows_sha256
source_gate2_campaign_id
host_pair_id
terraform_environment_id_and_revision
application_image_digest
model_contract_sha256
model_artifact_sha256
dataset_manifest_sha256
code_git_commit_sha
calibration_sha256
selected_predictor_version
feature_schema_version
network_extrapolation_and_safety_policy
uncertainty_method
validation_decision_sha256
sla_ms = 60000
max_new_tokens = 32
random_seed = 1707
```

Retain the initial cloud environment hardware definitions (`g6f.2xlarge` virtual edge and `g6.2xlarge` cloud if these are the **actual verified nodes**, rather than assumptions) and application/image/model hashes. Do not mix any coefficient fits from the old Windows physical RTX pair or WSL localhost. Archive all artifacts to private versioned S3/checkpoint storage, with local SHA-256 receipts. Do not commit licensed VQA images, model weights, secrets or oversized raw trial logs to GitHub.

### G2. Single final evaluation after freeze (only if explicitly proceeding)

The **only** allowed test slice is `evaluation_request_ids × {N4,N9}`: 360 scheduled attempts. Generate predictions using the **already frozen model**, independently of the withheld measured outcomes. Once all predictions are sealed and hashed, the evaluator may read final measured latencies to calculate the exact same metrics as validation, clearly separating N4 and N9. Report any N9 extrapolation failure rather than changing the model.

Do **not** re-run fitting, select a new model, change extrapolation envelope, modify uncertainty width or adjust thresholds based on N4 or N9 results. If final performance is weak, report it faithfully; a new experiment requires a declared new campaign and a genuinely untouched test design.

> If this task request is interpreted as **training and validation only**, **stop at the freeze marker**. Do not automatically open or score final-test data; leave it for the later explicitly authorized final-evaluation milestone.

---

## 11. Outputs and reporting templates

Produce a versioned immutable artifact bundle and documentation, with **real measurements only**:

```text
artifacts/phase3/cloud-v2/<run-id>/
  source_manifest.json
  partition_audit.json
  training_data_manifest.json
  calibration.json
  training_diagnostics.json
  evaluation_validation.json
  predictions_validation.jsonl
  coverage_report.json
  feasibility_report.json
  candidate_model_comparison.json
  frozen_model_decision.json
  run_environment.json
  model_freeze_marker.json
  file_hashes.json
  checkpoint_receipts/                # Where supported by cloud campaign
  # Only after an authorized, post-freeze final evaluation:
  evaluation_final.json
  predictions_final.jsonl
```

The Markdown handoff `docs/phase3_cloud_g3_handoff.md` must contain:

1. **Environment identity:** actual edge/cloud instance types, driver versions, GPU, CUDA, Docker image digest, model/manifest fingerprints, Git commit, campaign ID.
2. **Exact data partition:** independent request IDs and block IDs, attempted/success/failed counts for train and validation, explicit confirmation 4,680 crossed attempts were excluded.
3. **Predictor equations:** named non-overlapping components, actual measured transfer units and fitted coefficients; output length prior and uncertainty method.
4. **Training process:** grouping methodology, internal CV, number of independent requests/blocks, type of regression, regularization, fitted network RTT behavior.
5. **Validation results:** per-configuration and per-N2/N7 median/P90 APE, MAE, signed bias, predicted ranking accuracy, constant-median/recalibrated V1 comparison, coverage, failure rate, excluded candidates, interval coverage.
6. **Extrapolation:** why N7 is lower-RTT extrapolation and how N9 will later test upper-RTT extrapolation, with any operational restrictions.
7. **Failures and anomalies:** OOMs, timeouts, negative residuals, inconsistent network probes, invalid predictions, environment drift.
8. **Gate decision:** Pass/Fail/Blocked against predeclared thresholds; exact artifacts and hashes supporting each claim.
9. **Reproduction:** exact tested commands and outputs, no invented invocations, restore/hash verification, explicit instructions to avoid N4/N9 until freeze.
10. **G4 handoff:** calibration path/hash, prediction API, feasible-policy registry, test fixtures and caveats; later G4 measured-oracle regret/SLA work stays separate.

### Recommended summary table for the actual validation report

| Predictor | Placement | Block | N successes / attempts | Valid prediction coverage | Median APE | P90 APE | MAE (ms) | Signed bias | Top-1 ranking contribution | Interpolation / extrapolation |
|---|---|---|---|---|---|---|---|---|---|---|
| `v1_recalibrated` | `edge/full` | N2/N7 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `v2_a` | `cloud/raw_image` | N2/N7 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `v2_b` (if implemented) | `distributed/projected/synchronous` | N2/N7 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The table is an output template, **not** fabricated data. Produce all placement/block/model combinations rather than only the illustrative rows above.

---

## 12. Ordered task checklist (execute in this order)

### Phase 0 — audit and freeze source inputs
- [ ] Inspect repository branch, current code, profile schema and past V1 handoff.
- [ ] Verify frozen cloud campaign + 60/20/20 assignments and nine network blocks.
- [ ] Record Git/model/hardware/manifest/environment provenance; baseline tests.

### Phase 1 — isolate data
- [ ] Implement strict three-way data loader and row allowlists.
- [ ] Validate complete schedules, uniqueness, measured network evidence and data quality.
- [ ] Seal protected final-test data and exclude all 4,680 crossed rows.
- [ ] Add data-leakage and provenance unit tests.

### Phase 2 — implement V2
- [ ] Implement pre-inference task/network/resource feature contract.
- [ ] Implement `v1_recalibrated`, `v2_a`, optional `v2_b`.
- [ ] Fit non-overlapping stage and transport models with correct unit math.
- [ ] Add conservative resource feasibility and explicit coverage/extrapolation status.
- [ ] Preserve backward-compatible predictor/selector interfaces.

### Phase 3 — fit training-only
- [ ] Fit models exclusively on 60 training requests and N1/N3/N5/N6/N8.
- [ ] Run request-grouped CV and training-only leave-one-network-block-out diagnostics.
- [ ] Estimate uncertainty from training-only out-of-fold residuals.
- [ ] Record all fitted coefficients, provenance, failures and coverage.

### Phase 4 — validate
- [ ] Predict 20 validation requests under N2/N7 with frozen candidate fits.
- [ ] Score latency errors, coverage, feasibility, ranking and uncertainty by block and configuration.
- [ ] Compare to constant-median and recalibrated V1 baselines.
- [ ] Make a **single** predeclared model-selection decision; mark failures truthfully.

### Phase 5 — quality and freeze
- [ ] Run test, lint, type-check and integration suite.
- [ ] Generate validation artifacts, hashes and G3 handoff.
- [ ] Freeze calibration, extrapolation envelope, safety policy and code/model versions.
- [ ] **Do not inspect N4/N9 evaluation outcomes** as part of the fitting/validation task.

### Phase 6 — later final evaluation (separate authorization)
- [ ] Check frozen marker and immutable calibration digest.
- [ ] Generate/seal predictions for final 20 requests × N4/N9.
- [ ] Read final measured outcomes once; score without changing the model.
- [ ] Hand off to G4 selection/oracle analysis.

---

## 13. Hard stop conditions

Stop fitting and report a **blocker** instead of modifying the protocol when any of these occurs:

- Frozen request or network membership cannot be verified, or the available cloud split differs from the committed assignments.
- Actual Gate 2 source data is incomplete, corrupted, inconsistently checkpointed, mixed across host environments, or lacks trustworthy network probes for claimed controlled states.
- GPU/model/container/manifest identity differs from the frozen cloud campaign in a way that invalidates its measured costs.
- A training operation would access any validation or final outcome data, or any extra crossed row.
- Memory feasibility evidence is unavailable for a candidate the selector would otherwise attempt.
- Required features are unknowable before inference (e.g. actual response length) and no approved training-only prior exists.
- N7 is unsupported under the frozen coverage policy and the evaluator would otherwise hide it or count it as successful.
- Metric sample sizes, missing trials or failures invalidate a gate; report this instead of modifying gates after looking at outcomes.
- Evaluation reveals a weak model: **do not** use N4/N9 to iterate.

The agent may still complete non-data-dependent implementation, deterministic unit tests and documentation, and should state which data-dependent gates remain blocked.

---

## 14. Agent completion response contract

At the end of execution, reply with:

1. **Implemented:** exact file paths changed, major methods/classes, commit IDs if actually created.
2. **Evidence:** actual tests/lint/type checks and verified run/environment hashes.
3. **Data audit:** training/validation request-block membership, all attempted/success/failure counts and exclusions.
4. **Validated results:** table of numerical error and ranking comparisons for N2 and N7, **or** explicit explanation why unmeasured.
5. **Gate:** `G3 PASSED`, `G3 FAILED`, or `G3 BLOCKED`, each backed by artifact paths and predeclared criteria.
6. **Frozen predictor:** calibration path and hash, selected variant, supported coverage regime, warnings and remaining constraints.
7. **Next step:** G4 selector/oracle integration; protected N4/N9 scoring only in the separately authorized post-freeze evaluation phase.

**Do not say implementation or validation succeeded unless real artifacts and measurements establish it.**

---

## 15. Research grounding

- **Janus: Collaborative Vision Transformer Under Dynamic Network Environment** — Sections III-C/III-D present lightweight profiling and network-aware dynamic scheduling, an architectural precedent for transparent component-based latency modeling. The present thesis extends that idea to a full VLM and a distinct three-policy deployment; Janus does **not** provide this thesis's measured coefficients or split protocol.
- **Distributed VLMs: Efficient Vision-Language Processing through Cloud-Edge Collaboration** — architecture places vision/projector on the edge and language generation on the server, with throughput-focused evaluation. Here, Predictor V2 must separately estimate per-request **end-to-end latency**, feasibility and network effects, not equate batch throughput with request latency.
- Existing repository documentation: [`docs/phase2_profile_schema.md`](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/docs/phase2_profile_schema.md), [`docs/phase3_predictor_schema.md`](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/docs/phase3_predictor_schema.md), [`docs/dynamic_scheduler_implementation_plan.md`](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/main/docs/dynamic_scheduler_implementation_plan.md), and the frozen evaluation split.

**End of executable G3 specification.**
