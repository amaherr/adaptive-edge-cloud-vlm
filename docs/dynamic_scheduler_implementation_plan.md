# Dynamic VLM configuration scheduler: implementation plan

Status: implementation plan, based on the repository as inspected on 2026-10-25 and the supplied `dynamic_vlm_configuration_framework.md`. The gates below are proposed acceptance criteria for this phase; record any threshold changes **before** running the final evaluation.

## 1. Baseline audit and scope

The repository already provides three fixed execution policies through `ExecutionPolicy.infer(InferenceRequest)`: `LocalExecutionPolicy` (`edge`), `CloudExecutionPolicy` (`cloud`), and `DistributedExecutionPolicy` (`distributed`). The distributed path runs Moondream vision and projector on the edge, serializes the projected tensor, and runs image prefill and language generation on the server. It has synchronous and `prompt_first` pipelines. The pinned model/config fingerprint, bounded TCP frames, tensor integrity checks, frozen request manifests, seeded benchmark schedules, per-request records, comparison, and memory/timing hooks are useful foundations.

The adaptive scheduler, live network/resource sensing, performance predictor, and internal model split points are absent. Current `benchmark --mode` constructs **one** policy for a whole run. The cloud and distributed server CLI commands expose different request paths; one server is not currently advertised as serving both for per-request switching. `compare-runs` assumes one distinct mode per run, so it cannot evaluate repeated configurations within one adaptive run without a new analysis path. The current cloud hardware inventory is pending; the published 12-request real-model results were one-sweep localhost CPU correctness observations, not controlled latency estimates. The 4 GiB edge GPU did not reliably load the full model.

| Requirement from source document | Current evidence | Planning consequence |
| --- | --- | --- |
| One configuration interface | `execution/base.py` already has `infer`; CLI and benchmark construct only one mode at a time | Add a configuration registry and dispatcher, retaining existing policies as executors |
| Per-stage profiling | `models/moondream.py` records vision, projection, image prefill, combined language generation, output tokens and memory; remote policies record wire bytes, client elapsed and server metrics | Reuse hooks, add exact missing timing boundaries and separate prompt prefill/decode |
| Tensor-size accounting | `ProjectedVisionFeatures` validates a fixed `(729, 2048)` tensor; `TensorPayload` records payload bytes; protocol framing records wire bytes | Profile payload **and** full request/response bytes; inspect each new split before enabling it |
| Task state | `InferenceRequest` contains image path and prompt only; model config fixes `max_new_tokens: 32` | Add request metadata with explicit output budget/estimate; keep generation contract stable for quality comparisons |
| Online state | No bandwidth/RTT or GPU load collector | Begin with validated manual/replay state, then add live probes with freshness and failure behavior |
| Oracle/evaluation | Fixed-policy benchmark and `compare-runs` exist | Add an independent measured-choice oracle and paired adaptive evaluation |

**Thesis milestone:** a request-level, system- and task-aware selector among the three *already executable* placements. Internal vision-layer or language-layer splits are a separately gated extension. A successful framework need not show that a new internal split is feasible or faster; any unsupported candidate must be explicitly marked unavailable rather than silently treated as implemented.

### V1 scope and terminology

- **Configuration** is `(mode, split_id, pipeline, placement_profile_id, model_contract_id)`. Initial IDs: `edge/full`, `cloud/raw_image`, `distributed/projected/synchronous`. Treat `distributed/projected/prompt_first` as a separate optional candidate because its overlap changes the latency model. `split_id` is `none` for edge/cloud. No compression or quantization parameter in V1.
- **System state V0** is uplink goodput in **bits/s**, RTT in **ms**, measured timestamp, and source (`manual`, `replay`, `probe`). Add free edge memory, edge/cloud load, and queue state only after calibration. Unknown or stale state must have a documented fallback.
- **Task state V0** is image byte size, decoded image dimensions, prompt token count using the pinned tokenizer/template, expected output token count or explicit generation budget, and optional latency SLA. Add task type and quality requirement later. Token count cannot be inferred reliably from character count. Cloud-only clients currently need no model weights; load only the verified tokenizer artifact for task analysis and measure its overhead, or declare prompt tokens unavailable and use a separately calibrated fallback.
- **Prediction** returns total latency and named terms, bytes, memory requirement, validity/reason, model/profile version, and uncertainty or coverage status. Use milliseconds, bytes, MiB and bits/s consistently.
- **Objective** is minimum predicted latency among eligible configurations. Eligibility includes available executor/server, model fingerprint, hardware memory/headroom, profile coverage, optional quality floor, and SLA. If no candidate meets only the SLA, report `sla_unmet` and select the lowest predicted latency among the other feasible candidates. If none is safe/executable, return a typed failure; never execute an unvalidated split. Stable ID ordering breaks ties.

The runtime must log the full ranked candidate set, rejection reasons, selected ID, state/task snapshot, prediction/profile IDs, observed result and fallback. The scheduler does not run inference or mutate model weights.

## 2. Proposed component and artifact contracts

Keep the existing `InferenceRequest` and fixed-policy commands backward compatible. Introduce small typed records in a new scheduler package (names may be adjusted during implementation):

| Record / API | Required contents | Suggested integration |
| --- | --- | --- |
| `ConfigurationSpec` and `CapabilityRegistry` | Stable ID, execution mode, split boundary, pipeline, required edge/cloud components, supported device and protocol/model fingerprint | `src/adaptive_vlm/scheduler/configuration.py`; registry built from actually loaded policies and server health/capability response |
| `SystemState` / `TaskState` | Typed values with units, timestamp, provenance, missing/unknown flags; task token estimates and image metadata | `scheduler/state.py`; derive task features once before selection |
| `ProfileRow` | Schema/version, host pair, placement, model/manifest hashes, candidate and task bucket, raw measurements, success/failure, timestamps | `scheduler/profile.py`; JSONL under ignored `artifacts/profiles/`; optionally export CSV for analysis |
| `Predictor.predict(spec, state, task)` | Component estimate, total estimate, memory/bytes, coverage status | `scheduler/predictor.py`; no model execution |
| `Scheduler.select(...)` | Ordered candidates, exclusions, decision, reason and profile version | `scheduler/selector.py`; deterministic pure logic |
| `AdaptiveExecutionPolicy` or dispatcher | `select` then call the matching existing executor once; record selection and execution overhead | `execution/adaptive.py`, CLI and benchmark extension |

Version the new profile/decision artifacts separately. Do not force scheduler fields into the existing flat `InferenceResult.metrics` except a small set of scalar identifiers and timings: the rich decision should be a companion JSON object/JSONL row. Preserve existing `RequestRecord`/benchmark artifacts or explicitly bump schema and provide a migration reader. Pin input manifest, model contract, hardware profiles, software lock and profile hash in every adaptive run.

### Initial profile schema

One row per request/configuration/trial, with nulls and a reason for genuinely inapplicable fields, never zero as a substitute for missing data:

```text
schema_version, trial_id, request_id, configuration_id, status, failure_reason
model_contract_sha256, dataset_manifest_sha256, host_pair_id, placement_profile_id
profile_campaign_id, measured_at_utc, image_width, image_height, image_bytes
prompt_tokens, requested_max_new_tokens, actual_output_tokens, task_type
network_uplink_bps, network_rtt_ms, network_downlink_bps, server_load_bucket
edge_vision_ms, edge_projector_ms, edge_other_ms, edge_total_ms
serialization_ms, request_payload_bytes, request_wire_bytes
network_exchange_ms, server_receive_queue_ms, deserialization_ms
cloud_image_prefill_ms, cloud_prompt_prefill_ms, cloud_decode_ms
response_wire_bytes, observed_end_to_end_ms
edge_peak_allocated_mib, edge_peak_reserved_mib, edge_process_rss_mib
cloud_peak_allocated_mib, cloud_peak_reserved_mib, cloud_process_rss_mib
output_token_hash, quality_metric, quality_score
```

Use a long-form typed JSONL source of truth so failures, nulls and nested provenance are represented without CSV ambiguity; produce a flat CSV as an export. Timing boundaries must be defined in a schema note. The current `language_generation_ms` combines prompt prefill and decode; an extra instrumented boundary is needed. Remote `server_model_*` metrics are server-side durations; the client round trip already contains server work. Do **not** sum both as if independent. The current `network_round_trip_ms` similarly includes server processing. Record queue/wait separately where possible and calculate a residual transport term as client exchange minus server processing with clock-independent durations; flag negative residuals as measurement error rather than silently replacing them with zero. The synchronous path can initially use an additive model; `prompt_first` needs a critical-path/overlap model calibrated from observations.

For `cloud/raw_image`, use the raw image plus metadata as uplink and answer frame as downlink. For `distributed/projected`, the current BF16 payload alone is `729 × 2048 × 2 = 2,985,984` bytes; full framed request and response are larger. In the network model, use uplink and downlink separately when measured:

`L_hat = edge_compute + serialization + protocol/queue + RTT_effective + 8*request_wire_bytes/B_up + 8*response_wire_bytes/B_down + deserialization + cloud_compute`.

`RTT_effective` is empirical protocol/setup overhead and must be calibrated against the observed TCP exchange (one connection per `exchange_frame`), not blindly assumed to be exactly one ping RTT. For edge-only, all remote terms are zero. Include scheduler/monitor overhead in observed end-to-end results. Fit lookup/median or simple regression by candidate, task bucket and hardware profile; interpolate only inside measured coverage and explicitly report out-of-distribution cases. A separate preflight memory estimate uses conservative observed peaks plus headroom, not instantaneous free VRAM alone.

## 3. Phases and passing gates

Each gate is checked in order. A failed gate produces an artifact explaining the failed condition and blocks claims or activation that depend on it. CPU-safe synthetic tests can run in CI; real-model and physical-network gates require the named hardware and preserved raw artifacts.

### Phase 0 — Freeze and verify the reference baseline

**Work.** Record the Git commit, model artifact/fingerprint, dependency lock, current hardware inventory and exact benchmark manifest. Re-run CPU-safe checks. On the existing local host, repeat correctness for edge/cloud/distributed synchronous with the same pinned requests and inspect `complete_and_equal`, failure counts and token hashes. Separately provision the actual cloud host, verify its inventory, cache and reachable protected transport, then reproduce the three modes on two physical hosts. Keep single-host and physical-host findings distinct. Capture actual bandwidth/RTT independently. Archive run directories and environment manifests.

**Gate G0-local.** Ruff, strict mypy and pytest/coverage pass; a real-model same-device correctness run has all planned pairs present, zero failed measured requests and `complete_and_equal=true`. The run record names the hardware and manifest hashes. **Gate G0-physical for external validity.** The same checks pass on the intended host pair, with verified cloud inventory and measured network state. If the physical host is unavailable, development may continue against local/replay profiles, but no physical edge–cloud performance claim passes.

### Phase 1 — Lock the executable configuration space

**Work.** Implement typed configuration/state/decision contracts and deterministic candidate IDs. Build a capability registry from loaded executors; validate matching model contract, server request paths and device availability. Resolve how to keep `cloud/raw_image` and `distributed/projected` available at the same time: either a tested full-model server accepting both message types, or separate preloaded server endpoints with explicit routing. Measure any extra memory/load contention. Avoid per-request model reload; if impossible, model switch/load latency as part of the configuration and disclose it. Keep `prompt_first` disabled until its overlap behavior is profiled. Add command/config examples for manual state and three-policy selection. Preserve existing fixed commands.

**Gate G1.** For each registered candidate, a deterministic smoke test calls exactly its intended executor and returns the correct configuration ID; unavailable server/model/memory states exclude it with a reason. At least `edge/full`, `cloud/raw_image`, and `distributed/projected/synchronous` can be available concurrently in the intended deployment without a per-request restart. On the pinned correctness set, all three produce the baseline token IDs on the same numerical placement; cross-device differences are recorded rather than concealed. Existing fixed-policy tests still pass.

### Phase 2 — Instrument, profile, and define internal-split feasibility

**Work.** Define timing boundaries and add the missing prompt prefill/decode, serialization/deserialization, response-byte, memory and scheduler-overhead measurements. Implement an offline profiler that uses `make_schedule`/frozen manifests, excluded warmups and resumable per-trial records. Sweep a predeclared small task grid: image sizes present in the manifest, prompt-token buckets, and expected output-length bins declared before execution, keeping the pinned `max_new_tokens=32` for every initial run. Include controlled bandwidth/RTT cases, repeated sweeps, and failures; randomize/interleave configurations or block by condition to reduce drift. Validate actual prompt token counts with the pinned tokenizer. Record actual output length separately for calibration/analysis; do not feed the realized length of a held-out request into its pre-execution prediction.

**Internal split discovery (research branch).** Inspect the pinned Moondream computation graph and shortlist stable boundaries such as vision encoder sub-blocks, encoder output before projector, and existing projected features. For each, document tensor shape/dtype/bytes, required side inputs and skip connections, edge/cloud model partition, server handler, reconstruction fidelity and host memory. The current fixed `(729, 2048)` `ProjectedVisionFeatures` and `TENSOR_REQUEST` semantics are specific to the existing split; a new boundary needs its own validated payload contract and protocol/capability version. Do not name an internal split executable because it can be observed in a hook alone.

**Gate G2.** The profiler emits versioned, hash-linked rows with no ambiguous units, no silent missing values and exact wire-byte accounting. Instrumentation overhead is measured and reported; stage accounting is reconciled against request duration with a documented residual on every successful sample. For each declared supported task/network bucket, require at least 10 successful measured trials per baseline configuration drawn from at least three distinct request IDs; all other buckets are marked uncovered. Failures remain in the raw data and failure rates are reported. Each activated new split passes tensor round-trip and end-to-end token equivalence on the correctness manifest and fits payload/memory limits. Otherwise the V1 registry remains at three candidates.

### Phase 3 — Build and calibrate a simple predictor

**Work.** Implement the component model above using profile medians or a small regression, with explicit host-pair and model fingerprint matching. Compare against a constant-policy median predictor. Hold out whole request IDs and at least one network-condition block from fitting; test interpolation and unseen conditions separately. Report median and 90th-percentile absolute percentage error, signed bias, interval coverage and ranking accuracy per candidate/task/network bucket. Avoid percentage error for near-zero components; evaluate total latency where denominators are meaningful. Include real serialization cost and response transfer, which the source framework formula omits.

**Gate G3.** On a predeclared held-out **in-distribution** set, target total-latency median APE ≤15% and p90 APE ≤30%, with no worse ranking accuracy than the constant-median baseline. These are proposed engineering targets, not a claim that hardware noise will meet them; if they fail, collect data or revise the model and rerun a fresh holdout. Out-of-coverage inputs are flagged and trigger a documented safe fallback. Predictor unit tests verify units, monotonicity of transfer time with bandwidth, deterministic outputs, missing-state handling and memory exclusion.

### Phase 4 — Establish independent offline oracle and scheduler V1

**Work.** Implement `select` as a pure enumeration/filter/minimum operation. Create two explicitly different references: a **predicted oracle** (minimum predictor output, useful for checking selector logic) and a **measured oracle** (minimum *observed* matched latency for each request/state among all feasible candidates, computed from held-out executions). The measured oracle must never use a candidate outcome to decide that same live request. For noisy trials, compare paired condition blocks and report oracle uncertainty. Compute regret in milliseconds and as a percentage of measured oracle latency; also report top-1 agreement, SLA violations, exclusions, and no-feasible cases. Preserve all candidate predictions and reasons in the decision log.

**Gate G4.** Synthetic fixtures exhaustively test ties, invalid/NaN predictions, missing profiles, OOM/health exclusions, SLA feasible/unmet, and zero feasible candidates. Predicted oracle and scheduler agree on every deterministic fixture. On held-out measured observations, report selection agreement/regret with confidence intervals; target median regret ≤10% of oracle latency and no unsafe execution. A regret miss blocks performance claims and requires calibration or policy revision, not a hidden threshold change.

### Phase 5 — Execute adaptive decisions end to end

**Work.** Add an adaptive CLI/benchmark mode that derives task features, accepts manual/replay state, selects once, dispatches once, and records prediction plus actual result. Retain a stable decision even if health changes mid-request; on a pre-inference failure, make at most one explicit safe retry with a fresh decision and record both attempts and added latency. Extend adaptive run analysis instead of applying `compare-runs` directly, because that comparator requires distinct fixed modes and matching fixed schedules. Preserve fixed-policy runs as controls. Keep same prompt, preprocessing, precision and generation parameters for quality equality.

**Gate G5.** A mixed-choice run on the same frozen manifest has exactly one completed result or explicit failure per scheduled trial, a reproducible decision trace, and no unlogged fallback. For supported same-device paths, adaptive output tokens match the fixed-policy reference for every successful paired request; any cross-device mismatch is disclosed by placement. The run includes selection overhead in observed latency, correct wire bytes for the selected mode, and all expected manifest/contract hashes. Fixed-policy CLI and benchmark behavior remains intact.

### Phase 6 — Add live sensing and robustness

**Work.** First replay known bandwidth/RTT traces through the selector. Add bounded passive throughput estimates and an active RTT probe or application-level health probe; distinguish probe RTT from transfer goodput. Record sampling interval, smoothing method, age, timeout and provenance. Add hysteresis only if measured switching instability occurs, and count switching/load costs. Later, add edge free/peak memory and cloud queue/load if obtainable from the server; avoid pretending process RSS is free GPU memory. Define conservative fallback for stale/failed probes, missing cloud health and extrapolated task state.

**Gate G6.** Deterministic trace tests show the expected configuration transition when a measured crossover is crossed, zero switches during a predeclared constant-condition noisy trace after warmup, and safe behavior on stale/missing probes. Live runs record probe values and timestamps alongside every decision. A monitoring failure never selects a candidate with unknown memory safety or an unavailable endpoint. Report monitor overhead and its effect on latency.

### Phase 7 — Task-aware constraints and thesis evaluation

**Work.** Expand the request set beyond synthetic fixtures to a frozen VQA v2 validation subset and at least one predeclared short/long response task if supported by the pinned model. The current model cap is 32 new tokens: a 100/300-token study requires a separately pinned generation contract and fresh equivalence/profiles, and should not be mixed into the 32-token campaign. Derive prompt length and task type before execution; estimate output length without looking at generated outputs. Use the official VQA evaluator for VQA accuracy. If configurations execute the same model/precision and differ only by exact placement, expect equal quality; report equality rather than claiming a quality tradeoff. Add a quality floor only if a validated optimization changes quality. Energy is optional and requires a measured sensor and boundary definition.

**Evaluation design.** Predeclare workload, network regimes, hardware, warmups, repetitions, randomization/blocking, exclusions, confidence intervals and primary metrics. Compare adaptive V1 to always-edge, always-cloud, fixed projected split and a simple best-static validation choice. Evaluate under controlled network traces and, when G0-physical is met, on the physical host pair. Use held-out request IDs and network blocks; keep all methods on the same model/generation contract and account for failures. Report latency distribution (median/p95), transfer bytes, SLA success, peak edge memory, selector overhead, accuracy/equivalence, prediction error, oracle regret and decision distribution. A negative performance result is still a valid outcome if the experiment is complete and reported faithfully.

**Gate G7.** All predeclared cases have archived raw rows, hashes, verified environment/network metadata, failure counts and a reproducible analysis script/table. The report includes paired uncertainty, fixed-policy comparisons and measured-oracle regret without training/test leakage. No quality or physical-network claim is made without its corresponding official evaluator or G0-physical evidence. The thesis can conclude either improvement or a measured limit of adaptation.

## 4. Minimum implementation order and dependency map

```text
G0-local ──► G1 registry/dispatch ──► G2 profiling ──► G3 predictor
                                          │                    │
                                          └─► internal splits   ▼
                                               (optional)   G4 oracle/selector
                                                               │
                                                               ▼
                                                       G5 adaptive run
                                                               │
                                                               ▼
                                                       G6 live state ──► G7 evaluation
                                                                         ▲
G0-physical ──────────────────────────────────────────────────────────────┘
                                         (required for physical-host claims)
```

Start with the three existing executable placements and manual/replayed state. This produces a complete scheduler experiment before taking on model surgery. Promote an internal split only after its G2 equivalence, payload, memory and end-to-end execution evidence exists; then repeat profile calibration and oracle evaluation with the expanded candidate set. Keep `prompt_first` separate for the same reason. Learned prediction or RL is a later comparison only if the simple model fails a predeclared goal and there is enough independent training data.

## 5. Immediate work package

1. Run G0-local and record the hashes/hardware/results in an ignored artifact directory; provision the physical cloud inventory when available.
2. Implement `ConfigurationSpec`, capability checks and a three-policy adaptive dispatcher, including a simultaneous cloud/split serving strategy.
3. Define and validate profile/decision schemas, then add the missing timing hooks and the offline profiling command.
4. Freeze the task/network sweep and predictor/oracle evaluation protocol **before** fitting or selecting thresholds from observed results.
5. Complete G3–G5 with manual/replayed state; only then enable live sensing and the broader task-quality study.

The source framework document supplies the conceptual sequence. Repository code and README supply implementation facts; the quantitative gates and module names above are proposed engineering decisions to validate during this phase.
