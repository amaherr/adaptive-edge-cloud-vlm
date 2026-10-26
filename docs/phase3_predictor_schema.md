# Phase 3 component predictor contract

`adaptive-vlm calibrate-predictor` reads a Phase 2 `profile.jsonl` and a frozen train/holdout specification. It verifies the profile SHA-256 in the specification, exact model/dataset/host identity, disjoint request IDs, and enough successful training rows for all three baseline configurations. Warmups and failed trials do not fit coefficients; held-out request IDs and declared held-out network buckets never enter fitting. The command writes a hash-identified `calibration.json`, `evaluation.json`, `environment.json`, and one `predictions.jsonl` row per held-out attempt. It refuses an existing output directory.

## Predictor inputs and coverage

`Predictor.predict(spec, state, task)` returns a typed `Prediction`. `state` carries host-pair ID, model-contract SHA-256, placement-profile SHA-256, network bucket/source, uplink/downlink goodput in **bits/s**, RTT in **ms**, and optional edge memory budget in **MiB**. `task` carries a declared task bucket, image bytes and dimensions, exact prompt tokens, the pinned generation budget, and an optional *pre-execution* output-length estimate.

The predictor checks host/model/placement and generation contract first, then the training task bucket and min/max feature bounds. A missing exact prompt-token count, unseen task, or unsupported network condition returns `valid=false` with a named reason. The local profile supports only `loopback-uncontrolled` with null bandwidth/RTT. Giving it a claimed goodput or RTT does not silently extrapolate. A CPU edge profile has no measured per-request peak memory requirement; if a memory budget is supplied, edge and split candidates return `edge_memory_requirement_unknown`. When a CUDA reserved-memory peak exists, the fitter stores its largest training observation plus the predeclared headroom.

Invalid candidates must be excluded by the Phase 4 selector. If no configuration has both a valid prediction and proven memory/executor availability, selection returns a typed failure. This Phase 3 module does not execute a candidate or silently choose an unprofiled fallback.

## Component equation

Each candidate uses medians from successful **training** rows. Edge uses non-decode model work plus predicted decode. Cloud uses image file read, server non-decode work, predicted decode, transport and client residual. Split adds edge vision and tensor serialization before the server work. The non-overlapping client residual comes from `observed_end_to_end_ms - stage_sum_ms`; `server_processing_ms` is never added again to `network_exchange_ms`.

`decode_ms_per_token` is the median training `decode_ms / actual_output_tokens` where output length is positive. If a task supplies no pre-execution estimate, the predictor uses the **training median** output length and labels that choice. The evaluator always passes no output estimate. It reads held-out `actual_output_tokens` only as an outcome after prediction, so generated output cannot leak into task features.

For cloud, predicted request bytes are image bytes plus the training median framing/metadata cost. For split, they are the measured projected payload plus framing/metadata cost. Response bytes use a training median. For controlled network data, the model uses:

`transport_ms = fitted_intercept_ms + fitted_rtt_coefficient × RTT_ms + 8000 × request_wire_bytes / uplink_bps + 8000 × response_wire_bytes / downlink_bps`

The RTT coefficient is fitted from median transport residuals across measured RTT conditions after removing both transfer terms, constrained to be nonnegative. Predictions are valid only inside the measured bandwidth and RTT bounds. A single uncontrolled loopback bucket instead uses its observed median `network_exchange_ms - server_processing_ms`; it has no calibrated physical transfer coefficient. The standalone `transfer_ms` function verifies units and monotonicity.

`lower_ms` and `upper_ms` are total prediction ± the 90th percentile absolute **training** residual. Their held-out coverage is reported, but these intervals are not statistically calibrated on independent data. The constant-policy comparison is the training median observed total for each configuration.

## Evaluation and gate interpretation

The evaluator reports median and p90 total-latency absolute percentage error, mean signed bias, interval coverage, constant-median APE, invalid/failure counts, and paired ranking accuracy for each candidate/task/network bucket. Request-ID holdout, network-condition holdout, and their intersection have separate sections. Ranking compares predicted and observed winners only when all three candidates completed and have valid predictions. Ties use stable configuration IDs.

`local_request_holdout_targets_met` applies the plan's 15% median APE, 30% p90 APE, and no-worse-than-constant ranking targets to the declared local request holdout. It is diagnostic with the tiny fixture set. `full_g3_gate_evaluable` requires a declared, paired network-condition holdout; `full_g3_gate_passed` cannot be true without it. An out-of-coverage condition remains invalid and is reported rather than counted as a successful prediction.
