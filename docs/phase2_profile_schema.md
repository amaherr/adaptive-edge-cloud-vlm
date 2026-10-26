# Phase 2 profile contract (schema version 1)

`adaptive-vlm profile-runs` converts completed, resumable `benchmark` runs into an immutable `profile.jsonl` and `summary.json`. The command requires a predeclared specification such as [`phase2_local_fixture.json`](../configs/profiles/phase2_local_fixture.json). It rejects an existing output directory, incomplete schedules, duplicate trials or modes, mismatched model/config/manifest hashes, image hashes, token counts, and invalid wire-byte bounds. Warmups stay in the raw profile with `warmup=true` but are excluded from coverage and failure-rate denominators. Failed measured attempts stay in both the rows and the denominators.

## Timing boundaries

Every duration ends in `_ms` and uses a monotonic clock. Bytes are exact binary frame lengths or file/payload sizes. Memory fields ending in `_mib` use 1,048,576 bytes per MiB. `prompt_tokens` counts the actual pinned tokenizer/template IDs handed to Moondream prefill, including template prefix and suffix. `actual_output_tokens` is the length generated after execution; the pre-execution budget is `requested_max_new_tokens=32`. Actual output length must never become a predictor input for that same held-out trial.

| Field | Boundary |
| --- | --- |
| `observed_end_to_end_ms` | Benchmark timer around one `policy.infer` call, including client orchestration and serialization. Model loading and warmup happen outside it. |
| `edge_vision_ms`, `edge_projector_ms` | Timed calls inside the edge vision graph. Their sum excludes preprocessing and other vision work. |
| `edge_total_ms` | Edge-only model call duration, or the complete edge vision call for distributed mode. |
| `serialization_ms`, `deserialization_ms` | Projected tensor serialization on the edge and tensor reconstruction on the server. |
| `network_exchange_ms` | Client TCP connect, send, server processing, response receipt and frame decoding. It **contains** `server_processing_ms`; do not add both. |
| `network_connect_ms`, `network_send_ms`, `network_receive_ms` | Subintervals of the exchange. Send includes frame encoding and `sendall`; receive includes waiting for inference and response frame decoding. |
| `server_processing_ms` | Server handler duration after receiving a full frame through model execution and response object creation. TCP accept queue, full frame receive, and response socket send are outside it. |
| `transport_residual_ms` | `network_exchange_ms - server_processing_ms`. This contains client/server framing, transfer, connection and any unobserved queue time. A negative result is reported as a measurement issue, never clipped. |
| `prompt_prepare_ms` | Prompt tokenization and prompt tensor creation (or tensor creation only if `prompt_first` supplied prepared IDs). |
| `prompt_prefill_ms` | Model prompt prefill, including its first predicted token. |
| `decode_ms` | Remaining token loop, ending with device synchronization where applicable. `decode_steps` remains in `raw_metrics`. |
| `image_prefill_ms` | Explicit projected-feature image prefill for edge/distributed. All-cloud uses `image_encode_and_prefill_ms`, a combined pinned-model operation. |
| `stage_sum_ms`, `stage_residual_ms` | Sum of *non-overlapping outer* stages and `observed_end_to_end_ms - stage_sum_ms`. Edge sums its complete model call; cloud sums image file read and TCP exchange; distributed sums edge vision, tensor serialization and TCP exchange. The residual includes Python orchestration, response parsing, decoding, and timer effects. It may be negative and is retained. |
| `scheduler_overhead_ms` | Null for these fixed-policy benchmark runs with reason `fixed_policy_benchmark_no_scheduler`. The Phase 1 manual dispatcher now separately records `selection_overhead_ms`; the eventual automatic scheduler will need its own timing in Phase 5. |

`measurement_call_overhead_ms_median_cpu` in `summary.json` is a 1,000-call no-op calibration of the timing wrapper on the profiling machine. It estimates only the CPU call/timer cost, not CUDA synchronization, model perturbation or per-request cumulative overhead. The stage residual is the per-request accounting check. Further controlled A/B instrumentation work is required before claiming a full overhead bound on another device.

## Nulls and provenance

Each row has `schema_version`, campaign, trial/request/configuration IDs, run and plan hashes, a canonical run-manifest hash, Git commit/dirty flag, dependency lock hash, model/config/dataset hashes, measured UTC start, host pair and placement, task/network buckets, status/failure reason, token hash, observed duration, normalized `fields`, `null_reasons`, and original `raw_metrics`. Every null in `fields` has a reason. Inapplicable remote fields on edge, unavailable CUDA peaks on CPU, the unobservable sequential-server queue, and the combined all-cloud image prefill are distinguished. `rss_after_mib` is a process reading after inference, **not** a per-request peak. CUDA `max_memory_*` is reset per request in the model adapter. The existing runtime does not measure a CPU RSS high-water mark per request.

The profile specification names the network bucket and explicitly marks whether its bandwidth/RTT were controlled. The local fixture specification has null bandwidth and RTT because localhost timings are not physical-network calibration. `summary.json` reports per-configuration coverage (at least 10 successful measured trials and three request IDs), failures, negative transport residuals, complete successful triples and token/prompt equality. The local and controlled-network gate flags are separate. A caller must additionally inspect the specification and environment evidence before making any physical-host claim.

`output_token_hash` is SHA-256 of the compact JSON token-ID list; it is suitable for checking equality across configurations, not for reconstructing output. The benchmark `requests.jsonl` retains the full token IDs and exact per-attempt error text.
