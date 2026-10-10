# Baseline implementation plan

## 1. Goal and definition of done

Build one reproducible inference system for a single pinned pretrained VLM with
three fixed execution policies:

1. `edge`: image preprocessing, vision encoder, projector, and language model
   execute on the edge host.
2. `cloud`: the edge sends the original image and prompt; the complete VLM
   executes on the cloud host.
3. `distributed`: image preprocessing, vision encoder, and projector execute
   on the edge; the projected visual representation and prompt are sent to the
   cloud; the language model executes on the cloud.

The milestone is complete only when all three policies use the same model
revision, preprocessing, prompt format, precision, decoding parameters, and
request manifest; pass the correctness gates below; and produce repeatable,
machine-readable benchmark results.

Adaptive scheduling, quantization, feature compression, token pruning,
dynamic batching, multiple edge nodes, and internal LLM splits are explicitly
out of scope.

## 2. Decisions to freeze before implementation

Record the following in a version-controlled experiment configuration. Do not
hide any of them in code defaults.

| Decision | Initial value | Rule |
| --- | --- | --- |
| Python | 3.11 | Pin the exact patch release in the environment record |
| Model family | Moondream2 candidate | Finalize only after the split-feasibility spike |
| Model revision | Exact Hugging Face commit/release, not `main` | Same artifact in every mode |
| Precision | BF16 if both GPUs support it; otherwise FP16 | No quantization in baseline |
| Batch size | 1 | Fixed for latency experiments |
| Decoding | Greedy, sampling disabled | Deterministic settings |
| Output length | `max_new_tokens: 32` initially | Fixed across policies |
| Image transform | Model's canonical transform | Run exactly once per request |
| Dataset | Frozen manifest of image/prompt/answer records | Same ordered IDs in every mode |
| Warm-up | 10 requests | Excluded from reported results |
| Measured repetitions | At least 30 per condition | Report median, p95, and dispersion |
| Protocol | Versioned, length-prefixed TCP | No pickle or Python object transfer |

Create `configs/model/moondream2.yaml`, `configs/hardware/*.yaml`, and
`configs/experiments/*.yaml`. Each completed run must copy the resolved config
into its result directory.

### Model selection gate

Do not commit the project to a model merely because monolithic inference works.
The model must pass this spike:

- a pinned revision can be loaded without modifying weights;
- image preprocessing can be invoked explicitly;
- the output after the projector (the visual embeddings consumed by the LLM)
  can be captured as a tensor;
- generation can resume from that tensor without re-running the vision stack;
- the vision stack/projector and LLM can be placed in separate processes;
- local monolithic and locally decomposed greedy generation return identical
  token IDs on the same hardware and dtype;
- the projected tensor's shape, dtype, and byte count are known;
- the licenses of the model, code, and evaluation data permit thesis use.

Start the spike with `vikhyatk/moondream2` because it is small and exposes an
image-encoding API, but pin a revision only after inspecting and testing its
actual remote-code implementation. Treat `trust_remote_code=True` code as a
vendored, reviewed dependency: record its commit hash and hash the downloaded
files. If the split cannot be implemented cleanly in two working days, test a
small Hugging Face-native LLaVA-family checkpoint and choose the model with the
most stable explicit component boundary. This is a research-infrastructure
decision, not a model-quality comparison.

### Hardware gate

The final plan requires the exact edge and cloud machines. Record at minimum:

- CPU model, core count, RAM, OS, and kernel;
- GPU model, VRAM, driver, CUDA, and PyTorch versions;
- storage type and available space;
- network interface, nominal link speed, and physical topology;
- whether clocks are synchronized and by what mechanism;
- idle and experiment-time GPU temperature/utilization where available.

The Windows host has an NVIDIA GeForce RTX 3050-class GPU with 4,096 MiB VRAM,
a 75 W power limit, WDDM driver 610.62, and CUDA UMD 13.3. The current WSL
environment exposes Python 3.12 and about 8 GiB of RAM but cannot currently
access that GPU through `nvidia-smi`. Use a dedicated Python 3.11 environment
for the first compatibility pass and repair WSL GPU passthrough before GPU
experiments. The 4 GiB VRAM limit makes all-edge BF16/FP16 feasibility a real
go/no-go test; do not assume the approximately 2B-parameter model will fit once
runtime allocations and KV cache are included. This WSL instance is suitable
for repository work and CPU-only unit tests but must not be treated as the
measured edge environment until CUDA access is verified from PyTorch.

## 3. Proposed repository layout

Use one package and one model adapter rather than separate implementations for
each policy.

```text
adaptive-edge-cloud-vlm/
├── pyproject.toml
├── README.md
├── configs/
│   ├── model/
│   ├── hardware/
│   └── experiments/
├── src/adaptive_vlm/
│   ├── cli.py
│   ├── config.py
│   ├── models/
│   │   ├── base.py
│   │   └── moondream2.py
│   ├── execution/
│   │   ├── base.py
│   │   ├── edge.py
│   │   ├── cloud.py
│   │   └── distributed.py
│   ├── transport/
│   │   ├── framing.py
│   │   ├── messages.py
│   │   ├── client.py
│   │   └── server.py
│   ├── benchmark/
│   │   ├── timing.py
│   │   ├── memory.py
│   │   ├── recorder.py
│   │   └── runner.py
│   └── data/
│       ├── manifest.py
│       └── preprocessing.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
│   ├── shape_network.sh
│   ├── collect_environment.sh
│   └── summarize_results.py
├── docs/
└── artifacts/                 # gitignored; configs and summaries retained
```

Expose a single user-facing CLI:

```bash
adaptive-vlm serve --config configs/experiments/cloud.yaml
adaptive-vlm run --mode edge --config configs/experiments/baseline.yaml
adaptive-vlm run --mode cloud --config configs/experiments/baseline.yaml
adaptive-vlm run --mode distributed --config configs/experiments/baseline.yaml
adaptive-vlm benchmark --matrix configs/experiments/network_matrix.yaml
```

The three policies implement the same request/result interface. A future
adaptive policy should only select one of them; it should not require a new
model or benchmark path.

## 4. Core interfaces

Define contracts before implementing policies.

```python
@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    image_path: Path
    prompt: str

@dataclass
class VisualPayload:
    tensor: torch.Tensor
    model_fingerprint: str

@dataclass
class InferenceResult:
    request_id: str
    output_text: str
    output_token_ids: list[int]
    metrics: dict[str, int | float | str | None]

class VLMAdapter(Protocol):
    def preprocess_image(self, image: Image.Image) -> Any: ...
    def encode_vision(self, model_input: Any) -> torch.Tensor: ...
    def project_vision(self, features: torch.Tensor) -> torch.Tensor: ...
    def generate_from_visual(self, visual: torch.Tensor, prompt: str) -> InferenceResult: ...
    def generate(self, image: Image.Image, prompt: str) -> InferenceResult: ...

class ExecutionPolicy(Protocol):
    def infer(self, request: InferenceRequest) -> InferenceResult: ...
```

If the selected model combines vision encoding and projection in one public
method, the adapter may expose `encode_and_project` internally. However, it
must document that limitation, capture the true transmitted boundary, and
must not invent separate component timings that the implementation cannot
measure reliably.

## 5. Wire protocol

Use a protocol version plus fixed-width header length, UTF-8 JSON metadata,
and zero or more binary payloads. Do not use pickle.

Every message includes:

- magic bytes and protocol version;
- message type (`raw_request`, `visual_request`, `response`, `error`, `ping`);
- request ID;
- model and model-revision fingerprint;
- payload length(s), maximum permitted length, and checksum;
- tensor shape, dtype, byte order, and layout for visual payloads;
- prompt/image encoding metadata;
- generation settings hash;
- structured error information.

Implement `recv_exact(n)`, socket timeouts, clean disconnect handling, bounds
checks before allocation, and a protocol-level compatibility rejection. The
server must reject a visual tensor generated by a different model revision or
generation contract.

For `cloud`, transmit the exact encoded image bytes from the frozen dataset
plus the prompt. This makes `input_bytes` well defined. Server-side decoding
and preprocessing are included in server latency and recorded separately when
possible.

For `distributed`, transmit the prompt and the contiguous projected tensor in
its native baseline dtype. Record metadata bytes and payload bytes separately.
No feature compression or dtype conversion is permitted in the baseline.

## 6. Measurement contract

### Timing boundaries

Use `time.perf_counter_ns()` for wall-clock intervals. Synchronize the relevant
CUDA device immediately before and after a measured GPU region, or use CUDA
events for device-only component time. Record both when useful and name them
unambiguously.

Capture these fields where applicable:

- request preprocessing and image decoding;
- edge vision and projection;
- serialization and deserialization;
- bytes sent and received at application level;
- server queue time;
- cloud vision and projection;
- LLM prefill, decode, and total generation time if the model permits it;
- response serialization;
- client-observed end-to-end latency;
- input token count, visual token count, output token count, and tokens/s;
- edge/cloud peak CUDA allocated and reserved memory;
- edge/cloud process RSS peak;
- outcome (`ok`, `timeout`, `oom`, `protocol_error`, or `model_error`).

Do not claim independent one-way upload/download latency from unsynchronized
client and server clocks. The authoritative network-facing metric is
client-observed end-to-end time; server durations are returned as metadata.
If one-way estimates are required, synchronize clocks (for example with PTP or
chrony), measure the remaining offset, and report its uncertainty. Otherwise
report transfer/acknowledgement RTT-style spans and configured network delay.

### Result schema

Write one JSONL or Parquet row per request and a separate run manifest. CSV may
be exported for analysis but should not be the canonical nested schema.

The run manifest contains:

- run ID and UTC timestamp;
- git commit and dirty-worktree flag;
- resolved configuration and random seeds;
- model ID, revision, artifact fingerprint, and precision;
- Python/package lock hash, OS, hardware, CUDA, and driver information;
- dataset-manifest hash;
- edge/cloud host IDs and protocol version;
- network condition and shaping command;
- warm-up count and measured repetition count.

Never average failed/OOM requests into latency. Report feasibility and failure
rates separately.

### Benchmark procedure

1. Reboot or restore a documented idle state when practical.
2. Start the server and verify model fingerprints with a health check.
3. Run 10 unmeasured warm-ups using the same request shape distribution.
4. Run at least 30 measured requests per condition; use more if confidence
   intervals remain unstable.
5. Interleave or randomize policy/condition order to reduce thermal and
   time-of-day bias.
6. Store raw per-request observations; never retain only aggregates.
7. Report median, p95, interquartile range, bootstrap 95% confidence interval,
   failure rate, and sample count. Mean and standard deviation may be included.
8. Run the all-edge condition once per hardware/software configuration; do not
   pretend network shaping affects it.

Cold model-load/startup time is a separate deployment metric and is excluded
from steady-state inference latency.

## 7. Dataset and correctness strategy

Use three tiers:

1. **Unit fixtures:** tiny deterministic tensors and 2-3 small repository-safe
   images for protocol and preprocessing tests.
2. **Correctness manifest:** 10-20 fixed image/prompt pairs that exercise short
   answers, descriptions, counting, OCR-like text, and differing prompt lengths.
3. **Benchmark manifest:** initially 100, then 500 or more fixed records drawn
   from a recognized VQA validation set. Store IDs, prompts, references, and
   hashes—not redistributed images unless licensing permits it.

Correctness has four gates:

- **G1 tensor split:** monolithic visual embeddings equal locally decomposed
  embeddings within a declared tolerance.
- **G2 serialization:** deserialized tensor has identical dtype, shape, bytes,
  and checksum; numerical values are exactly equal.
- **G3 same-host policies:** on the same device/dtype with greedy decoding,
  edge, cloud, and distributed produce identical token IDs.
- **G4 real hardware:** compare actual edge/cloud outputs. Exact token equality
  is preferred, but cross-device floating-point divergence is possible. Report
  exact-match rate plus a predeclared task metric; investigate every mismatch
  rather than silently relaxing the test.

For the baseline thesis result, add an established task metric appropriate to
the selected dataset (for example VQA accuracy) so that “preserving model
performance” is measured, even though the policies should be semantically
equivalent.

## 8. Work breakdown and acceptance gates

Effort estimates are focused working days and assume access to both target
machines. Total expected effort is roughly 33-46 days (about 7-9 weeks), with
hardware access, model internals, and reproducibility being the critical path.

### M0 — Freeze scope and experiment contract (2-3 days)

Tasks:

- document edge/cloud hardware and network topology;
- select initial data and licensing strategy;
- create config schemas, result schema, and metric definitions;
- specify exact preprocessing and decoding settings;
- create a risk log and experiment naming convention.

Exit gate: one checked-in resolved example configuration can completely
describe a run; no baseline-specific setting remains implicit.

### M1 — Model feasibility spike (3-5 days)

Tasks:

- create the Python package and locked development environment;
- load the candidate pinned model and run deterministic monolithic inference;
- inspect the model graph and identify the actual projected-visual boundary;
- execute local `vision -> projector -> LLM` inference;
- measure tensor shape/bytes and component memory;
- write an architecture decision record choosing the checkpoint/revision.

Tests: repeated monolithic output, local split token equivalence, component
device placement, no-grad/inference mode, and model fingerprint stability.

Exit gate: all model-selection criteria in Section 2 pass. If they do not,
change model before writing the transport layer.

### M2 — Shared local runtime and instrumentation (4-5 days)

Tasks:

- implement `VLMAdapter`, request/result objects, config validation, and seeds;
- implement synchronized timers and process/GPU memory measurement;
- implement canonical structured result recording;
- implement the `edge` execution policy;
- distinguish model load, warm-up, prefill, decode, and end-to-end times.

Tests: CPU timer test, CUDA timer test when available, schema validation,
repeated inference, bounded memory growth, OOM captured as a result rather than
a crashed benchmark.

Exit gate: all-edge produces correct saved outputs and trustworthy metrics, or
is reproducibly marked infeasible on the target edge hardware.

### M3 — Transport and all-cloud policy (4-6 days)

Tasks:

- implement and fuzz the framing/serialization layer;
- implement server lifecycle, health check, request timeout, and graceful error
  response;
- implement raw image/prompt request and response paths;
- return server-side timing and resource metrics;
- exercise localhost first, then two physical hosts.

Tests: partial reads/writes, truncated/oversized frames, disconnects, wrong
protocol/model version, concurrent connection rejection or handling, localhost
cloud equivalence, and real-link smoke test.

Exit gate: all-cloud is token-equivalent on the same-host gate, completes 100
sequential requests without leakage/deadlock, and accounts for all application
bytes.

### M4 — Fixed distributed policy (6-9 days)

Tasks:

- load only vision/projector-required weights on the edge when the model permits
  it and only LLM-required weights on the cloud;
- serialize and transfer the projected tensor without conversion;
- combine prompt and visual representations exactly as monolithic inference
  does;
- add model/generation fingerprint enforcement;
- first implement synchronous transfer;
- after synchronous correctness passes, implement the paper-aligned overlap:
  send prompt metadata first, prepare/tokenize text on the cloud while the edge
  computes vision/projector output, then join the visual message by request ID;
- bound pending request state with timeouts and cleanup, and expose overlap as
  an explicit distributed-pipeline setting rather than a fourth policy.

Tests: tensor round-trip bit equality, wrong shape/dtype rejection, local and
network split token equivalence, 100 sequential requests, memory stability,
and clean recovery from a killed client.

Exit gate: the cloud does not recompute vision, the edge does not execute the
LLM, and the result passes G1-G4 with complete component metrics.
Synchronous and overlapped variants must generate the same tokens. Report both
once to quantify the overlap benefit, then declare which one is the primary
distributed baseline for all later comparisons.

### M5 — Cross-policy validation and hardening (4-5 days)

Tasks:

- run the full correctness manifest in all feasible modes;
- add integration tests that launch a real server process on an ephemeral port;
- verify config/model/dataset hashes in every result;
- profile unexplained timing gaps and define an acceptable accounting residual;
- create a one-command smoke test and documented failure troubleshooting.

Exit gate: CI passes CPU-only unit tests, hardware integration tests pass on
both machines, outputs meet the declared equivalence policy, and every original
baseline requirement R1-R18 has linked evidence.

### M6 — Controlled benchmark campaign (5-7 days)

First run an unshaped LAN reference condition. Then test:

- bandwidth: 10, 25, 50, 100, 250, and 1000 Mbit/s where supported;
- RTT: 5, 20, 50, and 100 ms;
- policies: cloud and distributed at every network condition, plus the single
  all-edge reference;
- at least 30 measured requests per cell after warm-up.

Use Linux traffic control (`netem` for delay/loss and an appropriate rate
limiter such as TBF/HTB) on the explicitly identified interface. Verify each
condition with an independent network measurement before inference. Save the
commands and measured bandwidth/RTT with the run. Avoid shaping SSH/control
traffic accidentally; maintain an out-of-band recovery route where possible.

Run a smaller pilot matrix first: unshaped plus {10, 100, 1000 Mbit/s} x
{5, 50 ms}. Inspect results before spending time on the full factorial matrix.

Exit gate: raw data is complete, conditions are verified, rerunning a sample of
cells gives compatible confidence intervals, and crossover regions (if any)
are supported by observations rather than a single run.

### M7 — Analysis, documentation, and reproducibility (5-6 days)

Tasks:

- produce latency decomposition, communication-volume, throughput, memory,
  feasibility, output-quality, and crossover plots;
- compare observed transfer time with `payload_bits / bandwidth + RTT effects`;
- explain anomalies, failures, and threats to validity;
- write setup, two-host deployment, benchmark, and result-reproduction guides;
- tag the baseline release and archive configs, manifests, raw rows, summary
  tables, dependency lock, and commit identifier.

Exit gate: a new user with the two machines can reproduce a smoke run from the
README, and every thesis table/figure can be regenerated from raw results by a
checked-in command.

## 9. Test plan

### Unit tests

- config validation and resolved-default serialization;
- tensor serialization for supported dtypes/shapes, including non-contiguous
  input handling;
- framing under fragmented socket reads and writes;
- size limits, checksums, timeouts, and error messages;
- metric calculations and result schema;
- deterministic dataset ordering and manifest hashing.

### Integration tests

- monolithic versus decomposed adapter on one device;
- cloud client/server on localhost;
- distributed client/server on localhost;
- server subprocess startup/readiness/shutdown;
- version/fingerprint mismatch;
- timeout, disconnect, and server-side OOM behavior;
- repeated requests and memory-growth threshold.

### Hardware/research tests

- model feasibility on edge and cloud;
- equivalence across physical hosts;
- idle and sustained thermal behavior;
- unshaped LAN repeatability;
- each network matrix condition independently verified;
- output-quality metric invariant across policies within declared tolerance.

CPU-only CI should run unit and protocol tests with a fake model adapter.
GPU/model tests should be explicitly marked and run on the target machines;
they should not make ordinary CI download multi-gigabyte weights.

## 10. Experimental outputs

At minimum produce:

1. hardware/software table;
2. feasibility and peak-memory table by policy;
3. projected-feature versus encoded-input byte table;
4. median/p95 end-to-end latency by policy and network condition;
5. stacked latency decomposition with an explicit unaccounted residual;
6. tokens/s and requests/s under the defined sequential workload;
7. output exact-match and task-quality table;
8. latency-versus-bandwidth plots at fixed RTTs;
9. latency-versus-RTT plots at fixed bandwidths;
10. empirical optimal-policy/crossover map with confidence or uncertainty.

Do not mix throughput and single-request latency workloads. If a concurrency
experiment is added later, give it a separate configuration and analysis.

## 11. Risks and mitigations

| Risk | Early signal | Mitigation |
| --- | --- | --- |
| Model API hides the split boundary | Cannot resume generation from visual tensor | Enforce M1 gate; switch to a checkpoint with explicit internals |
| Remote model code changes | Results change after reinstall | Pin exact revision, hash artifacts, lock dependencies |
| All-edge OOM | Load/inference allocation failure | Record infeasibility; do not quantize within this baseline |
| Edge cannot load only its components | Full state allocation before pruning | Build/load component-specific state dictionaries or choose a cleaner model |
| Cross-device token mismatch | First differing greedy token | First prove same-device equality; log logits/tensors near split; report real-hardware divergence |
| Invalid GPU timings | Components do not sum or times are implausibly small | Synchronize CUDA and validate against E2E wall time |
| False one-way network claims | Unsynchronized timestamps | Use client E2E plus server durations, or synchronize clocks and report uncertainty |
| OS socket buffering masks shaping | Measured throughput differs from target | Verify with an independent tool and sufficiently large transfers |
| Thermal throttling/drift | Latency trends with run order | Randomize/interleave conditions and log temperature/clocks |
| Dataset results are not reproducible | Changed downloads/order | Pin dataset revision and commit a hashed request manifest |
| Experiment matrix becomes too large | Campaign cannot finish | Run pilot matrix, then expand only after validation |

## 12. Requirement traceability and final checklist

Maintain `docs/requirements_traceability.md` mapping R1-R18 from the original
specification to code, tests, configs, and result artifacts. Release only when:

- [ ] the model and dataset revisions and all dependencies are pinned;
- [ ] all modes share one adapter and one preprocessing/generation contract;
- [ ] all-edge is measured or explicitly documented as infeasible;
- [ ] all-cloud sends the raw request and runs the full VLM remotely;
- [ ] distributed sends the exact projected visual representation and runs only
      the LLM in the cloud;
- [ ] same-host split and network serialization correctness tests pass;
- [ ] actual-hardware equivalence/quality results are recorded;
- [ ] timing, bytes, tokens, memory, failures, and environment are structured;
- [ ] warm-ups and raw per-request measurements are preserved;
- [ ] controlled network conditions are independently verified;
- [ ] every plot/table can be regenerated automatically;
- [ ] setup, deployment, benchmarking, and limitations are documented;
- [ ] the baseline is tagged and its artifacts are archived.

## 13. Immediate next actions

Execute these in order:

1. Fill in an edge/cloud hardware worksheet and confirm GPU access on both
   machines.
2. Implement M0 configuration/result schemas and repository skeleton.
3. Run the two-day Moondream2 split-feasibility spike and write the checkpoint
   decision record.
4. Freeze the small correctness manifest and deterministic generation contract.
5. Implement the local adapter and all-edge path before any socket code.

The first go/no-go review is the end of M1. Until token-equivalent local split
inference works, networking and large-scale benchmarking should not begin.

## 14. Manageable implementation chunks

The baseline should be implemented through the following self-contained work
chunks. They are intentionally sequential at the important correctness
boundaries, although documentation and CPU-only tests can be improved between
chunks. Each chunk should end with a small commit and a short handoff note
containing commands run, test results, unresolved risks, and the next starting
point.

### Chunk 1 — Repository foundation and experiment contracts

**Scope**

- create the Python 3.11 package, dependency-management files, and source/test
  directory skeleton;
- define validated model, hardware, and experiment configuration schemas;
- define request, result, run-manifest, and per-request metric schemas;
- add deterministic seed handling, structured logging, and a fake model adapter;
- add CPU-only tests and basic CI without downloading real model weights;
- record the RTX 3050 host details and distinguish Windows and WSL environments.

**Deliverables**

- installable package and working CLI help;
- example resolved configurations for all three policies;
- result/run-manifest schema with a sample artifact;
- environment collection script;
- unit-test and lint/type-check commands documented in the README.

**Completion gate**

A clean checkout can create the Python environment and run all CPU-only tests.
One command validates and prints a fully resolved experiment configuration.

**Depends on:** nothing.

### Chunk 2 — GPU access and monolithic model feasibility

**Scope**

- diagnose Windows-to-WSL GPU visibility without changing the host driver
  unless separately authorized;
- verify `nvidia-smi` and `torch.cuda.is_available()` inside the actual runtime;
- record the PyTorch-visible GPU name, VRAM, driver/runtime versions, and BF16
  support;
- inspect, pin, and hash a Moondream2 revision and its remote model code;
- run deterministic monolithic inference on several correctness fixtures;
- measure model-load time, steady-state latency, output tokens, peak VRAM, and
  process RAM;
- attempt the unmodified all-edge precision selected by the experiment
  contract, recording OOM as a result rather than changing precision ad hoc.

**Deliverables**

- GPU diagnostic report;
- pinned model configuration and architecture decision record;
- monolithic smoke-inference command and saved structured results;
- explicit all-edge feasibility or infeasibility result for the 4 GiB GPU.

**Completion gate**

The model produces repeatable greedy output with a pinned artifact, or the
attempt yields enough evidence to reject it and evaluate the documented
fallback model. Do not proceed with an unpinned or changing checkpoint.

**Depends on:** Chunk 1.

### Chunk 3 — Model decomposition and local all-edge policy

**Scope**

- locate and document the exact vision-encoder, projector, and LLM boundary;
- implement the real `VLMAdapter` behind the interfaces established in Chunk 1;
- capture projected visual tokens and resume generation from them;
- compare monolithic and decomposed tensors, token IDs, and text;
- implement synchronized component timers and CPU/GPU memory measurements;
- implement the `edge` execution policy using the shared adapter;
- test repeated inference and memory stability.

**Deliverables**

- model adapter and all-edge policy;
- model-boundary diagram including tensor shape, dtype, visual-token count, and
  byte size;
- monolithic-versus-decomposed equivalence tests;
- per-component all-edge measurement records.

**Completion gate**

On the same device and dtype, decomposed greedy inference produces the same
token IDs as monolithic inference. If the selected model cannot expose this
boundary cleanly, stop and change model before implementing networking.

**Depends on:** Chunk 2.

### Chunk 4 — Reliable transport and all-cloud policy

**Scope**

- implement the versioned length-prefixed TCP framing protocol;
- implement safe image, prompt, tensor, response, metric, and error messages;
- test fragmented reads/writes, checksums, size bounds, timeouts, disconnects,
  and model/protocol fingerprint rejection;
- implement server health/readiness and graceful shutdown;
- implement the all-cloud client/server path on localhost first;
- deploy it across the two real hosts once the cloud machine is available;
- capture client-observed E2E time, server component times, bytes, and memory.

**Deliverables**

- reusable transport package and fake-adapter protocol tests;
- cloud server and cloud execution policy;
- localhost and physical-host smoke-test instructions;
- 100-request stability result.

**Completion gate**

The localhost cloud path returns the same token IDs as the local monolithic
path, rejects incompatible requests safely, and completes 100 sequential
requests without deadlock or unbounded memory growth.

**Depends on:** Chunk 3 for the real adapter; protocol unit tests may start
after Chunk 1.

### Chunk 5 — Fixed distributed policy

**Scope**

- load the vision encoder/projector on edge and the required language-model
  components on cloud without executing the wrong stage on either host;
- implement bit-preserving projected-tensor serialization and validation;
- implement synchronous distributed inference first;
- verify local tensor round-trip and same-host token equivalence;
- verify real edge/cloud behavior and document cross-device differences;
- implement the prompt-first overlapped pipeline only after the synchronous
  path passes, including pending-request timeout and cleanup;
- compare synchronous and overlapped outputs and latency.

**Deliverables**

- distributed execution policy with synchronous and overlap settings;
- tensor serialization and integration test suite;
- evidence of component placement and memory use on each host;
- distributed correctness and 100-request stability report.

**Completion gate**

The transmitted tensor is byte-identical after reconstruction, the server does
not rerun vision, the edge does not run the LLM, and both distributed variants
meet the declared output-equivalence policy. One variant is then named as the
primary distributed baseline.

**Depends on:** Chunks 3 and 4.

### Chunk 6 — Unified benchmark runner and correctness campaign

**Scope**

- freeze the correctness and benchmark request manifests;
- implement one runner that selects `edge`, `cloud`, or `distributed` without
  changing model, data, generation, or measurement code;
- implement warm-up exclusion, repetitions, randomized run order, failure
  recording, and resumable result storage;
- collect run manifests and raw per-request JSONL/Parquet rows automatically;
- execute correctness gates G1-G4 and the selected VQA/task-quality metric;
- link R1-R18 to tests and artifacts in a traceability document.

**Deliverables**

- single policy-selecting benchmark command;
- frozen, hashed request manifests;
- cross-policy correctness report;
- requirements traceability matrix;
- validated raw-result and summary generation pipeline.

**Completion gate**

All feasible policies run from the same resolved configuration, results are
reproducible, and every output mismatch is either fixed or explicitly analyzed.
The benchmark may not silently drop failed or OOM requests.

**Depends on:** Chunk 5.

### Chunk 7 — Controlled network experiments and thesis artifacts

**Scope**

- implement reversible, interface-specific network shaping and restoration;
- independently verify each shaped bandwidth and RTT condition;
- execute the pilot matrix and inspect timing/accounting quality;
- execute the full approved matrix with warm-ups and at least 30 measured
  requests per cell;
- calculate robust summaries and confidence intervals from raw observations;
- produce latency decomposition, bandwidth/RTT, communication, memory,
  throughput, quality, and optimal-policy/crossover figures;
- document setup, deployment, reproduction, threats to validity, and failures;
- archive and tag the finished baseline artifacts.

**Deliverables**

- verified network-condition logs and complete raw benchmark dataset;
- reproducible analysis command and thesis-ready tables/figures;
- baseline reproduction guide and limitations section;
- tagged baseline release.

**Completion gate**

Every reported number and figure can be regenerated from archived raw rows and
resolved configs. Repeated sample cells have compatible uncertainty intervals,
and any crossover claim is supported by multiple observations.

**Depends on:** Chunk 6 and stable access to both physical hosts.

### Chunk execution rule

Work on only one primary chunk at a time. At the start of a chunk, inspect the
previous handoff and confirm its completion gate still passes. At the end,
update the traceability matrix and record:

```text
Chunk:
Commit/worktree state:
Commands run:
Tests passed/failed:
Artifacts produced:
Decisions made:
Known limitations:
Next chunk entry point:
```

If a completion gate fails, keep the work in that chunk until the cause is
resolved or a documented architecture decision changes the plan. In
particular, do not compensate for a failed model split by hiding model-specific
logic in the transport or execution-policy layers.
