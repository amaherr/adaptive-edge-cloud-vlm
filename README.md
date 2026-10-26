# Adaptive Edge–Cloud VLM Inference

Research prototype for distributing inference from a single pretrained vision–language model (VLM) between an edge device and a cloud machine. The project aims to select an internal execution configuration per image-and-text request using network conditions, available compute and memory, and request characteristics. The intended outcome is lower end-to-end latency and communication cost without changing the model's answer quality or exceeding edge resources.

The **automatic** adaptive selector is not implemented yet. The current, reproducible foundation has three fixed policies: all-edge, all-cloud, and edge vision/projector with cloud language generation. Phase 1 adds a manual configuration dispatcher and a server that can serve both remote paths without a restart. Keeping the same pinned model, preprocessing, prompt, precision, and greedy decoding across policies makes later adaptive decisions measurable rather than conflating them with model changes.

## Contents

- [Project architecture and scope](#project-architecture-and-scope)
- [Current implementation and methodology](#current-implementation-and-methodology)
- [Setup](#setup)
- [Usage](#usage)
- [CLI reference](#cli-reference)
- [Using a separate cloud PC](#using-a-separate-cloud-pc)
- [Results and limitations](#results-and-limitations)
- [Next research steps](#next-research-steps)
- [Technology and repository layout](#technology-and-repository-layout)

## Project architecture and scope

The envisioned runtime accepts an image and a prompt, observes the current edge, network, and cloud state, then selects a feasible placement of the **same** VLM. A future controller will compare predicted end-to-end latency and transfer cost while enforcing edge memory and task-quality constraints. Benchmark observations will inform its decisions. The fixed policies, manual dispatch, and dual-path server exist today; automatic selection, live resource/network sensing, and prediction remain future work.

~~~mermaid
flowchart LR
    R["Image + prompt"] --> S["Adaptive selector<br/>(planned)"]
    T["Network, edge/cloud load,<br/>request features (planned)"] -.-> S
    S --> E["All edge<br/>(implemented)"]
    S --> C["All cloud<br/>(implemented)"]
    S --> D["Edge vision + cloud LLM<br/>(implemented)"]
    E --> O["Answer + timings + resource/byte records"]
    C --> O
    D --> O
    O -. "future feedback" .-> S
~~~

The initial research question is whether dynamic selection among these internal configurations improves latency and communication efficiency under changing conditions. This repository does **not** yet contain a dynamic scheduler, compression, quantization, multi-node orchestration, network shaping, or a completed physical edge–cloud experiment.

## Current implementation and methodology

### Three execution policies

| Policy | Edge process | Cloud process | Request crossing the link |
| --- | --- | --- | --- |
| `edge` | Image preprocessing, vision encoder, projector, and LLM | None | None |
| `cloud` | Reads the image and sends the request | Complete VLM, including vision and LLM | Original encoded image bytes and prompt |
| `distributed` | Image preprocessing, vision encoder, and projector | LLM image/prompt prefill and greedy generation | Prompt and unmodified projected visual tensor |

All policies use the pinned [Moondream2 model configuration](configs/model/moondream2.yaml): revision `9a7d4024050840e001defacec2b00727e89149e6`, BF16, Pillow preprocessing, batch size one, no sampling, and at most 32 new tokens. The [artifact manifest](configs/model/moondream2_artifact.yaml) pins the weights, tokenizer, and reviewed remote-code hashes. Model loading verifies the artifact before inference.

For the split path, the adapter captures the tensor returned after the vision encoder and projector, **before** LLM image prefill. The current tensor is BF16 `[729, 2048]`: 729 visual tokens and 2,985,984 raw bytes. The public model image-encoding API is not the boundary because it proceeds into LLM prefill. The distributed server does not recompute vision, and the edge vision process does not generate language tokens.

~~~mermaid
flowchart LR
    I["Image + prompt"] --> L["Edge: full Moondream<br/>edge policy"]
    L --> O["Answer"]
    I --> RC["Edge: raw-image client"]
    RC -- "image bytes + prompt<br/>versioned TCP" --> FM["Cloud role: full Moondream"]
    FM --> O
    I --> VP["Edge: image preprocessing<br/>vision encoder + projector"]
    VP -- "projected BF16 tensor + prompt<br/>versioned TCP" --> TM["Cloud role: text-only Moondream"]
    TM --> O
~~~

`distributed` defaults to the verified `synchronous` pipeline. The optional `prompt_first` variant sends text ahead of vision encoding so the server can prepare it while the edge works; it remains the **same policy**, not a fourth placement. Both variants were checked for 32-token equivalence on a fixture. The loader currently constructs the full checkpoint before pruning unneeded modules, so **load-time RAM is not partitioned**, even though steady-state components are.

### Transport and the localhost cloud

The cloud role is currently mimicked by a **separate server process on the same PC**, bound to `127.0.0.1:5050`. This validates component separation, actual TCP framing, health checks, and exact application bytes; it does not model an independent cloud CPU/GPU or a WAN/LAN link. Only one cloud-role server should use port 5050 at a time: stop `cloud-server` before starting `distributed-server`.

The protocol uses versioned, length-prefixed frames with bounded JSON metadata, a binary payload and SHA-256 checksum. The client checks the model contract fingerprint and protocol version. Bad frames, mismatched contracts, timeouts, and server errors are rejected or recorded. Cloud requests carry the original image bytes; distributed requests carry the contiguous native-dtype tensor without baseline compression or quantization. The server processes requests sequentially for the batch-size-one latency experiment. There is **no TLS, authentication, or authorization**; see [separate-host deployment](#using-a-separate-cloud-pc).

### Correctness, benchmarking, and observability

- A [12-request correctness manifest](data/manifests/correctness.jsonl) and a [100-request fixture manifest](data/manifests/benchmark_fixture.jsonl) are checked in. The loader verifies manifest revision, unique IDs, and each image's SHA-256 before running.
- The unified `benchmark` command selects any of the three policies against one resolved experiment configuration. It seeds the runtime, runs warm-ups first, shuffles each measured sweep reproducibly, and records one attempt per request/repetition. The model is loaded once per run; cold load time is separate from steady-state request latency.
- Each run writes `plan.json`, `run_manifest.json`, fsynced `requests.jsonl`, and `summary.json` to its unique run directory. The run manifest includes resolved config, dataset/dependency hashes, Git state, host/Python/GPU inventory, and seed. Raw rows include config/model/dataset/image hashes, output tokens, timings, bytes, status, and any error.
- The runner preserves failed and out-of-memory attempts in the denominator, but excludes their latency from successful-latency summaries. Summaries provide median, p95, IQR, mean, failure rate, wire bytes where relevant, and scored-task counts. `--resume` accepts only the original plan and skips already recorded trials.
- `compare-runs` verifies provenance and compares **every measured request/repetition pair** across policies. It reports exact-token and decoded-text mismatches, first divergent token, failures, and per-policy task scores.
- Per-request metrics include client-observed end-to-end time, available vision/projector/prefill/generation stage times, process RSS/CUDA memory, and exact application-layer wire bytes. Progress/readiness messages and structured records are available; **continuous telemetry and adaptive monitoring are not yet implemented**. Application bytes and localhost timings are not physical link throughput or one-way delay.

The synthetic fixture quality score is case-insensitive exact match after trimming whitespace. It is only a correctness diagnostic. The VQA importer accepts local VQA v2 validation files and preserves ten annotator answers, but the runner's VQA consensus normalization is provisional: use the [official VQA evaluation procedure](https://visualqa.org/evaluation.html) before reporting public-dataset accuracy.

## Setup

Run commands from the repository root. The model environment and lock are targeted at Linux/WSL with Python 3.11 and CUDA 12.8 PyTorch; the dependency-light development environment also works with Python 3.12. The full model weights are **not** committed and require a first download and several gigabytes of disk space. The current 4 GiB RTX 3050 cannot reliably load the complete BF16 model; use CPU for exact all-policy reference runs.

### 1. Development environment

~~~bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/adaptive-vlm config validate configs/experiments/chunk6_local_campaign.yaml
.venv/bin/pytest -q
~~~

Python 3.12 may be used for this CPU-only development/test environment. `.venv/bin/adaptive-vlm` is the installed entry point; `.venv/bin/python -m adaptive_vlm.cli` is equivalent.

### 2. Pinned model environment

~~~bash
python3.11 -m venv .venv-model
.venv-model/bin/python -m pip install --upgrade pip
.venv-model/bin/python -m pip install -r requirements/model-cu128.lock
~~~

The lock includes an editable install of this package and pinned PyTorch, Transformers, Hugging Face Hub, Safetensors, Pillow, NumPy, and psutil versions. Use a Linux/WSL machine for the CUDA lock. Check `nvidia-smi` and PyTorch CUDA visibility separately if using a GPU; CPU execution does not require CUDA visibility.

For the **first** model run, omit `--offline` so the exact pinned weights, tokenizer, and remote code can populate `.model-cache`. This downloads a multi-gigabyte checkpoint and verifies it against [the artifact manifest](configs/model/moondream2_artifact.yaml):

~~~bash
.venv-model/bin/adaptive-vlm edge-infer \
  --config configs/experiments/baseline_edge.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --vision-device cpu --text-device cpu \
  --image tests/fixtures/moondream_smoke.ppm \
  --prompt "Describe this image." --output-dir artifacts/first-model-run
~~~

After the cache is populated, add `--offline` to require local artifacts only. A model server or split client on a different PC needs its own verified local cache. The baseline templates are valid for one-off inference, but their pending dataset/cloud fields are **not** ready for a physical benchmark.

### 3. Configuration files

Every experiment YAML references a model file and edge/cloud hardware inventories and supplies a dataset manifest/revision, seed, warm-up count, repetition count, transport address/timeout/version, and artifact directory. Paths are resolved relative to the experiment file. Validate a new configuration with `adaptive-vlm config validate PATH`; unknown fields, moving model revisions, sampling, and non-unit batch sizes are rejected.

| Experiment file | Use today |
| --- | --- |
| [`chunk1_smoke.yaml`](configs/experiments/chunk1_smoke.yaml) | Dependency-free `fake-infer` sanity check. |
| [`chunk4_fake_cloud.yaml`](configs/experiments/chunk4_fake_cloud.yaml) | Fake localhost server/client and transport tests. |
| [`chunk4_local_cloud.yaml`](configs/experiments/chunk4_local_cloud.yaml) | Real localhost full-model cloud server and one-off client. |
| [`chunk5_local_distributed.yaml`](configs/experiments/chunk5_local_distributed.yaml) | Real localhost text-only server and vision client. |
| [`chunk6_fake_benchmark.yaml`](configs/experiments/chunk6_fake_benchmark.yaml) | Fast benchmark-runner checks without a VLM; edge or fake cloud. |
| [`chunk6_local_campaign.yaml`](configs/experiments/chunk6_local_campaign.yaml) | Shared real-model correctness/benchmark config for `--mode edge`, `cloud`, or `distributed`. |
| [`baseline_edge.yaml`](configs/experiments/baseline_edge.yaml), [`baseline_cloud.yaml`](configs/experiments/baseline_cloud.yaml), [`baseline_distributed.yaml`](configs/experiments/baseline_distributed.yaml) | Physical-host templates with pending dataset/cloud details; copy and fill before physical experiments. |

The one-off server commands require an experiment file whose YAML `mode` matches the server role. In contrast, `benchmark --mode` selects the policy independently of the YAML `mode`, allowing all three runs to share the exact same resolved model/data/generation configuration. Remote server and benchmark files must still agree on model contract, address, port, and protocol. `configs/hardware/local_model_cloud.yaml` describes the **same host** acting as cloud, not a second PC.

## Usage

All commands below assume setup is complete and are run from the repository root. A one-off inference saves a run manifest and request record under `--output-dir/RUN_ID/` when that option is supplied. Final JSON is printed to stdout; progress is generally printed to stderr, while `benchmark` also prints trial progress to stdout. For automation, read its saved `summary.json` instead of assuming stdout contains JSON alone.

### Fake smoke tests

~~~bash
.venv/bin/adaptive-vlm fake-infer \
  --config configs/experiments/chunk1_smoke.yaml \
  --image tests/fixtures/smoke_image.txt \
  --prompt "Describe this fixture." --output-dir artifacts/fake-smoke

# In a separate terminal, start this server before the client:
.venv/bin/adaptive-vlm cloud-server --config configs/experiments/chunk4_fake_cloud.yaml
.venv/bin/adaptive-vlm cloud-infer \
  --config configs/experiments/chunk4_fake_cloud.yaml \
  --image tests/fixtures/smoke_image.txt --prompt "Describe this fixture."
~~~

Stop a server with Ctrl+C. The fake adapter hashes input bytes and text; it tests infrastructure, **not** VLM task quality.

### Give the real model a photo and prompt

~~~bash
.venv-model/bin/adaptive-vlm edge-infer \
  --config configs/experiments/baseline_edge.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --vision-device cpu --text-device cpu \
  --image /path/to/photo.jpg --prompt "What is happening in this photo?" \
  --output-dir artifacts/manual-edge
~~~

Use a real JPEG/PNG or the repository PPM fixture. The CPU/CPU placement is the exact-equivalence reference. `--vision-device cuda:0 --text-device cpu` is an experimental mixed-device placement: it has faster vision inference on this machine, but its complete greedy token sequence differed from the CPU reference after a 19-token common prefix. Do not silently use it for same-device correctness claims.

### All-cloud on localhost

Terminal A:

~~~bash
.venv-model/bin/adaptive-vlm cloud-server \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
~~~

After the readiness JSON appears, terminal B:

~~~bash
.venv-model/bin/adaptive-vlm cloud-infer \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --image /path/to/photo.jpg --prompt "What is happening in this photo?" \
  --output-dir artifacts/manual-cloud
~~~

The edge client sends raw image bytes and text, while the server runs the complete model. The client does not need model weights in this mode.

### Split vision and language on localhost

Stop `cloud-server` first, then start the text-only server in terminal A:

~~~bash
.venv-model/bin/adaptive-vlm distributed-server \
  --config configs/experiments/chunk5_local_distributed.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
~~~

Terminal B:

~~~bash
.venv-model/bin/adaptive-vlm distributed-infer \
  --config configs/experiments/chunk5_local_distributed.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --vision-device cpu \
  --pipeline synchronous --image /path/to/photo.jpg \
  --prompt "What is happening in this photo?" \
  --output-dir artifacts/manual-distributed
~~~

Change `--pipeline` to `prompt_first` only when testing that explicit overlap variant. Each side verifies the same model/generation fingerprint.

### Reproducible benchmark and comparison

Start with the [real shared configuration](configs/experiments/chunk6_local_campaign.yaml). The following is a **one-sweep correctness check**; the config defaults to 10 warm-ups and 30 full measured sweeps if overrides are omitted. A sweep processes every manifest request once in seeded random order.

~~~bash
# No server needed:
.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --mode edge --run-dir artifacts/example-edge \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --vision-device cpu --text-device cpu --warmups 1 --repetitions 1

# Start cloud-server with chunk4_local_cloud.yaml in another terminal:
.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --mode cloud --run-dir artifacts/example-cloud \
  --warmups 1 --repetitions 1

# Stop cloud-server; start distributed-server with chunk5_local_distributed.yaml:
.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --mode distributed --run-dir artifacts/example-distributed \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --vision-device cpu \
  --pipeline synchronous --warmups 1 --repetitions 1

.venv/bin/adaptive-vlm compare-runs \
  artifacts/example-edge artifacts/example-cloud artifacts/example-distributed \
  --output artifacts/example-comparison.json
~~~

Choose **fresh** run directories; an existing directory is refused unless `--resume` is given with the same arguments, config, manifest, and plan. The benchmark records individual failures but may still exit successfully after completing the schedule; inspect `status_counts`/`failure_rate`. Likewise, inspect `compare-runs`'s `complete_and_equal` field rather than relying only on its exit code.

For a fast 100-request storage/scheduling run without Moondream, use `chunk6_fake_benchmark.yaml`, `--mode edge`, `--manifest data/manifests/benchmark_fixture.jsonl`, `--warmups 10`, and `--repetitions 1`. That fixture has no task-quality references.

### Phase 2 profiling

The fixed-policy benchmark now records trial start time, exact model prompt-token count, prompt prefill/decode durations, transport subintervals, response bytes, and memory readings. After three completed runs with the same frozen schedule, build a validated profile:

~~~bash
.venv/bin/adaptive-vlm profile-runs \
  artifacts/example-edge artifacts/example-cloud artifacts/example-distributed \
  --specification configs/profiles/phase2_local_fixture.json \
  --output-dir artifacts/example-profile
~~~

Inspect `artifacts/example-profile/summary.json` for coverage/failures and `profile.jsonl` for each warmup, measured success, and failure. The [schema note](docs/phase2_profile_schema.md) defines timing boundaries, null reasons, residuals, and units. The [split inspection](docs/phase2_split_feasibility.md) documents candidate graph boundaries that remain inactive. The supplied specification labels the local loopback network as uncontrolled; it cannot support physical-network crossover claims.

### Phase 3 predictor calibration

Fit the component predictor and evaluate the frozen request holdout from the Phase 2 local profile:

~~~bash
.venv/bin/adaptive-vlm calibrate-predictor \
  --profile artifacts/phase2/2026-10-26/local-1/profile/profile.jsonl \
  --split configs/predictor/phase3_local_holdout.json \
  --output-dir artifacts/example-predictor
~~~

Read `evaluation.json` for per-candidate error, interval coverage, ranking against the constant-median baseline, and gate flags. `calibration.json` contains the fitted components and coverage bounds; `predictions.jsonl` keeps each held-out task snapshot and prediction. The [Phase 3 handoff](docs/phase3_handoff.md) reports the local gate miss and the [predictor contract](docs/phase3_predictor_schema.md) defines units and out-of-coverage behavior. This calibration is diagnostic and is not connected to automatic execution.

### Phase 4 offline selector and oracle

Evaluate the pure selector on the frozen local Phase 3 predictions and matched observed outcomes:

~~~bash
.venv/bin/adaptive-vlm evaluate-selector \
  --predictions artifacts/phase3/2026-10-26/clean-1/predictions.jsonl \
  --specification configs/oracle/phase4_local_holdout.json \
  --output-dir artifacts/example-selector
~~~

Read `evaluation.json` for regret, top-choice agreement, uncertainty, exclusions, and gate flags. `decisions.jsonl` preserves each candidate prediction and reason beside the retrospective measured oracle. The [Phase 4 handoff](docs/phase4_handoff.md) gives the local result and second-machine steps; the [selector contract](docs/phase4_selector_contract.md) defines eligibility and pairing. The local G4 performance gate remains open and automatic selection is inactive.

### Phase 5 adaptive diagnostic run

With the pinned Moondream cache, start `adaptive-server` as in the [Phase 5 handoff](docs/phase5_handoff.md), then run the frozen local mixed-choice campaign:

~~~bash
.venv-model/bin/adaptive-vlm adaptive-benchmark \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --manifest data/manifests/correctness.jsonl \
  --calibration artifacts/phase3/2026-10-26/clean-1/calibration.json \
  --replay configs/adaptive/phase5_local_replay.json \
  --run-dir artifacts/example-adaptive \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --warmups 1 --repetitions 1 --diagnostic-allow-unpassed-gates
~~~

The diagnostic flag explicitly permits the current failed G3/open G4 fit for an integration check; it does not approve performance selection. `requests.jsonl` contains every candidate prediction, decision, actual result, overhead and any retry. Use `analyze-adaptive` with matching edge/cloud/distributed fixed controls, as described in the [execution contract](docs/phase5_execution_contract.md). The command requires a resident full edge adapter for exact prompt-token counting.

### Phase 6 network trace and live monitor

Replay the [synthetic controlled trace](configs/network_traces/phase6_synthetic_trace.json) through the predictor and selector without loading a model:

~~~bash
.venv/bin/adaptive-vlm replay-network-trace \
  --calibration configs/network_traces/phase6_synthetic_calibration.json \
  --trace configs/network_traces/phase6_synthetic_trace.json \
  --output-dir artifacts/example-network-trace
~~~

For a real diagnostic, add `--monitor-mode live --probe-timeout-ms 2000 --probe-max-age-ms 5000` to `adaptive-benchmark`, then run `analyze-monitor RUN_DIR --output RUN_DIR/monitor_analysis.json`. The [Phase 6 handoff](docs/phase6_handoff.md) gives the complete local commands and second-machine setup; the [monitor contract](docs/phase6_monitor_contract.md) defines probe age, failure behavior and passive-rate limits. Health-probe round trip and socket-send rate are recorded as distinct application observations and are not fed to the current loopback predictor.

### Import a VQA validation subset

The repository does not download or redistribute the VQA questions, answers, or COCO images. Once the official VQA v2 val2014 JSON files and images are available locally, freeze a deterministic subset:

~~~bash
.venv/bin/adaptive-vlm import-vqa-v2 \
  --questions data/raw/vqa/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations data/raw/vqa/v2_mscoco_val2014_annotations.json \
  --images-dir data/raw/coco/val2014 \
  --output data/manifests/vqa_val2014_seed42.jsonl \
  --count 100 --seed 42 --revision vqa-v2-val2014-local
~~~

Set a copied experiment YAML's `dataset.manifest_path` and `dataset.revision` to the new file and exact revision. The importer samples paired question/annotation IDs with a seed, stores ten human answers and image hashes, and refuses to overwrite an existing manifest. See the [official VQA data format](https://visualqa.org/download.html). Before thesis-quality claims, run the official evaluator: the in-run VQA consensus score lacks full official answer normalization.

### Validation and helper scripts

~~~bash
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest --cov=adaptive_vlm --cov-report=term-missing
.venv/bin/adaptive-vlm environment --output artifacts/environment.json
~~~

CPU-only CI runs Ruff, strict mypy, and pytest with an 80% coverage threshold without downloading model weights. The real TCP subprocess test uses an ephemeral port and skips only when a restricted sandbox forbids loopback binding. `scripts/verify_local_decomposition.py`, `verify_cloud_equivalence.py`, and `verify_distributed_equivalence.py` reproduce focused model correctness checks; `verify_cloud_stability.py` and `verify_distributed_stability.py` exercise repeated requests. `scripts/probe_moondream.py` probes feasibility, and `inspect_model_artifact.py` inspects local safetensors. Use `--help` on a script for its experiment-specific options.

## CLI reference

Run `adaptive-vlm --help` or `adaptive-vlm COMMAND --help` for machine-readable, version-matched usage. This table covers the current public commands; bracketed flags are optional.

| Command | Required arguments | Optional arguments and defaults |
| --- | --- | --- |
| `config validate` | `PATH` | `--output PATH` writes resolved JSON; it is always printed. |
| `environment` | None | `--workdir PATH` (current directory), `--output PATH`. |
| `fake-infer` | `--config PATH --image PATH --prompt TEXT` | `--request-id` (`smoke-0001`), `--output-dir PATH`, `-v/--verbose`. |
| `edge-infer` | `--config PATH --artifact-manifest PATH --image PATH --prompt TEXT` | `--cache-dir` (`.model-cache`), `--request-id` (`edge-0001`), `--vision-device` (`cpu`), `--text-device` (`cpu`), `--offline`, `--output-dir PATH`, `-v/--verbose`. |
| `cloud-server` | `--config PATH` | `--artifact-manifest PATH` (required for Moondream, not fake), `--cache-dir` (`.model-cache`), `--device` (`cpu`), `--offline`, `--bind-host HOST` (otherwise config host), `-v/--verbose`. |
| `adaptive-server` | `--config PATH --artifact-manifest PATH` | Loads one full Moondream model for raw-image and projected-tensor requests. `--cache-dir` (`.model-cache`), `--device` (`cpu`), `--offline`, `--bind-host HOST`, `-v/--verbose`. |
| `cloud-infer` | `--config PATH --image PATH --prompt TEXT` | `--request-id` (`cloud-0001`), `--output-dir PATH`, `-v/--verbose`. |
| `adaptive-infer` | `--config PATH --configuration-id ID --image PATH --prompt TEXT` | Manual Phase 1 selection among `edge/full`, `cloud/raw_image`, and `distributed/projected/synchronous`; see the [Phase 1 handoff](docs/phase1_handoff.md) for edge component, memory, network-state and artifact options. |
| `distributed-server` | `--config PATH --artifact-manifest PATH` | `--cache-dir` (`.model-cache`), `--device` (`cpu`), `--offline`, `--bind-host HOST`, `-v/--verbose`. |
| `distributed-infer` | `--config PATH --artifact-manifest PATH --image PATH --prompt TEXT` | `--cache-dir` (`.model-cache`), `--request-id` (`distributed-0001`), `--vision-device` (`cpu`), `--pipeline` (`synchronous` or `prompt_first`), `--offline`, `--output-dir PATH`, `-v/--verbose`. |
| `benchmark` | `--config PATH --mode edge|cloud|distributed --run-dir PATH` | `--manifest PATH` (config dataset path), `--artifact-manifest PATH` (required for real edge/distributed), `--cache-dir` (`.model-cache`), `--vision-device` (`cpu`), `--text-device` (`cpu`), `--offline`, `--warmups N` and `--repetitions N` (config values), `--pipeline` (`synchronous` or `prompt_first`), `--resume`, `-v/--verbose`. |
| `compare-runs` | At least two `RUN_DIR` paths | `--output PATH` saves comparison JSON; comparison is always printed. |
| `profile-runs` | Completed matching `RUN_DIR` paths | `--specification PATH --output-dir PATH`; validates raw trials, hashes and coverage and writes versioned profile rows. |
| `calibrate-predictor` | `--profile PATH --split PATH --output-dir PATH` | Fits only frozen training rows and writes a calibration, held-out evaluation and per-trial predictions. |
| `evaluate-selector` | `--predictions PATH --specification PATH --output-dir PATH` | Checks a frozen predictions hash and writes offline decisions plus matched measured-oracle metrics. |
| `adaptive-benchmark` | `--config PATH --manifest PATH --calibration PATH --replay PATH --run-dir PATH --artifact-manifest PATH` | Runs a resumable mixed-choice campaign; requires passing G3/G4 reports or explicit `--diagnostic-allow-unpassed-gates`. |
| `analyze-adaptive` | `ADAPTIVE_RUN_DIR EDGE_DIR CLOUD_DIR DISTRIBUTED_DIR` | Matches each choice to a fixed control and audits tokens, placement, wire bytes, and decisions. |
| `replay-network-trace` | `--calibration PATH --trace PATH --output-dir PATH` | Replays frozen bandwidth/RTT/health states through the predictor and selector; reports switches and safety. |
| `analyze-monitor` | `RUN_DIR --output PATH` | Audits live probe timestamps, age, fail-closed decisions, passive updates, and monitor overhead. |
| `import-vqa-v2` | `--questions PATH --annotations PATH --images-dir PATH --output PATH --revision LABEL` | `--count N` (100), `--seed N` (42). |

`--config` references an experiment YAML, not a model YAML. `--artifact-manifest` references [the pinned artifact manifest](configs/model/moondream2_artifact.yaml); it is distinct from the request `--manifest` used by `benchmark`. `--device` places the cloud server's model/text stage; `--vision-device` and `--text-device` name edge-side placements. `--offline` forbids model downloads and is appropriate only after the cache is populated. `--bind-host` controls the listening interface, while `transport.host` in the config is the client-reachable address. `-v` adds detail; core phase milestones and benchmark trials are already visible by default.

## Using a separate cloud PC

The current PC can be the edge and a second Linux/WSL PC can be the cloud once its hardware and network are known. **No separate cloud machine has been validated yet.** These are deployment instructions, not a claim of physical-host results.

1. Check out the same Git commit on both PCs. Install the pinned model environment on the cloud; install it on the edge as well for `edge` and `distributed`. Place and verify the pinned artifact in each host's local cache before using `--offline`. The split loader still needs RAM for a transient full checkpoint on **both** hosts.
2. Record the cloud PC's actual OS, CPU, RAM, GPU/VRAM, driver, and software in a new `configs/hardware/*.yaml` file with `status: verified`. Do not label [`cloud_pending.yaml`](configs/hardware/cloud_pending.yaml) or [`local_model_cloud.yaml`](configs/hardware/local_model_cloud.yaml) as a physical machine.
3. Copy the cloud and distributed experiment files into site-specific YAMLs. Point `cloud_hardware_config` at the verified inventory and set `transport.host` to the cloud address reachable from the edge (same port, timeout, and protocol on both machines). Keep `model_config`, generation settings, request manifest/revision, and seed aligned. The server YAML must use `mode: cloud` for `cloud-server` or `mode: distributed` for `distributed-server`. Keep a single shared benchmark YAML for all three client-side `--mode` selections.
4. Restrict inbound TCP to the edge host on a trusted network, or use a protected VPN/SSH tunnel. The transport has integrity checks but **no encryption or authentication**. Never expose port 5050 directly to the public internet. If the cloud must listen on an interface other than its client-reachable address, pass `--bind-host` appropriately (for example `0.0.0.0` only behind a restrictive firewall).
5. For all-cloud, start `cloud-server` on the cloud with the site cloud YAML, `--artifact-manifest`, and `--device cpu` or a suitable GPU. Wait for readiness and matching model fingerprint, then run `cloud-infer` or `benchmark --mode cloud` on the edge with the site benchmark YAML. Only the cloud needs the full model for this policy.
6. Stop that server. For distributed, start `distributed-server` on the cloud using the site distributed YAML. Then run `distributed-infer` or `benchmark --mode distributed` on the edge with its verified vision model and cache. The server holds the text role; the edge sends the projected tensor.
7. Run `edge` on the edge as a local reference, then `compare-runs` over the three matching manifests. Archive the raw run directories, both host inventories, server readiness metadata, model/dependency hashes, and network topology. Investigate every token mismatch; CPU/GPU floating-point differences are possible. Measure bandwidth/RTT independently before drawing network-performance conclusions.

The actual cloud host, real-link G4 correctness, network-condition campaign, and any cross-device task-quality result remain pending.

## Results and limitations

The [Phase 0 handoff](docs/phase0_handoff.md) records a fresh three-policy localhost CPU correctness campaign, its input hashes, passing gates, reproducible runner, and local artifact paths. All 12 measured requests passed in each policy and the comparison found 12/12 matching token sequences. The separate physical cloud host is still unavailable, so the physical-host gate remains pending. These are correctness observations, not controlled performance estimates.

The [Phase 1 handoff](docs/phase1_handoff.md) records a resident three-policy localhost campaign through the new manual dispatcher and dual-path server. All 36 request/configuration results completed, all 12 request triples matched, and each matched the Phase 0 edge tokens. The report includes the observed client and server memory footprint; it is not a dynamic scheduling or physical-host performance result.

The [Phase 2 handoff](docs/phase2_handoff.md) documents per-stage timing and exact-byte instrumentation, the validated profile command, and a fresh local Moondream campaign. The local profile has 12/12 measured successes per configuration, 12/12 matching token triples, and zero negative accounting residuals. Its declared loopback condition is uncontrolled; the physical-host and controlled-network G2 checks remain pending.

The [Phase 3 handoff](docs/phase3_handoff.md) records the frozen local predictor fit. On two held-out requests, per-configuration median latency APE was 2.87–4.89%, but the predictor's fastest-choice ranking was 0/2 versus 2/2 for the constant-median baseline. The local G3 ranking target failed; a controlled-network holdout and physical-host validation remain unavailable.

The [Phase 4 handoff](docs/phase4_handoff.md) records the pure selector and offline measured oracle. The two local paired requests produced 1.90% median regret but 0/2 observed top-choice agreement. This is a diagnostic replay of the already inspected Phase 3 holdout; full G4 and validated automatic selection remain pending physical controlled conditions and a passing new G3 fit.

The [Phase 5 handoff](docs/phase5_handoff.md) records a real 12-request mixed-choice local Moondream run: 12/12 successful, four selections per configuration, zero same-placement token mismatches or wire/trace errors, and a passing **local G5 integration gate**. The choices were forced by declared replay health exclusions; G3/G4 and physical performance claims remain open.

The [Phase 6 handoff](docs/phase6_handoff.md) records a nine-sample synthetic controlled trace with the expected edge-to-split crossover and zero noisy-segment switches after warmup. A three-request local Moondream run recorded live bounded health probes beside every decision; its monitor audit found no unsafe selections and median probe cost of 1.51 ms. Physical network, free GPU memory, and cloud load remain unmeasured.

On one WSL host, a real pinned Moondream2 CPU/BF16 correctness campaign ran the 12-request synthetic manifest with one excluded warm-up per policy. All three modes completed 12/12 measured requests, and the local `artifacts/chunk6/correctness-comparison.json` audit reported **12/12 identical token sequences and decoded texts**, zero failures, and a fixture exact-match score of 3/9 scored prompts in each policy. G1 projected-feature equivalence and G2 tensor serialization checks also passed in earlier focused tests. A fake-adapter 100-request fixture run completed 100/100 measured attempts after 10 warm-ups.

| Policy | Median observed request time | Mean application wire bytes | Fixture score |
| --- | ---: | ---: | ---: |
| All edge, CPU | 38.5 s | Not applicable | 3/9 |
| All cloud, localhost CPU | 35.1 s | 1,580 bytes | 3/9 |
| Distributed synchronous, localhost CPU | 30.9 s | 2,987,580 bytes | 3/9 |

These are **one-sweep correctness observations**, not controlled latency estimates or evidence that distributed execution is faster. The runs were sequential, not interleaved across policies or subject to verified network conditions. The full model did not reliably load on this machine's 4 GiB GPU, so CPU all-edge is the feasibility reference; CUDA vision plus CPU text is an experimental mixed-device placement. Synthetic fixtures are not a recognized VQA benchmark. Physical-host correctness, official VQA accuracy, continuous resource sensing, adaptive placement, and controlled network experiments remain to be done.

Generated `artifacts/`, `.model-cache/`, and `data/raw/` are ignored by Git. Preserve and archive raw results separately before moving machines; the README and committed manifests alone do not include the experimental JSONL rows. Project source, config, tests, and this README remain version-controlled.

## Next research steps

The next research milestone is to complete controlled Phase 2–3 profiling on a separate cloud host, with a fresh predeclared request and network holdout, then revise and retest the predictor's ranking. Evaluate the Phase 4 selector and Phase 6 monitor on that physical holdout, and rerun the Phase 5 adaptive path with matching fixed controls before enabling validated automatic selection. A local VQA v2 validation subset and the official evaluator are also needed for task-quality claims beyond synthetic fixtures.

## Technology and repository layout

| Technology | Role |
| --- | --- |
| Python 3.11, PyTorch 2.7.1 + CUDA 12.8 | Model execution, BF16 tensors, device placement, and synchronized measurements. |
| Hugging Face Transformers/Hub, Safetensors, Tokenizers | Load and verify the immutable Moondream2 checkpoint, source, and tokenizer. |
| Pillow, NumPy, psutil | Canonical image preparation, tensor support, and process resource metrics. |
| PyYAML and standard-library TCP/JSON/hashlib | Strict experiment configuration, bounded wire protocol, and content-addressed artifacts. |
| pytest, Ruff, mypy, GitHub Actions | Unit/integration checks, lint, strict typing, coverage, and CPU-safe CI. |

~~~text
README.md                         Single project guide
pyproject.toml                     Package, CLI entry point, test/lint/type settings
requirements/model-cu128.lock      Pinned model-runtime dependencies
configs/model/                     VLM and artifact contract; fake adapter settings
configs/hardware/                  Edge, localhost cloud, and pending cloud inventories
configs/experiments/               Runnable smoke, local, shared benchmark, and templates
data/manifests/                    Frozen synthetic correctness and benchmark requests
src/adaptive_vlm/cli.py            Public commands and policy/server wiring
src/adaptive_vlm/models/moondream.py  Pinned full, vision-only, and text-only adapter
src/adaptive_vlm/execution/        Edge, cloud, and distributed execution policies
src/adaptive_vlm/scheduler/        Phase 1 configuration, state, capability, and manual decision contracts
src/adaptive_vlm/transport/        Framing, checksum, tensor payload, client, server
src/adaptive_vlm/benchmark/        Scheduling, raw-result recording, summaries, comparison
src/adaptive_vlm/data/             Manifest validation and VQA v2 subset importer
src/adaptive_vlm/config.py         Strict configuration and model-contract fingerprint
src/adaptive_vlm/contracts.py      Request, response, status, run schemas
src/adaptive_vlm/environment.py    Host, GPU, Python, and Git inventory
scripts/                          Focused verification, probe, and inspection utilities
tests/                            CPU-safe tests and repository image fixtures
artifacts/                         Generated run data (ignored; archive separately)
.model-cache/                      Downloaded pinned model/cache (ignored)
~~~
