# Usage guide

How to install the project, run each execution path on one machine or on two hosts, and use every command of the `adaptive-vlm` CLI. The campaign procedures are in the runbooks ([AWS](runbooks/aws-campaign.md), [physical pair](runbooks/physical-campaign.md)); regenerating the thesis results from the archived data is in [Reproducing the results](reproducibility.md).

Many commands below belong to the **local prototype**: the first, single-machine version of the system, with three configurations, CPU or loopback hosts and the schema-1 predictor. They still work and are tested. The AWS campaigns use `interleaved-benchmark`, the campaign supervisor and the r2 pipeline instead (see [Interleaved placements](#interleaved-placements-g2-v2-g2-v2-and-g5)).

## Contents

- [Execution paths, transport and measurement](#execution-paths-transport-and-measurement)
- [Setup](#setup)
- [Usage](#usage)
- [CLI reference](#cli-reference)
- [Using a separate cloud machine](#using-a-separate-cloud-machine)

## Execution paths, transport and measurement

### Configurations

One pinned model, [Moondream2](../configs/model/moondream2.yaml) at revision `9a7d4024050840e001defacec2b00727e89149e6`, runs in every configuration with the same BF16 weights, Pillow preprocessing, batch size one, greedy decoding and at most 32 new tokens. The [artifact manifest](../configs/model/moondream2_artifact.yaml) pins the weights, the tokenizer and the reviewed remote-code hashes, and loading verifies them before any inference.

| Configuration | CLI placement / mode | Edge | Cloud | Crosses the link |
| --- | --- | --- | --- | --- |
| `edge/full` | `edge` | Preprocessing, vision encoder, projector and language model | — | Nothing |
| `cloud/raw_image` | `cloud` | Reads the image file | The whole model | Encoded image bytes and prompt |
| `distributed/projected/synchronous` | `distributed` | Preprocessing, vision encoder and projector | Image and prompt prefill, then generation | Prompt and the projected BF16 tensor |
| `distributed/projected/prompt_first` | `distributed_prompt_first` (`--pipeline prompt_first` for one-off runs) | Same as synchronous | Same; prepares the prompt while the edge encodes | Prompt first, then the tensor |
| `distributed/projected/int8` | `distributed_int8` | Same, then per-row int8 quantisation | Dequantises to BF16, then as synchronous | int8 tensor and one fp16 scale per row |

The local prototype commands (`benchmark`, `adaptive-infer`, `adaptive-benchmark`) use the first three; `interleaved-benchmark` and the AWS campaigns use all five. The adaptive policy (G5) is a sixth placement of `interleaved-benchmark`; see [Adaptive runtime](design/adaptive-runtime.md).

For the split paths, the adapter captures the tensor after the vision encoder and projector, **before** the language model's image prefill: BF16 `[729, 2048]`, 729 visual tokens and 2,985,984 raw bytes. The model's public image-encoding API is not the boundary, because it continues into prefill. The cloud does not recompute vision, and the edge does not generate tokens. The [split boundary](design/split-boundary.md) note documents the other candidate boundaries, none of which is executable. The loader builds the full checkpoint before pruning unneeded modules, so **load-time RAM is not partitioned**, even though the steady-state components are.

~~~mermaid
flowchart LR
    I["Image + prompt"] --> L["Edge: full Moondream"]
    L --> O["Answer"]
    I --> RC["Edge: raw-image client"]
    RC -- "image bytes + prompt" --> FM["Cloud: full Moondream"]
    FM --> O
    I --> VP["Edge: vision encoder + projector"]
    VP -- "projected tensor (bf16 or int8) + prompt" --> TM["Cloud: language model"]
    TM --> O
~~~

### Transport

The wire protocol (`src/adaptive_vlm/transport/`) uses versioned, length-prefixed frames with bounded JSON metadata, a binary payload and a SHA-256 checksum. The client checks the model contract fingerprint and protocol version; bad frames, mismatched contracts, timeouts and server errors are rejected and recorded. The server handles one request at a time, matching the batch-size-one latency experiment. `adaptive-server` accepts both raw images and projected tensors, so one loaded model serves every remote configuration. There is **no TLS, authentication or authorisation**: use it only on a private network (see [Using a separate cloud machine](#using-a-separate-cloud-machine)).

On one machine, the cloud role is a separate server process bound to `127.0.0.1:5050`. That validates the component separation, the TCP framing, the health checks and the exact application bytes, but not an independent cloud GPU or a real link. Only one cloud-role server can use port 5050 at a time.

### Correctness, benchmarking, and observability

- A [12-request correctness manifest](../data/manifests/correctness.jsonl) and a [100-request fixture manifest](../data/manifests/benchmark_fixture.jsonl) are checked in. The loader verifies manifest revision, unique IDs, and each image's SHA-256 before running.
- The `benchmark` command runs one of the three original policies against one resolved experiment configuration; `interleaved-benchmark` runs several placements back to back per request. It seeds the runtime, runs warm-ups first, shuffles each measured sweep reproducibly, and records one attempt per request/repetition. The model is loaded once per run; cold load time is separate from steady-state request latency.
- Each run writes `plan.json`, `run_manifest.json`, fsynced `requests.jsonl`, and `summary.json` to its unique run directory. The run manifest includes resolved config, dataset/dependency hashes, Git state, host/Python/GPU inventory, and seed. Raw rows include config/model/dataset/image hashes, output tokens, timings, bytes, status, and any error.
- The runner preserves failed and out-of-memory attempts in the denominator, but excludes their latency from successful-latency summaries. Summaries provide median, p95, IQR, mean, failure rate, wire bytes where relevant, and scored-task counts. `--resume` accepts only the original plan and skips already recorded trials.
- `compare-runs` verifies provenance and compares **every measured request/repetition pair** across policies. It reports exact-token and decoded-text mismatches, first divergent token, failures, and per-policy task scores.
- Per-request metrics include client-observed end-to-end time, available vision/projector/prefill/generation stage times, process RSS/CUDA memory, and exact application-layer wire bytes. The adaptive path also records bounded live health/capability probes and selection overhead. Application bytes, socket-send rate, and localhost timings are not physical link throughput or one-way delay.

The synthetic fixture quality score is case-insensitive exact match after trimming whitespace. It is only a correctness diagnostic. The VQA importer accepts local VQA v2 validation files and preserves ten annotator answers, but the runner's VQA consensus normalization is provisional: use the [official VQA evaluation procedure](https://visualqa.org/evaluation.html) before reporting public-dataset accuracy.

## Setup

Run commands from the repository root. The model environment and lock target Linux or WSL with Python 3.11 and CUDA 12.8 PyTorch; the dependency-light development environment also works with Python 3.12. The model weights are **not** committed: the first run downloads about 3.9 GB. A 4 GiB GPU cannot hold the full BF16 model resident; use `--edge-residency swap` there, or the CPU for exact reference runs.

### 1. Development environment

~~~bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/adaptive-vlm config validate configs/experiments/local_campaign.yaml
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

For the **first** model run, omit `--offline` so the exact pinned weights, tokenizer, and remote code can populate `.model-cache`. This downloads a multi-gigabyte checkpoint and verifies it against [the artifact manifest](../configs/model/moondream2_artifact.yaml):

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
| [`fake_smoke.yaml`](../configs/experiments/fake_smoke.yaml) | Dependency-free `fake-infer` sanity check. |
| [`fake_cloud.yaml`](../configs/experiments/fake_cloud.yaml) | Fake localhost server/client and transport tests. |
| [`local_cloud.yaml`](../configs/experiments/local_cloud.yaml) | Real localhost full-model cloud server and one-off client. |
| [`local_distributed.yaml`](../configs/experiments/local_distributed.yaml) | Real localhost text-only server and vision client. |
| [`fake_benchmark.yaml`](../configs/experiments/fake_benchmark.yaml) | Fast benchmark-runner checks without a VLM; edge or fake cloud. |
| [`local_campaign.yaml`](../configs/experiments/local_campaign.yaml) | Shared real-model correctness/benchmark config for `--mode edge`, `cloud`, or `distributed`. |
| [`baseline_edge.yaml`](../configs/experiments/baseline_edge.yaml), [`baseline_cloud.yaml`](../configs/experiments/baseline_cloud.yaml), [`baseline_distributed.yaml`](../configs/experiments/baseline_distributed.yaml) | Physical-host templates with pending dataset/cloud details; copy and fill before physical experiments. |

The one-off server commands require an experiment file whose YAML `mode` matches the server role. In contrast, `benchmark --mode` selects the policy independently of the YAML `mode`, allowing all three runs to share the exact same resolved model/data/generation configuration. Remote server and benchmark files must still agree on model contract, address, port, and protocol. `configs/hardware/local_model_cloud.yaml` describes the **same host** acting as cloud, not a second PC.

## Usage

All commands below assume setup is complete and are run from the repository root. A one-off inference saves a run manifest and request record under `--output-dir/RUN_ID/` when that option is supplied. Final JSON is printed to stdout; progress is generally printed to stderr, while `benchmark` also prints trial progress to stdout. For automation, read its saved `summary.json` instead of assuming stdout contains JSON alone.

### Fake smoke tests

~~~bash
.venv/bin/adaptive-vlm fake-infer \
  --config configs/experiments/fake_smoke.yaml \
  --image tests/fixtures/smoke_image.txt \
  --prompt "Describe this fixture." --output-dir artifacts/fake-smoke

# In a separate terminal, start this server before the client:
.venv/bin/adaptive-vlm cloud-server --config configs/experiments/fake_cloud.yaml
.venv/bin/adaptive-vlm cloud-infer \
  --config configs/experiments/fake_cloud.yaml \
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
  --config configs/experiments/local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
~~~

After the readiness JSON appears, terminal B:

~~~bash
.venv-model/bin/adaptive-vlm cloud-infer \
  --config configs/experiments/local_cloud.yaml \
  --image /path/to/photo.jpg --prompt "What is happening in this photo?" \
  --output-dir artifacts/manual-cloud
~~~

The edge client sends raw image bytes and text, while the server runs the complete model. The client does not need model weights in this mode.

### Split vision and language on localhost

Stop `cloud-server` first, then start the text-only server in terminal A:

~~~bash
.venv-model/bin/adaptive-vlm distributed-server \
  --config configs/experiments/local_distributed.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
~~~

Terminal B:

~~~bash
.venv-model/bin/adaptive-vlm distributed-infer \
  --config configs/experiments/local_distributed.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --vision-device cpu \
  --pipeline synchronous --image /path/to/photo.jpg \
  --prompt "What is happening in this photo?" \
  --output-dir artifacts/manual-distributed
~~~

Change `--pipeline` to `prompt_first` only when testing that explicit overlap variant. Each side verifies the same model/generation fingerprint.

### Reproducible benchmark and comparison

Start with the [real shared configuration](../configs/experiments/local_campaign.yaml). The following is a **one-sweep correctness check**; the config defaults to 10 warm-ups and 30 full measured sweeps if overrides are omitted. A sweep processes every manifest request once in seeded random order.

~~~bash
# No server needed:
.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/local_campaign.yaml \
  --mode edge --run-dir artifacts/example-edge \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --vision-device cpu --text-device cpu --warmups 1 --repetitions 1

# Start cloud-server with local_cloud.yaml in another terminal:
.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/local_campaign.yaml \
  --mode cloud --run-dir artifacts/example-cloud \
  --warmups 1 --repetitions 1

# Stop cloud-server; start distributed-server with local_distributed.yaml:
.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/local_campaign.yaml \
  --mode distributed --run-dir artifacts/example-distributed \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --vision-device cpu \
  --pipeline synchronous --warmups 1 --repetitions 1

.venv/bin/adaptive-vlm compare-runs \
  artifacts/example-edge artifacts/example-cloud artifacts/example-distributed \
  --output artifacts/example-comparison.json
~~~

Choose **fresh** run directories; an existing directory is refused unless `--resume` is given with the same arguments, config, manifest, and plan. The benchmark records individual failures but may still exit successfully after completing the schedule; inspect `status_counts`/`failure_rate`. Likewise, inspect `compare-runs`'s `complete_and_equal` field rather than relying only on its exit code.

For a fast 100-request storage and scheduling run without Moondream, use `fake_benchmark.yaml`, `--mode edge`, `--manifest data/manifests/benchmark_fixture.jsonl`, `--warmups 10` and `--repetitions 1`. That fixture has no task-quality references.

### Interleaved placements (G2 v2, G2 v2+ and G5)

`interleaved-benchmark` is the measurement command of every campaign since G2 v2. For each request it runs every selected placement back to back, in a seeded random order, so slow drift affects all placements alike.
- **Placements:** `--placements` chooses among `edge`, `cloud`, `distributed`, `distributed_prompt_first` and `distributed_int8` (default: the first three). The `--adaptive-*` options add the G5 adaptive policy as a sixth placement.
- **One edge model** serves `edge/full` and the vision half of every split placement. Before each timed request the needed stage is activated outside the timer, and its cost is recorded as `stage_prepare_ms`. `--edge-residency swap` keeps the weights in pinned host memory for GPUs that can't hold the model (not supported on the AWS vGPU, which runs `resident`).
- **Output:** one run directory per placement under `--run-dir`, each accepted by `profile-runs`, plus the global order in `interleave.json`.
- **Loads:** `--edge-load-config` and `--edge-load-level` run the edge GPU load generator, as a separate supervised process or, with `"generator": "in_process"` (the AWS levels, amendment A2a-3), on its own CUDA stream inside the benchmark; `--cloud-load-level` only records the cloud level, which the campaign supervisor drives on the cloud.
- **Failures:** if the local CUDA context fails, the attempt is recorded and the command exits with code 75 so a supervisor can resume it with `--resume` (76: the edge load failed). Core dumps are disabled.

~~~bash
# Start adaptive-server (fake or Moondream) first; then:
.venv-model/bin/adaptive-vlm interleaved-benchmark   --config configs/experiments/aws_vqa_v2.yaml   --run-dir artifacts/example-interleaved   --split configs/evaluation/physical_vqa_split_v2.json --split-name training   --placements edge cloud distributed distributed_prompt_first distributed_int8   --order-seed 5113 --repetition-indices 0   --artifact-manifest configs/model/moondream2_artifact.yaml   --cache-dir .model-cache --offline --edge-residency resident
~~~

In the campaigns, `scripts/campaign/run_g2v2_campaign.py` builds these commands per sub-block from the grid and settings, shapes the link and drives the loads; see the [AWS campaign runbook](runbooks/aws-campaign.md). To try the whole supervisor locally with fake hosts and injected faults:

~~~bash
.venv-model/bin/python scripts/campaign/rehearse_g2v2_faults.py   --settings configs/campaigns/g5_loopback_rehearsal.json --fresh
~~~

### Analysis of the AWS campaigns

The G2 v2+ derivations, the r2 predictor (G3), the G4 evaluation, the G5 analysis and the final figures run from scripts under `scripts/campaign/` and `scripts/prediction/`. The commands, inputs and outputs for every table and figure are in [Reproducing the results](reproducibility.md).

### Profiling fixed-policy runs

The fixed-policy benchmark now records trial start time, exact model prompt-token count, prompt prefill/decode durations, transport subintervals, response bytes, and memory readings. After three completed runs with the same frozen schedule, build a validated profile:

~~~bash
.venv/bin/adaptive-vlm profile-runs \
  artifacts/example-edge artifacts/example-cloud artifacts/example-distributed \
  --specification configs/profiles/local_fixture.json \
  --output-dir artifacts/example-profile
~~~

Inspect `artifacts/example-profile/summary.json` for coverage/failures and `profile.jsonl` for each warmup, measured success, and failure. The [schema note](design/profile-format.md) defines timing boundaries, null reasons, residuals, and units. The [split inspection](design/split-boundary.md) documents candidate graph boundaries that remain inactive. The supplied specification labels the local loopback network as uncontrolled; it cannot support physical-network crossover claims.

### Prototype predictor calibration (schema 1)

Fit the component predictor and evaluate the frozen request holdout from a local profile:

~~~bash
.venv/bin/adaptive-vlm calibrate-predictor \
  --profile artifacts/phase2/2026-10-26/local-1/profile/profile.jsonl \
  --split configs/predictor/local_holdout.json \
  --output-dir artifacts/example-predictor
~~~

Read `evaluation.json` for per-candidate error, interval coverage, ranking against the constant-median baseline, and gate flags. `calibration.json` contains the fitted components and coverage bounds; `predictions.jsonl` keeps each held-out task snapshot and prediction. The [predictor contract](design/predictor-v1-format.md) defines units and out-of-coverage behavior. This calibration is used only by explicitly flagged diagnostic adaptive runs.

### Prototype offline selector and oracle

Evaluate the pure selector on frozen local predictions and matched observed outcomes:

~~~bash
.venv/bin/adaptive-vlm evaluate-selector \
  --predictions artifacts/phase3/2026-10-26/clean-1/predictions.jsonl \
  --specification configs/oracle/local_holdout.json \
  --output-dir artifacts/example-selector
~~~

Read `evaluation.json` for regret, top-choice agreement, uncertainty, exclusions, and gate flags. `decisions.jsonl` preserves each candidate prediction and reason beside the retrospective measured oracle. The [selector contract](design/selector-and-oracle.md) defines eligibility and pairing. The local prototype never met its G3 or G4 targets, so the prototype's `adaptive-benchmark` requires a diagnostic override. (The thesis gates G3 and G4 passed later on AWS with the r2 pipeline.)

### Prototype adaptive run

With the pinned Moondream cache, start `adaptive-server` with `configs/experiments/local_campaign.yaml`, then run the frozen local mixed-choice campaign:

~~~bash
.venv-model/bin/adaptive-vlm adaptive-benchmark \
  --config configs/experiments/local_campaign.yaml \
  --manifest data/manifests/correctness.jsonl \
  --calibration artifacts/phase3/2026-10-26/clean-1/calibration.json \
  --replay configs/adaptive/local_replay_health.json \
  --run-dir artifacts/example-adaptive \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --warmups 1 --repetitions 1 --diagnostic-allow-unpassed-gates
~~~

The diagnostic flag explicitly permits the local fit, which failed its G3 target, for an integration check; it does not approve performance selection. `requests.jsonl` contains every candidate prediction, decision, actual result, overhead and any retry. Use `analyze-adaptive` with matching edge/cloud/distributed fixed controls, as described in the [execution contract](design/adaptive-execution.md). The command requires a resident full edge adapter for exact prompt-token counting.

### Prototype network trace replay and live monitor

Replay the [synthetic controlled trace](../configs/network_traces/synthetic_trace.json) through the predictor and selector without loading a model:

~~~bash
.venv/bin/adaptive-vlm replay-network-trace \
  --calibration configs/network_traces/synthetic_calibration.json \
  --trace configs/network_traces/synthetic_trace.json \
  --output-dir artifacts/example-network-trace
~~~

For a real diagnostic, add `--monitor-mode live --probe-timeout-ms 2000 --probe-max-age-ms 5000` to `adaptive-benchmark`, then run `analyze-monitor RUN_DIR --output RUN_DIR/monitor_analysis.json`. The [monitor contract](design/live-monitor-and-trace-replay.md) defines probe age, failure behavior and passive-rate limits. Health-probe round trip and socket-send rate are recorded as distinct application observations and are not fed to the current loopback predictor.

### Import a VQA validation subset

The repository does not redistribute VQA questions, answers, or COCO images. Once the official VQA v2 val2014 JSON files and images are available locally, freeze a deterministic subset:

~~~bash
.venv/bin/adaptive-vlm import-vqa-v2 \
  --questions data/raw/vqa/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations data/raw/vqa/v2_mscoco_val2014_annotations.json \
  --images-dir data/raw/coco/val2014 \
  --output data/manifests/vqa_val2014_seed42.jsonl \
  --count 100 --seed 42 --revision vqa-v2-val2014-local
~~~

Set a copied experiment YAML's `dataset.manifest_path` and `dataset.revision` to the new file and exact revision. The importer samples paired question/annotation IDs with a seed, stores ten human answers and image hashes, and refuses to overwrite an existing manifest. See the [official VQA data format](https://visualqa.org/download.html). Before thesis-quality claims, run the official evaluator: the in-run VQA consensus score lacks full official answer normalization.

The physical v1 campaign already freezes 100 official validation requests in `data/manifests/physical_vqa_v1.jsonl`, with a disjoint split in `configs/evaluation/physical_vqa_split_v1.json` and experiment config `configs/experiments/physical_vqa_v1.yaml`. Its [dataset protocol](physical-testbed/vqa-dataset.md) records source hashes and edge setup. `scripts/vqa/fetch_vqa_subset_images.py` fetches only the seeded image subset from the official COCO bucket; source files remain ignored under `data/raw/`.

### Prototype paired evaluation

The frozen [paired evaluation protocol](design/evaluation-protocol.md) defines the split, metrics, confidence intervals and open gates. Produce the local descriptive report from fixed-policy and adaptive runs:

~~~bash
.venv/bin/adaptive-vlm analyze-thesis \
  --specification configs/evaluation/local_thesis_diagnostic.json \
  --adaptive-run artifacts/phase5/2026-10-26/local-1 \
  --fixed-runs artifacts/phase2/2026-10-26/local-1/edge \
    artifacts/phase2/2026-10-26/local-1/cloud \
    artifacts/phase2/2026-10-26/local-1/distributed \
  --output-dir artifacts/phase7-reproduction
~~~

`report.json` contains validation-only best-static choice, paired latency/SLA intervals, failure counts, bytes, prediction error and measured-oracle regret. `paired_trials.jsonl` keeps the request-level evidence. Each input file and the frozen specification is hashed. This 2-request local holdout was already inspected, so its intervals are descriptive, not independent evidence.

For VQA, run `prepare-vqa-eval-subset` with the imported manifest and original official question/annotation JSONs, then `export-vqa-predictions` for each policy run and repetition. Feed those files to `scripts/vqa/run_official_vqa_eval.py` with a pinned checkout of the [official VQA evaluator](https://github.com/GT-Vision-Lab/VQA/tree/master/PythonEvaluationTools). The upstream source uses Python 2 syntax, so run the adapter in a compatible Python 2 environment. The exporter includes failed requests as empty answers and writes a hash/provenance sidecar. The official loader requires predictions for exactly the annotation IDs, which is why the subset command is needed. Record the upstream commit and the adapter's hashed evaluation output with the campaign. The repository does not claim official accuracy until that evaluation has run.

### Validation and helper scripts

~~~bash
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest --cov=adaptive_vlm --cov-report=term-missing
.venv/bin/adaptive-vlm environment --output artifacts/environment.json
~~~

CPU-only CI runs Ruff, strict mypy, and pytest with an 80% coverage threshold without downloading model weights. The real TCP subprocess test uses an ephemeral port and skips only when a restricted sandbox forbids loopback binding. `scripts/verification/verify_local_decomposition.py`, `verify_cloud_equivalence.py`, and `verify_distributed_equivalence.py` reproduce focused model correctness checks; `verify_cloud_stability.py` and `verify_distributed_stability.py` exercise repeated requests. `scripts/model/probe_moondream.py` probes feasibility, and `inspect_model_artifact.py` inspects local safetensors. Use `--help` on a script for its experiment-specific options.

## CLI reference

Run `adaptive-vlm --help` or `adaptive-vlm COMMAND --help` for machine-readable, version-matched usage. This table covers the current public commands; bracketed flags are optional.

The parser and command routing live in `cli.py`; command execution lives in [`commands/`](../src/adaptive_vlm/commands). Campaign drivers and analyses are scripts, not CLI commands: see `scripts/campaign/`, `scripts/prediction/`, `scripts/cloud/` and `scripts/vqa/`, each with `--help`.

| Command | Required arguments | Optional arguments and defaults |
| --- | --- | --- |
| `config validate` | `PATH` | `--output PATH` writes resolved JSON; it is always printed. |
| `environment` | None | `--workdir PATH` (current directory), `--output PATH`. |
| `fake-infer` | `--config PATH --image PATH --prompt TEXT` | `--request-id` (`smoke-0001`), `--output-dir PATH`, `-v/--verbose`. |
| `edge-infer` | `--config PATH --artifact-manifest PATH --image PATH --prompt TEXT` | `--cache-dir` (`.model-cache`), `--request-id` (`edge-0001`), `--vision-device` (`cpu`), `--text-device` (`cpu`), `--offline`, `--output-dir PATH`, `-v/--verbose`. |
| `cloud-server` | `--config PATH` | `--artifact-manifest PATH` (required for Moondream, not fake), `--cache-dir` (`.model-cache`), `--device` (`cpu`), `--offline`, `--bind-host HOST` (otherwise config host), `-v/--verbose`. |
| `adaptive-server` | `--config PATH --artifact-manifest PATH` | Loads one full Moondream model for raw-image and projected-tensor requests. `--cache-dir` (`.model-cache`), `--device` (`cpu`), `--offline`, `--bind-host HOST`, `--report-resource-state` (sample NVML and host memory every 0.5 s in the background and attach the latest sample to each response as `server_state_*`), `--load-heartbeat PATH` (cloud-load worker heartbeat; adds the load level, liveness and duty to the state), `-v/--verbose`. |
| `cloud-infer` | `--config PATH --image PATH --prompt TEXT` | `--request-id` (`cloud-0001`), `--output-dir PATH`, `-v/--verbose`. |
| `adaptive-infer` | `--config PATH --configuration-id ID --image PATH --prompt TEXT` | Manual selection among `edge/full`, `cloud/raw_image` and `distributed/projected/synchronous` through one resident dispatcher; see `--help` for the edge component, memory, network-state and artifact options. |
| `distributed-server` | `--config PATH --artifact-manifest PATH` | `--cache-dir` (`.model-cache`), `--device` (`cpu`), `--offline`, `--bind-host HOST`, `-v/--verbose`. |
| `distributed-infer` | `--config PATH --artifact-manifest PATH --image PATH --prompt TEXT` | `--cache-dir` (`.model-cache`), `--request-id` (`distributed-0001`), `--vision-device` (`cpu`), `--pipeline` (`synchronous` or `prompt_first`), `--offline`, `--output-dir PATH`, `-v/--verbose`. |
| `benchmark` | `--config PATH --mode edge\|cloud\|distributed --run-dir PATH` | `--manifest PATH` (config dataset path), `--artifact-manifest PATH` (required for real edge/distributed), `--cache-dir` (`.model-cache`), `--vision-device` (`cpu`), `--text-device` (`cpu`), `--offline`, `--warmups N` and `--repetitions N` (config values), `--pipeline` (`synchronous` or `prompt_first`), `--resume`, `-v/--verbose`. |
| `interleaved-benchmark` | `--config PATH --run-dir PATH --order-seed N` | `--split PATH --split-name NAME` (default: all manifest IDs), `--placements` (any of `edge cloud distributed distributed_prompt_first distributed_int8`; default the first three), `--manifest PATH`, `--repetition-indices N…` (all configured), `--warmups N` (config), `--edge-device` (`cuda:0`), `--edge-residency` (`resident` or `swap`), `--artifact-manifest PATH`, `--cache-dir`, `--offline`, `--resume`, `--max-consecutive-failures N`, `--resource-state` (edge NVML, host memory and Windows GPU counters in every record), `--edge-load-config PATH --edge-load-level L1` (run the edge load generator; `L0` = none), `--cloud-load-level C0\|C2` (recorded only), and the G5 adaptive placement: `--adaptive-calibration PATH --adaptive-sensing PATH --adaptive-features PATH` with `--adaptive-network-probe PATH` (the sub-block's `network_pre.json`) or `--adaptive-network fixed:<bps>:<ms>`, and `--adaptive-edge-probe torch\|fixed:<ms>`. Exit code 75: the local CUDA context failed; 76: the load generator failed. Resume after either. |
| `compare-runs` | At least two `RUN_DIR` paths | `--output PATH` saves comparison JSON; comparison is always printed. |
| `profile-runs` | Completed matching `RUN_DIR` paths | `--specification PATH --output-dir PATH`; validates raw trials, hashes and coverage and writes versioned profile rows. |
| `calibrate-predictor` | `--profile PATH --split PATH --output-dir PATH` | Fits only frozen training rows and writes a calibration, held-out evaluation and per-trial predictions. |
| `evaluate-selector` | `--predictions PATH --specification PATH --output-dir PATH` | Checks a frozen predictions hash and writes offline decisions plus matched measured-oracle metrics. |
| `adaptive-benchmark` | `--config PATH --manifest PATH --calibration PATH --replay PATH --run-dir PATH --artifact-manifest PATH` | Runs a resumable mixed-choice campaign; requires passing G3/G4 reports or explicit `--diagnostic-allow-unpassed-gates`. |
| `analyze-adaptive` | `ADAPTIVE_RUN_DIR EDGE_DIR CLOUD_DIR DISTRIBUTED_DIR` | Matches each choice to a fixed control and audits tokens, placement, wire bytes, and decisions. |
| `replay-network-trace` | `--calibration PATH --trace PATH --output-dir PATH` | Replays frozen bandwidth/RTT/health states through the predictor and selector; reports switches and safety. |
| `analyze-monitor` | `RUN_DIR --output PATH` | Audits live probe timestamps, age, fail-closed decisions, passive updates, and monitor overhead. |
| `import-vqa-v2` | `--questions PATH --annotations PATH --images-dir PATH --output PATH --revision LABEL` | `--count N` (100), `--seed N` (42), `--candidates PATH` (frozen candidate order), `--two-crop-count N`, `--prompt-template TEXT`. |
| `analyze-thesis` | `--specification PATH --adaptive-run DIR --fixed-runs EDGE_DIR CLOUD_DIR DISTRIBUTED_DIR --output-dir DIR` | Split-safe paired analysis with raw trial table and input hashes. |
| `prepare-vqa-eval-subset` | `--manifest PATH --questions PATH --annotations PATH --output-dir DIR` | Writes official-schema question/annotation files for exactly the frozen IDs. |
| `export-vqa-predictions` | `--manifest PATH --run-dir DIR --output PATH` | `--repetition N` (0); writes official-format answers and provenance. |

`--config` references an experiment YAML, not a model YAML. `--artifact-manifest` references [the pinned artifact manifest](../configs/model/moondream2_artifact.yaml); it is distinct from the request `--manifest` used by `benchmark`. `--device` places the cloud server's model/text stage; `--vision-device` and `--text-device` name edge-side placements. `--offline` forbids model downloads and is appropriate only after the cache is populated. `--bind-host` controls the listening interface, while `transport.host` in the config is the client-reachable address. `-v` adds detail; core phase milestones and benchmark trials are already visible by default.

## Using a separate cloud machine

These are the manual steps for a second Linux or WSL machine as the cloud, with the one-off and fixed-policy commands. They were followed for the physical pair ([hosts and baseline](physical-testbed/hosts-and-baseline.md)); the unattended campaigns automate them ([physical](runbooks/physical-campaign.md) and [AWS](runbooks/aws-campaign.md) runbooks). `adaptive-server` can replace both `cloud-server` and `distributed-server`, since it serves every remote configuration.

1. Check out the same Git commit on both PCs. Install the pinned model environment on the cloud; install it on the edge as well for `edge` and `distributed`. Place and verify the pinned artifact in each host's local cache before using `--offline`. The split loader still needs RAM for a transient full checkpoint on **both** hosts.
2. Record the cloud PC's actual OS, CPU, RAM, GPU/VRAM, driver, and software in a new `configs/hardware/*.yaml` file with `status: verified`. Do not label [`cloud_pending.yaml`](../configs/hardware/cloud_pending.yaml) or [`local_model_cloud.yaml`](../configs/hardware/local_model_cloud.yaml) as a physical machine.
3. Copy the cloud and distributed experiment files into site-specific YAMLs. Point `cloud_hardware_config` at the verified inventory and set `transport.host` to the cloud address reachable from the edge (same port, timeout, and protocol on both machines). Keep `model_config`, generation settings, request manifest/revision, and seed aligned. The server YAML must use `mode: cloud` for `cloud-server` or `mode: distributed` for `distributed-server`. Keep a single shared benchmark YAML for all three client-side `--mode` selections.
4. Restrict inbound TCP to the edge host on a trusted network, or use a protected VPN/SSH tunnel. The transport has integrity checks but **no encryption or authentication**. Never expose port 5050 directly to the public internet. If the cloud must listen on an interface other than its client-reachable address, pass `--bind-host` appropriately (for example `0.0.0.0` only behind a restrictive firewall).
5. For all-cloud, start `cloud-server` on the cloud with the site cloud YAML, `--artifact-manifest`, and `--device cpu` or a suitable GPU. Wait for readiness and matching model fingerprint, then run `cloud-infer` or `benchmark --mode cloud` on the edge with the site benchmark YAML. Only the cloud needs the full model for this policy.
6. Stop that server. For distributed, start `distributed-server` on the cloud using the site distributed YAML. Then run `distributed-infer` or `benchmark --mode distributed` on the edge with its verified vision model and cache. The server holds the text role; the edge sends the projected tensor.
7. Run `edge` on the edge as a local reference, then `compare-runs` over the three matching manifests. Archive the raw run directories, both host inventories, server readiness metadata, model/dependency hashes, and network topology. Investigate every token mismatch; CPU/GPU floating-point differences are possible. Measure bandwidth/RTT independently before drawing network-performance conclusions.

The actual cloud host, real-link G4 correctness, network-condition campaign, and any cross-device task-quality result remain pending.
