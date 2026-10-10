# Baseline progress report — Chunks 1 and 2

## What has been achieved

The project now has a reproducible foundation for comparing three VLM
execution configurations: all-edge, all-cloud, and vision-plus-projection at
the edge with language generation in the cloud.

Completed work includes:

- validated YAML contracts for models, hardware, and experiments;
- typed request, result, metric, and run-manifest structures;
- deterministic seeding, structured logging, atomic artifact writing, and a
  dependency-free fake inference path;
- CPU-only CI, linting, strict type checking, and 38 tests with 81.95% coverage;
- a pinned Moondream2 release, weight hash, remote-code fingerprints, tokenizer
  fingerprint, and locked Python 3.11 model environment;
- verified WSL access to the RTX 3050, CUDA execution, and BF16 operations;
- deterministic monolithic BF16 inference on three synthetic fixtures; and
- an explicit feasibility result: the complete model does not fit reliably on
  the 4 GiB GPU, but it runs on the edge CPU at a median warmed latency of about
  53 seconds and approximately 4.63 GiB process memory.

The full GPU failure is a baseline result, not a software fallback: two clean
loads failed consistently because model weights and KV cache leave no practical
VRAM for CUDA context, activations, or intermediate tensors.

## How to use the current implementation

### 1. Create the development environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the quality gates:

```bash
ruff check .
mypy
pytest --cov=adaptive_vlm --cov-report=term-missing
```

### 2. Validate an experiment configuration

```bash
adaptive-vlm config validate configs/experiments/baseline_edge.yaml
```

Replace `baseline_edge.yaml` with `baseline_cloud.yaml` or
`baseline_distributed.yaml` to inspect the other policies.

### 3. Exercise the infrastructure without downloading a VLM

```bash
adaptive-vlm fake-infer \
  --config configs/experiments/chunk1_smoke.yaml \
  --image tests/fixtures/smoke_image.txt \
  --prompt "Describe this fixture." \
  --output-dir artifacts/smoke
```

This validates configuration resolution, deterministic requests, policy
wiring, and artifact generation. It does not measure real VLM performance.

Collect non-secret environment metadata with:

```bash
scripts/collect_environment.sh artifacts/environment.json
```

### 4. Create the pinned model environment

Install `uv`, then run:

```bash
uv python install 3.11.17
uv venv --python 3.11.17 .venv-model
uv pip install --python .venv-model/bin/python \
  -r requirements/model-cu128.lock
```

The first online probe populates the ignored `.model-cache/` directory. Later
runs should add `--offline` to prevent network access.

```bash
.venv-model/bin/python scripts/probe_moondream.py \
  --revision 9a7d4024050840e001defacec2b00727e89149e6 \
  --artifact-sha256 70a7d94c0c8349eb58ed2d9e636ef2d0916960f321ecabeac6354b8ba3d7403f \
  --cache-dir .model-cache \
  --device cpu \
  --max-new-tokens 8 \
  --warmups 1 \
  --repetitions 2 \
  --image tests/fixtures/moondream_smoke.ppm \
  --output artifacts/chunk2/cpu_probe.json
```

After downloading, inspect the cached weight artifact and compare its SHA-256
with `configs/model/moondream2_artifact.yaml`:

```bash
.venv-model/bin/python scripts/inspect_model_artifact.py \
  .model-cache/hub/models--vikhyatk--moondream2/snapshots/\
9a7d4024050840e001defacec2b00727e89149e6/model.safetensors
```

Then repeat the probe with `--offline`. Pass `--image` multiple times to test
several fixtures under one model load. Generated results are written beneath
`artifacts/`, which is intentionally excluded from Git.

## Current limitations and next step

The cloud hardware is not configured, and the real all-cloud and distributed
policies do not exist yet. The public model API also returns text rather than
native generated token IDs; the current probe therefore records repeatable
text and re-encoded IDs.

Chunk 3 is the next correctness gate: implement the real Moondream adapter,
capture the projected visual tokens, resume language generation from that
boundary, and prove that monolithic and locally decomposed inference produce
the same native token IDs before adding networking.

For detailed evidence, see `chunk_2_feasibility.md`, the model-selection ADR,
and the Chunk 2 handoff in `docs/`.
