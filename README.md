# adaptive-edge-cloud-vlm

Adaptive runtime orchestration and model partitioning for low-latency
vision-language model inference across heterogeneous edge and cloud resources.

The repository is being built in gated chunks. Chunks 1 through 3 provide the
experiment contracts, deterministic test foundation, pinned Moondream2
artifact, verified CUDA inventory, explicit vision/projector boundary, and a
measured all-edge execution policy. Networking begins in Chunk 4.

## Planning

- [Baseline implementation plan](docs/BASELINE_IMPLEMENTATION_PLAN.md)

## Development setup

Python 3.11 is the model-runtime reference version. Python 3.12 remains suitable
for the CPU-only development and quality-check environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the pinned CUDA model environment used from Chunk 2 onward:

```bash
uv python install 3.11.17
uv venv --python 3.11.17 .venv-model
uv pip install --python .venv-model/bin/python \
  -r requirements/model-cu128.lock
```

The lock targets Linux and CUDA 12.8. The model artifact itself is not included
in the repository; its immutable revision and hashes are recorded in
[`configs/model/moondream2_artifact.yaml`](configs/model/moondream2_artifact.yaml).

## Configuration

Baseline experiment files reference separate model and hardware files:

```text
configs/
├── experiments/
│   ├── baseline_edge.yaml
│   ├── baseline_cloud.yaml
│   ├── baseline_distributed.yaml
│   └── chunk1_smoke.yaml
├── hardware/
└── model/
```

Validate and print a fully resolved configuration:

```bash
adaptive-vlm config validate configs/experiments/baseline_edge.yaml
```

The Moondream2 model is `approved` after exact local decomposition equivalence.
The edge hardware is verified; cloud hardware is still `pending` and must be
recorded before remote benchmarks.

## CPU-only smoke run

The fake adapter hashes the prompt and fixture bytes to produce deterministic
output. It verifies configuration, request/result contracts, seeding, execution
policy wiring, and artifact writing without pretending to perform VLM
inference.

```bash
adaptive-vlm fake-infer \
  --config configs/experiments/chunk1_smoke.yaml \
  --image tests/fixtures/smoke_image.txt \
  --prompt "Describe this fixture." \
  --output-dir artifacts/smoke
```

Generated experiment data is stored under `artifacts/` and ignored by Git. A
small illustrative manifest is available in
[`docs/examples/sample_run_manifest.json`](docs/examples/sample_run_manifest.json).

## Environment inventory

Collect non-secret host, Python, NVIDIA, and Git metadata:

```bash
scripts/collect_environment.sh artifacts/environment.json
```

An unavailable GPU is recorded as structured data instead of failing the
command. WSL CUDA access and BF16 execution were verified in Chunk 2; restricted
development sandboxes may still hide `/dev/dxg`.

## Monolithic model probe

The Chunk 2 probe records deterministic outputs, load and request latency,
process RAM, and CUDA memory when available. Pass `--image` more than once to
exercise multiple correctness fixtures in a single model load:

```bash
scripts/probe_moondream.py \
  --revision 9a7d4024050840e001defacec2b00727e89149e6 \
  --artifact-sha256 70a7d94c0c8349eb58ed2d9e636ef2d0916960f321ecabeac6354b8ba3d7403f \
  --cache-dir .model-cache \
  --offline \
  --device cpu \
  --image tests/fixtures/moondream_smoke.ppm \
  --output artifacts/chunk2/moondream_bf16_cpu_probe.json
```

See the [feasibility report](docs/reports/chunk_2_feasibility.md) for the
measured edge result and the retained GPU infeasibility evidence.

## Local decomposition and all-edge inference

Verify that the original and decomposed CPU paths produce identical projected
tensors and native token IDs:

```bash
.venv-model/bin/python scripts/verify_local_decomposition.py \
  --config configs/experiments/baseline_edge.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache \
  --offline \
  --image tests/fixtures/moondream_smoke.ppm \
  --repetitions 2 \
  --output artifacts/chunk3/equivalence.json
```

Run the correctness-preserving all-edge policy and save its manifest and
request record:

```bash
.venv-model/bin/adaptive-vlm edge-infer \
  --config configs/experiments/baseline_edge.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache \
  --offline \
  --vision-device cpu \
  --text-device cpu \
  --image tests/fixtures/moondream_smoke.ppm \
  --prompt "Describe this image." \
  --output-dir artifacts/edge
```

Using `--vision-device cuda:0 --text-device cpu` is supported and substantially
accelerates vision processing, but it is currently experimental: CPU and CUDA
BF16 vision kernels did not produce identical complete output sequences. See
the [Chunk 3 report](docs/reports/chunk_3_decomposition.md) for measurements.

## Quality checks

```bash
ruff check .
mypy
pytest --cov=adaptive_vlm --cov-report=term-missing
```

CI runs only these CPU-safe checks and never downloads model weights.
