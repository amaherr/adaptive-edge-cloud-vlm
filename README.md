# adaptive-edge-cloud-vlm

Adaptive runtime orchestration and model partitioning for low-latency
vision-language model inference across heterogeneous edge and cloud resources.

The repository is being built in gated chunks. Chunk 1 provides the
configuration contracts, structured artifacts, deterministic fake adapter,
environment inventory, and CPU-only test foundation. It does not download or
execute a real VLM.

## Planning

- [Baseline implementation plan](docs/BASELINE_IMPLEMENTATION_PLAN.md)

## Development setup

Python 3.11 is the reference version. Python 3.12 is also suitable for the
CPU-only infrastructure in Chunk 1, but the model environment will be frozen
after the model-feasibility work.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

No PyTorch or model dependency is installed at this stage.

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

The Moondream2 model is marked `candidate`, the edge hardware is `declared`,
and the cloud hardware is `pending`. Those statuses are intentional and must
be replaced with verified records during the model and hardware feasibility
chunk.

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
command. GPU access from PyTorch is handled in Chunk 2.

## Quality checks

```bash
ruff check .
mypy
pytest --cov=adaptive_vlm --cov-report=term-missing
```

CI runs only these CPU-safe checks and never downloads model weights.
