# adaptive-edge-cloud-vlm

Adaptive runtime orchestration and model partitioning for low-latency
vision-language model inference across heterogeneous edge and cloud resources.

The repository is being built in gated chunks. Chunks 1 through 4 provide the
experiment contracts, deterministic test foundation, pinned Moondream2
artifact, verified CUDA inventory, explicit vision/projector boundary, and a
measured all-edge policy plus a bounded all-cloud TCP path.

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

## All-cloud localhost inference

Start the pinned monolithic model server in one terminal:

```bash
.venv-model/bin/adaptive-vlm cloud-server \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache \
  --offline \
  --device cpu
```

After it prints a JSON readiness record, submit a raw image and prompt from a
second terminal:

```bash
.venv-model/bin/adaptive-vlm cloud-infer \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --image tests/fixtures/moondream_smoke.ppm \
  --prompt "Describe this image." \
  --output-dir artifacts/cloud
```

The client checks the protocol version and a fingerprint covering the model
artifact, precision, preprocessing, and generation contract before accepting a
response. See the [cloud deployment guide](docs/deployment/cloud.md) before
using separate hosts; this baseline transport intentionally has no TLS or
authentication and must not be exposed to an untrusted network.

Inference and verification commands print progress milestones to stderr by
default, including model loading, readiness checks, inference, shutdown, and
elapsed time. Add `--verbose` for configuration paths, model/device details,
contract fingerprints, and per-request stability progress. The final JSON stays
on stdout, so redirection and machine parsing remain safe.

Reproduce the Chunk 4 localhost gates with:

```bash
.venv/bin/python scripts/verify_cloud_stability.py \
  --config configs/experiments/chunk4_fake_cloud.yaml \
  --image tests/fixtures/smoke_image.txt \
  --requests 100 \
  --output artifacts/chunk4/cloud_stability.json

.venv-model/bin/python scripts/verify_cloud_equivalence.py \
  --config configs/experiments/chunk4_local_cloud.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache \
  --offline \
  --device cpu \
  --image tests/fixtures/moondream_smoke.ppm \
  --output artifacts/chunk4/cloud_equivalence.json
```

## Distributed vision and language inference

Start the text-only server in one terminal:

```bash
.venv-model/bin/adaptive-vlm distributed-server \
  --config configs/experiments/chunk5_local_distributed.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --device cpu
```

After the readiness JSON appears, run the vision stage in another terminal:

```bash
.venv-model/bin/adaptive-vlm distributed-infer \
  --config configs/experiments/chunk5_local_distributed.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --vision-device cuda:0 \
  --image tests/fixtures/moondream_smoke.ppm \
  --prompt "Describe this image." \
  --output-dir artifacts/distributed
```

The default `--pipeline synchronous` sends the projected BF16 tensor after
vision finishes. Use `--pipeline prompt_first` to send the prompt while vision
is running; the server prepares token IDs and joins them with the tensor by
request ID. The server bounds and expires unfinished prompt state. Each result
records the selected pipeline, exact wire bytes, component times, and memory
measurements. These two pipelines are variants of the distributed policy, not
additional execution configurations.

Run the localhost correctness and transport stability gates with:

```bash
.venv-model/bin/python scripts/verify_distributed_equivalence.py \
  --config configs/experiments/chunk5_local_distributed.yaml \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline \
  --vision-device cuda:0 --text-device cpu \
  --image tests/fixtures/moondream_smoke.ppm \
  --output artifacts/chunk5/distributed_variants_equivalence.json

.venv-model/bin/python scripts/verify_distributed_stability.py \
  --config configs/experiments/chunk5_local_distributed.yaml \
  --image tests/fixtures/moondream_smoke.ppm \
  --requests 100 \
  --output artifacts/chunk5/distributed_stability.json

.venv-model/bin/python scripts/verify_distributed_stability.py \
  --config configs/experiments/chunk5_local_distributed.yaml \
  --image tests/fixtures/moondream_smoke.ppm \
  --pipeline prompt_first --requests 100 \
  --output artifacts/chunk5/distributed_overlap_stability.json
```

The pinned loader currently constructs the whole checkpoint before discarding
the unneeded modules. Its steady-state placement is split, but load-time RAM
is still that of the full checkpoint. The localhost configuration validates
behavior and is not a physical network measurement.

## Unified benchmark and correctness runs

Run all three modes with the same resolved `chunk6_local_campaign.yaml` config
and frozen request manifest. Use separate run directories. A model server must
already be running for each remote mode; start `cloud-server` with
`chunk4_local_cloud.yaml` for `cloud`, or `distributed-server` with
`chunk5_local_distributed.yaml` for `distributed`. Stop one server before
starting the other on port 5050. Their model contract must match the benchmark
config; the benchmark performs a health/fingerprint check.

```bash
.venv-model/bin/adaptive-vlm benchmark \
  --config configs/experiments/chunk6_local_campaign.yaml \
  --mode edge --run-dir artifacts/chunk6/correctness-edge \
  --artifact-manifest configs/model/moondream2_artifact.yaml \
  --cache-dir .model-cache --offline --vision-device cpu --text-device cpu \
  --warmups 1 --repetitions 1

# Change --mode and --run-dir for cloud or distributed. The distributed client
# also needs --artifact-manifest and --vision-device; cloud needs neither.
.venv-model/bin/adaptive-vlm compare-runs \
  artifacts/chunk6/correctness-edge \
  artifacts/chunk6/correctness-cloud \
  artifacts/chunk6/correctness-distributed \
  --output artifacts/chunk6/correctness-comparison.json
```

The command prints every trial as it runs. An interrupted run can be continued
with identical arguments plus `--resume`; the plan and hashes must still match.
Each run stores `plan.json`, `run_manifest.json`, fsynced `requests.jsonl`, and
`summary.json`. Warm-ups are recorded but excluded from measured summaries;
failed/OOM attempts remain in the denominator. `compare-runs` checks every
measured request/repetition pair and lists exact token mismatches. Default
config settings are 10 warm-ups and 30 repetitions; the one-repetition command
above is a correctness smoke, not a latency study.

The checked-in correctness and 100-request manifests contain tiny synthetic
images. Their exact-match score is a fixture diagnostic, not public VQA
accuracy. If official VQA v2 val2014 JSON files and COCO val2014 images are
available locally, freeze a deterministic subset without copying images:

```bash
.venv/bin/adaptive-vlm import-vqa-v2 \
  --questions data/raw/vqa/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations data/raw/vqa/v2_mscoco_val2014_annotations.json \
  --images-dir data/raw/coco/val2014 \
  --output data/manifests/vqa_val2014_seed42.jsonl \
  --count 100 --seed 42 --revision vqa-v2-val2014-local
```

Set the experiment `dataset.manifest_path` and `dataset.revision` to that
manifest before running it. The importer expects the
[official VQA v2](https://visualqa.org/download.html)
question/annotation schema and ten human answers. The runner's VQA consensus
score is provisional because it does not implement the official answer
normalization; do not report it as official VQA accuracy. The official
[evaluator](https://visualqa.org/evaluation.html) remains necessary for
thesis-quality public-dataset results.

## Quality checks

```bash
ruff check .
mypy
pytest --cov=adaptive_vlm --cov-report=term-missing
```

CI runs only these CPU-safe checks and never downloads model weights.
