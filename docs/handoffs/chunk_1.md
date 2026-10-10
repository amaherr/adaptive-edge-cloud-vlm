# Chunk 1 handoff

## Status

Chunk 1 — repository foundation and experiment contracts — is complete in the
working tree. No commit has been created.

The implementation was exercised with Python 3.12.3 because Python 3.11 is not
installed in the current WSL environment. The package targets Python 3.11 or
newer, and CI is configured to verify Python 3.11.

## Worktree state

- base commit: `1f0ebea` (`Initial commit`);
- baseline planning changes and all Chunk 1 changes remain uncommitted;
- the local `.venv` and generated `artifacts/` paths are ignored by Git.

## Commands run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest --cov=adaptive_vlm --cov-report=term-missing
.venv/bin/adaptive-vlm --help
.venv/bin/adaptive-vlm config validate configs/experiments/baseline_edge.yaml
.venv/bin/adaptive-vlm config validate configs/experiments/baseline_cloud.yaml
.venv/bin/adaptive-vlm config validate configs/experiments/baseline_distributed.yaml
.venv/bin/adaptive-vlm fake-infer \
  --config configs/experiments/chunk1_smoke.yaml \
  --image tests/fixtures/smoke_image.txt \
  --prompt "Describe this fixture." \
  --output-dir /tmp/adaptive-vlm-smoke
.venv/bin/adaptive-vlm environment --workdir . \
  --output /tmp/adaptive-vlm-environment-final.json
bash -n scripts/collect_environment.sh
git diff --check
```

## Verification results

- Ruff: passed;
- strict mypy: passed for 14 source files;
- pytest: 27 passed;
- coverage: 82.89%, above the enforced 80% threshold;
- all four example experiment configurations: valid;
- editable package installation and console entry point: passed;
- fake inference and structured artifact writing: passed;
- environment inventory: passed and reported the WSL NVML failure accurately;
- shell syntax and Git whitespace checks: passed.

## Artifacts produced

- installable `adaptive_vlm` package and CLI;
- strict YAML configuration loader with resolved paths and unknown-field checks;
- baseline edge, cloud, and distributed configurations;
- declared RTX 3050 edge record and explicitly pending cloud record;
- request, result, per-request record, and run-manifest contracts;
- atomic JSON and JSONL artifact helpers;
- deterministic seed handling and JSON structured logging;
- dependency-free deterministic fake model and local execution policy;
- non-secret environment and Git-state collector;
- CPU-only unit tests and GitHub Actions workflow;
- sample manifest/request artifacts and development documentation.

## Decisions made

- keep Chunk 1 independent of PyTorch and model downloads;
- use standard-library dataclasses and `argparse`, with PyYAML as the only
  runtime dependency;
- reject moving model revisions such as `main`, `master`, and `latest`;
- distinguish `candidate`, `approved`, and `test` model states;
- distinguish `pending`, `declared`, and `verified` hardware states;
- keep real and fake model configurations separate so smoke results cannot be
  mistaken for VLM results;
- store generated experiment data under the Git-ignored `artifacts/` path;
- defer the full dependency lock until the real model stack is selected.

## Known limitations

- WSL cannot currently initialize NVML, although Windows detects the RTX 3050;
- Python 3.11 was not available for a local run; CI is the current 3.11 gate;
- the cloud machine is unknown and intentionally marked `pending`;
- the Moondream2 revision is a candidate and has not been downloaded, hashed,
  reviewed, or executed;
- the dataset manifest is intentionally pending;
- no real edge, cloud, distributed, networking, or GPU inference exists yet;
- the 4 GiB GPU feasibility question remains open for Chunk 2.

## Next entry point

Begin Chunk 2 by diagnosing WSL GPU passthrough with read-only checks, then
verify CUDA from PyTorch in the dedicated model environment. Only after that,
inspect and hash the pinned Moondream2 candidate and attempt deterministic
monolithic inference. Do not add network code in Chunk 2.

