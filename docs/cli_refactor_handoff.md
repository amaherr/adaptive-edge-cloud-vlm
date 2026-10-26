# CLI command-module refactor handoff

Date: 2026-10-27 (Africa/Cairo). This refactor moves orchestration out of `src/adaptive_vlm/cli.py` while keeping its public command names, options, output formats and error handling. The code checkpoint is commit `7498fdb`.

## Module layout

| Module | Commands |
| --- | --- |
| `commands/benchmark.py` | Fixed `benchmark` execution |
| `commands/profile.py` | `profile-runs` |
| `commands/calibrate.py` | `calibrate-predictor` |
| `commands/adaptive.py` | `adaptive-infer`, `adaptive-benchmark`, RSS reader and live-monitor orchestration |
| `commands/evaluate.py` | Run comparison, selector, trace, monitor, adaptive and thesis analyses |
| `commands/vqa.py` | Import VQA, prepare official subset, export predictions |
| `commands/infer.py` | Fake, edge, cloud and distributed single-request inference |
| `commands/server.py` | Cloud, adaptive dual-path and distributed server startup |
| `commands/config.py` | Configuration validation and environment inspection |

`cli.py` now contains the `argparse` definitions, a direct route for each subcommand, the common user-facing exception boundary and the entry point. The adaptive benchmark's RSS reader remains an explicit dependency. Tests now patch implementations in their command modules, where those dependencies are resolved.

The CLI shrank from **1,470 lines / 60,005 bytes** at the end of Phase 7 (1,294 lines / 52,838 bytes after the initial A1 extraction) to **431 lines / 20,321 bytes**. The command modules hold the same execution logic grouped by workflow; the extra `infer.py`, `server.py` and `config.py` keep those distinct paths out of benchmarking and evaluation modules.

## Verify and use

From the repository root:

```bash
.venv/bin/ruff check .
.venv/bin/mypy
COVERAGE_FILE=/tmp/adaptive-vlm-cli-refactor.coverage \
  .venv/bin/pytest --cov=adaptive_vlm --cov-report=term-missing
.venv/bin/adaptive-vlm --help
.venv/bin/adaptive-vlm adaptive-benchmark --help
```

The full local gate passed: **188 tests**, **81.56% branch-aware coverage** against the unchanged 80% threshold, Ruff and strict mypy. Public CLI smoke tests run a real fake-model fixed benchmark, the synthetic network trace, and VQA import/subset preparation. Existing loopback, inference, server and adaptive tests also pass. No model weights or second machine are needed for these checks.

## Physical baseline boundary

The previously pushed `physical-experiment-baseline-v1` tag still points to `867f11d68882faae63b83a823489f36d5d57e519`; this refactor is **after** that tag. To run the first physical campaign on the exact A2 baseline, check out the tag as instructed in the [A2 handoff](phase_a2_handoff.md). Using this refactored `main` branch for physical experiments would be a new source revision and should get its own frozen, CI-green baseline before comparing results. Do not move or overwrite the v1 tag.
