# Phase A1 handoff — coverage gate and first CLI extraction

Date: 2026-10-27 (Africa/Cairo). This implements only Phase A1 of the [physical validation roadmap](physical_adaptive_vlm_validation_roadmap.md), plus one bounded CLI cleanup requested alongside it. Phase A2's physical-experiment baseline freeze and the two-machine campaign have not begun.

## What changed

- Added replay-loader regression tests for the frozen Phase 5 input and invalid manifest/model/placement/task/network/health constraints. They exposed a Phase 7 regression: a mixed synthetic manifest with both labeled and unlabeled prompts was rejected even though its replay did not declare one task type. The loader now requires uniform types only when the replay explicitly declares `task_type`; VQA still requires `vqa_short_answer`. The existing replay loads again, while mismatched declared types fail closed.
- Corrected the existing Moondream adapter coverage omission to a path-independent pattern in both coverage tracing and reporting. The model adapter was intentionally excluded from CPU-only coverage before this phase, but its path still appeared in the local report because the report setting did not apply the omission. The **80% threshold is unchanged**. The new tests exercise real replay validation paths; coverage of `adaptive_runner.py` rose from 52% in the initial local report to 88%.
- Moved the adaptive benchmark's policy loading, live monitor setup, task feature extraction, registry construction, gate checks and run orchestration from `cli.py` to [`commands/adaptive.py`](../src/adaptive_vlm/commands/adaptive.py). `cli.py` retains argument parsing and a one-line route; the command receives an RSS reader so the existing inference path and test injection remain intact. A routing test checks the public command boundary. CLI size fell from 60,005 bytes / 1,470 lines to 52,838 bytes / 1,294 lines. This is the first extraction, not the full future command-package split.

## Local A1 gate result

| Check | Result |
| --- | ---: |
| Ruff | Passed |
| Strict mypy | Passed, 54 source files |
| Pytest | 185 passed |
| Branch-aware coverage | **80.54%**, threshold 80% |
| GitHub CI | Remote run pending |

The initial local run measured 75.73% because the configured model-adapter omission did not carry through to the coverage report; applying the intended omission to that same baseline measured 79.12%, consistent with the roadmap's approximately 79.15% failure. Focused replay tests and the CLI boundary test then brought the final local figure to 80.54%. The [machine-readable coverage result](../artifacts/phase_a1/2026-10-27/coverage.json) is Git-ignored. The local environment uses Python 3.12; GitHub CI is configured for Python 3.11, so a remote green run is still required before declaring the whole A1 gate closed.

## Run and inspect

From the repository root, with the development virtual environment installed:

```bash
.venv/bin/ruff check .
.venv/bin/mypy
COVERAGE_FILE=/tmp/adaptive-vlm-a1.coverage \
  .venv/bin/pytest --cov=adaptive_vlm --cov-report=term-missing
.venv/bin/adaptive-vlm adaptive-benchmark --help
```

The pytest footer reports the coverage gate and test count. The coverage JSON contains per-file line and branch measurements. `adaptive-benchmark --help` retains the same public options. To check the frozen replay without model weights, run `pytest tests/test_replay_validation.py`.

## Remaining before Phase A2

GitHub CI must run green on this code. The configured `gh` account currently reports an invalid token, so remote status could not yet be verified. Phase A2 requires a separate deliberate baseline commit/SHA after the CI result; no physical-hardware inventory, model change or network experiment was started in A1.
