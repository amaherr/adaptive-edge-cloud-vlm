# Predictor V2 — physical G2 execution revision

The user-provided [execution specification](Predictor_V2_Agent_Execution_Plan.md) is preserved as received. During the implementation audit the user explicitly changed its environment scope: **exclude AWS and Vultr; use all three configurations of the completed physical G2 campaign**. This document records that override; cloud-VM identity requirements and example cloud paths in the original specification do not apply to this revision.

The measured pair is the RTX 3050 laptop (4096 MiB dedicated VRAM) and RTX 2070 SUPER university PC (8192 MiB), under the existing native BF16 model/generation contract. The full-edge run used WDDM shared memory, so successful collection does not establish dedicated-VRAM feasibility. Model training here refers to fitting a latency predictor, never fine-tuning Moondream.

## Frozen data and isolation

The authoritative assignments remain `configs/evaluation/physical_vqa_split_v1.json`: 60 training requests with N1/N3/N5/N6/N8 (2700 attempts), 20 validation requests with N2/N7 (360), 20 evaluation requests with N4/N9 (360), and 4680 crossed attempts excluded. All three fixed placements and three repetitions are retained. No resampling or VQA prompt/scoring revision is authorized by this predictor task.

**Prior inspection:** the physical G2 stage analysis already examined N4/N9 outcomes before this predictor specification arrived. The V2 pipeline will prevent their use in fitting, feature scaling, candidate selection and validation, but cannot claim these are scientifically untouched measurements. No final evaluation is authorized in this training/validation phase. A genuinely unseen confirmatory test needs a later independently frozen experiment.

The new source of truth is the checksum-pinned `results/physical-g2-20261110/evidence.tar.gz`, not old CPU/loopback coefficients or AWS/Vultr results. Keep raw logs unchanged. Preserve V1 APIs and the three-configuration registry.

## Baseline checkpoint

Source starting point: `5740c3c` on main; work continues on `feature/predictor-v2-physical-g3`. The provided specification was initially untracked and is included in this checkpoint. The request manifest hash is `4e62826280e49359e8cf98d85154000c1583d63474be83a2e9a429da9bde72c9`; model contract is `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`.

Actual baseline commands and results:

- `.venv/bin/python -m pytest -q -rs --junitxml=artifacts/phase3/cloud-v2/implementation-audit/baseline-tests.xml`: **227 passed**, zero skipped, 41.10 s. This command ran before the scope override; its evidence was copied without alteration into `artifacts/phase3/physical-v2/implementation-audit/`.
- `.venv/bin/python -m ruff check . --no-cache`: passed.
- `.venv/bin/python -m mypy --cache-dir /tmp/predictor-v2-mypy`: passed, 65 source files.

The baseline audit verifies the exact 60/20/20 assignment, disjoint IDs, complete 100-ID manifest coverage, frozen network memberships, and dataset SHA. Original logs and an audit JSON remain in the ignored artifact directory above.

## Ordered implementation checkpoints

1. Specification/scope and baseline audit (this checkpoint).
2. Strict positive-allowlist partitioning, outcome/feature separation and leakage/provenance tests.
3. Versioned V2 runtime, analytical transport, low-dimensional compute fitting, training-only uncertainty and V1 compatibility.
4. Training-only grouped diagnostics, sealed validation predictions, one candidate selection and honest numerical/ranking/coverage gates.
5. Final tested handoff and hashes. No N4/N9 prediction/evaluation artifacts in this phase.

Regular commits are required after tested phases. Data-dependent gates and final calibration are **not passed or frozen yet**.

## Data-partition checkpoint

The data implementation uses metadata-only routing of shared JSONL profiles: outcome/stage values for crossed rows are skipped without decoding. It never extracts N4/N9 outcome members. Only schedule metadata from those blocks is read to verify the predeclared 8100 slots. The independent real-archive test rejects any attempt to open protected final targets.

Prepared bundle: `artifacts/phase3/physical-v2/data-r3`. It contains 2700 successful training attempts and 360 pre-execution validation snapshots; all 4680 crossed slots are excluded and all 360 final slots stay unmaterialized. Training JSONL SHA-256: `c9c4a4cbe87b4e576833f717e4aae0fd018d77cff56ce52bcee03a16b30480d0`. Validation-input SHA-256: `6d02cee6e3d1ee793c91b07a146409efe0c6de5f0deafbeeed7ff2a0a3915cda`.

Pre-execution input metadata is committed in `configs/predictor/physical_vqa_v2_input_features.json`: image byte count/header dimensions plus exact pinned Starmie tokenizer/template prompt counts. It contains no measured output or latency. The preparation script checks all source image hashes and the tokenizer/template SHA; it does not load a model or GPU.

**Network scope, declared before fitting/validation:** this is an offline block-constant netem study. Inputs use the pre-block independent iperf/echo probe only, never the post-block probe or profile's pre/post aggregate. A four-hour maximum block-probe age is an explicit engineering assumption for this offline controlled-block context, not a claim of continuously fresh goodput. Live prediction uses a separate 30-second freshness ceiling. The RTT envelope is 0–250 ms and bidirectional goodput 1–200 Mb/s; measured-support departures must be labeled physics extrapolation, with doubled empirical uncertainty and automatic extrapolation disabled. These bounds are engineering restrictions, not learned from held-out latency.

**Resource scope, declared before validation:** use candidate-specific training reserved-memory maxima with 10% headroom against dedicated budgets (4096 MiB edge, 8192 MiB cloud). Full edge already exceeds dedicated VRAM in the baseline; do not invent a safe shared-memory budget. Its diagnostic latency can be fitted but execution eligibility must remain conditional/excluded. Do not hide this candidate from coverage accounting or silently relax the G3 gate. Existing VQA correctness approval is not a new task-accuracy guarantee.

Windows DrvFS did not enforce `chmod 000` during the initial prototype. The corrected pipeline places validation targets in a private Linux `/tmp/predictor-v2-sealed-*` directory and verifies permission bits are actually zero. Its path and expected SHA are recorded in `partition_audit.json`. Training has no reader for this store. Validation may temporarily read it only after predictions are sealed and checked. It is ephemeral across WSL reboots: regenerate a fresh source-identical partition bundle if needed; never substitute a changed dataset.

Actual checkpoint checks: 12 partition/leakage tests passed (4.45 s), Ruff passed, strict mypy passed (66 source files). The real-archive integration test confirms permission-denied validation targets and no post-probe/final-outcome access. Fitting has not started and validation targets have not been scored for V2.

## Model implementation checkpoint (before fitting)

Added separate V2 classes while retaining V1 serialization and behavior. `CalibrationV2` is content-addressed with schema 2. `PredictorV2.predict` returns the existing prediction contract plus explicit execution feasibility, diagnostic-only latency, interpolation/extrapolation distance and live-selection eligibility. The selector now honors the V2 eligibility flag; old V1 predictions retain their existing behavior.

Predeclared comparisons are `constant_median`, refitted actual V1, and V2-A. V2-B is not implemented or a selectable candidate. V2-A uses analytical bidirectional byte transfer with nonnegative intercept/RTT sensitivity fitted to **five block-level median residuals**, and nonnegative low-dimensional computation fits to per-request median stage targets. Compute features are image pixels and prompt tokens where relevant; near-constant stages use medians. Fixed Ridge alpha is 1e-6. There is no forest, boosting model or neural network and no feature/hyperparameter tuning against validation.

Decode cost and output-length/response-byte priors come solely from successful training observations. Current measured decode/total/output bytes are not runtime inputs. Five request-grouped CV folds and five leave-training-network-block-out diagnostics precede uncertainty fitting. The 95th percentile of each request's maximum OOF absolute error defines an empirical width, doubled outside training support. It is not a calibrated probabilistic guarantee.

Known resource exclusions retain diagnostic component totals with `valid=false`; they cannot pass through the selector. Both edge and cloud reserved-memory evidence are considered. Offline/static-preflight predictions and physics extrapolation are not eligible for live automatic selection. No prediction is an execution command.

Checkpoint checks: 59 V1/V2/partition/selector compatibility tests passed (5.91 s), Ruff passed, strict mypy passed (68 source files). This includes units, monotonicity, component sums, schema reload/tamper detection, identity/freshness/availability errors, low/high RTT, both memory unknown/exceeded semantics, failure preservation, forbidden fit rows, grouped-CV outcome independence and exclusion of research predictions by the unchanged three-policy registry. No V2 validation outcome has been accessed.
