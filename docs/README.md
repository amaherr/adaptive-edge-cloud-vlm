# Documentation

Start with the [project README](../README.md) for the overview and the key results. This folder holds the protocol, the specifications, the design of each component, the runbooks and the history of the project. The measured evidence is in [`results/`](../results/), one folder per campaign or analysis, each with its own README and checksums.

## Status

| Milestone | Status | Read |
| --- | --- | --- |
| Local CPU/loopback prototype: three configurations, selector, monitor, evaluation pipeline | Done (October 2026); the local predictor missed its ranking target | [Local prototype validation](history/local-prototype-validation.md) |
| Physical pair: hosts, correctness, network grid | Done | [Physical testbed](#physical-testbed) |
| Physical G2 baseline: 8,100 requests over nine network blocks | Done (10–11 November) | [Results](../results/physical-g2-20261110/README.md) |
| Predictor V2 r1 (G3 on the physical data) | **Failed**: cloud error tails and edge memory coverage | [Validation report](predictor-v2/validation-report.md) |
| Physical G2 v2 baseline | Prepared, **never run**: superseded by G2 v2+ on AWS | [History](history/g2-v2-physical-preparation.md) |
| **G2 v2+ on AWS**: five configurations, edge and cloud load, 43,200 requests in 196 sub-blocks | **Passed** (19 November): every sub-block valid, quality identical across configurations. On the training cells the cloud wins 48 of 54 and the edge 6; split inference never wins | [Campaign](../results/aws-campaign-20261119/README.md), [baseline report](../results/aws-campaign-20261119/baseline/README.md), [derived checks](../results/aws-campaign-20261119/derived/README.md) |
| **G3**: Predictor V2 r2 | **Passed** (20 November): validation P90 APE 1.4–6.0% per configuration, 98.3% top-1 agreement | [r2 record](../results/aws-campaign-20261119/predictor_r2/README.md), [specification](predictor-v2/r2-specification.md) |
| **G4**: selector against the measured oracle, unseen networks | **Passed** (20 November): median relative regret 0.00%, 42.6 ms faster than always-cloud | [G4 record](../results/aws-campaign-20261119/g4/README.md), [specification](predictor-v2/g4-specification.md) |
| **G5**: the live adaptive policy on dynamic traces | **Passed** (21 November): 118 ms (11.5%) faster than always-cloud, 2.2% median regret, quality unchanged | [G5 record](../results/g5-aws-20261120/README.md), [adaptive runtime](design/adaptive-runtime.md) |
| Final analysis | Done (21 November): thesis tables and figures | [Final analysis](../results/final-analysis-20261121/README.md) |
| Archive and teardown | Done (21 November); the off-machine copy is manual | [Archive record](../results/archive-20261121/README.md), [reproducibility](reproducibility.md) |
| Thesis writing | Open | [Roadmap, phase 9](roadmap.md#phase-9-thesis-writing-in-parallel-from-phase-2) |

## Start here

- [Usage guide](usage.md): installation, every execution path, the CLI reference and running on two machines.
- [Reproducing the results](reproducibility.md): the commands that regenerate every thesis table and figure from the archived data.
- [Roadmap](roadmap.md): the thesis phases 0–9 from the end of the G2 v2+ campaign, with what each delivered.

## Protocol and specifications

These files are frozen: their SHA-256 hashes are pinned by the campaign grids, the freeze records or the git tags, so they are never edited.

- [G2 v2 protocol](protocols/g2-v2-protocol.md): the experimental design (prompt, dataset, splits, network grid, loads, validity gates, statistics), with amendments A1, A2a and A2b inside it.
- Amendments, in order: [A3](protocols/g2-v2-amendment-a3.md) (exposure of the validation and evaluation cells), [A4](protocols/g2-v2-amendment-a4.md) (G5: live adaptive policy, traces, gate), [A4a](protocols/g2-v2-amendment-a4a.md) (edge-probe shape), [A4b](protocols/g2-v2-amendment-a4b.md) (background cloud sensing).
- [Predictor V2 r2 specification](predictor-v2/r2-specification.md): the G3 models, fitting, coverage and gate, written before fitting.
- [G4 specification](predictor-v2/g4-specification.md): the selector evaluation, written before the evaluation cells were opened.

## Design

- [Adaptive runtime](design/adaptive-runtime.md): the five configurations, live sensing, the r2 predictor and the selector as one per-request policy (G5).
- [Selector and measured oracle](design/selector-and-oracle.md): selection rules, exclusions, oracle and regret.
- [Container runtime contract](design/container-runtime.md): the image, the services and their permissions on AWS.
- [Split boundary](design/split-boundary.md): where the vision–language split is taken in the pinned model.
- [Profile format](design/profile-format.md): timing boundaries and field definitions of the profile rows.
- Local prototype: [component predictor format](design/predictor-v1-format.md) (schema 1), [adaptive execution](design/adaptive-execution.md), [live monitor and trace replay](design/live-monitor-and-trace-replay.md), [paired evaluation protocol](design/evaluation-protocol.md).

## Runbooks

- [AWS campaign](runbooks/aws-campaign.md): infrastructure, image, references, settings, pilot, sessions, recovery, archive and teardown, for G2 v2+ and G5.
- [Physical campaign](runbooks/physical-campaign.md): the unattended physical G2 campaign on the laptop/PC pair.

## Physical testbed

- [Hosts and source baseline](physical-testbed/hosts-and-baseline.md): the edge laptop and cloud PC, the first pair verification and the frozen baseline tag.
- [GPU placement and memory](physical-testbed/gpu-placement-and-memory.md): stage placement, measured memory and edge swap residency.
- [Correctness criteria](physical-testbed/correctness-criteria.md): strict contract v1, CPU/GPU divergence, the approved heterogeneous-GPU criterion.
- [Network measurement](physical-testbed/network-measurement.md): shaping, independent iperf3/RTT verification, the nine-block grid.
- [Frozen VQA dataset](physical-testbed/vqa-dataset.md): the 100-question VQA v2 subset and its split.
- [Profiling protocol](physical-testbed/profiling-protocol.md): fixed-policy profiling blocks and the elapsed-time clock.

## Predictor V2 r1 (physical data)

- [Specification](predictor-v2/specification.md): the physical-scope revision as executed, followed by the original specification.
- [Validation report](predictor-v2/validation-report.md): the r1 fit, the single sealed validation pass and the failed G3 decision.
- [Training-only stage diagnosis](../results/physical-g2-20261110/predictor_v2/r1/stage_diagnosis/README.md).

## Campaign planning records

Both are linked from the frozen protocol as its rationale, so they keep their paths and wording.

- [G2 v2+ plan](g2-v2-plus-plan.md): adding the prompt-first and int8 configurations and cloud load, so one campaign covers the full aim.
- [G2 v2+ AWS roadmap](g2-v2-plus-aws-roadmap.md): the move to two AWS VMs, decisions D1–D11 and phases A–F with their results.

## Results

| Folder | Content |
| --- | --- |
| [physical-g2-20261110](../results/physical-g2-20261110/README.md) | Physical G2: 8,100 requests, official VQA scoring, stage analysis, Predictor V2 r1 |
| [prompt-calibration-20261115](../results/prompt-calibration-20261115/README.md) | Choice of the short-answer prompt P1 |
| [edge-load-calibration-20261116](../results/edge-load-calibration-20261116/README.md) | Edge-load levels on the physical laptop |
| [g2v2-local-rehearsal-20261116](../results/g2v2-local-rehearsal-20261116/README.md) | Fault-injection rehearsal of the G2 v2 supervisor |
| [aws-phase-e-20261118](../results/aws-phase-e-20261118/README.md) | AWS recalibration: residency, loads, int8 quality, winner estimate |
| [aws-pilot-20261118](../results/aws-pilot-20261118/README.md) | G2 v2+ pilot and go decision |
| [aws-campaign-20261119](../results/aws-campaign-20261119/README.md) | G2 v2+ campaign, baseline report, derived checks, G3 (r2) and G4 |
| [g5-rehearsal-20261120](../results/g5-rehearsal-20261120/README.md) | Loopback rehearsal of the G5 runtime |
| [g5-aws-20261120](../results/g5-aws-20261120/README.md) | G5: freeze record, pilots and campaign analysis |
| [final-analysis-20261121](../results/final-analysis-20261121/README.md) | Thesis tables and figures |
| [archive-20261121](../results/archive-20261121/README.md) | Archive inventory, restore test and source tags |

## History

- [Local prototype validation](history/local-prototype-validation.md): phases 0–7 on one machine (CPU and loopback), October 2026.
- [Physical validation roadmap](history/physical-validation-roadmap.md): the original plan for the physical pair, phases A–AR, with the gate definitions.
- [G2 v2 on the physical pair](history/g2-v2-physical-preparation.md): preparation, runbook and checklist for a campaign that moved to AWS instead.

## Renamed and removed documents

Sealed result files keep their original links and commands, and every record pins the exact source commit used. Use this table to find a document's current location. Code moved in the same reorganisations: predictors are under `src/adaptive_vlm/prediction/`, offline analyses under `src/adaptive_vlm/evaluation/`, scripts under topic folders in `scripts/`, and `phase`/`chunk` configs were renamed (for example `chunk6_local_campaign.yaml` → `local_campaign.yaml`).

| Former document | Now |
| --- | --- |
| `thesis-completion-roadmap.md` | [roadmap.md](roadmap.md) |
| `roadmap.md` (until 21 November 2026), `physical_adaptive_vlm_validation_roadmap.md` | [history/physical-validation-roadmap.md](history/physical-validation-roadmap.md) |
| `g2-v2-preparation.md`, `runbooks/g2v2-onsite-checklist.md`, the G2 v2 section of `runbooks/physical-campaign.md` | [history/g2-v2-physical-preparation.md](history/g2-v2-physical-preparation.md) |
| `cloud/container_runtime_contract_v1.md`, `cloud/container_runtime_contract_v2.md` | [design/container-runtime.md](design/container-runtime.md) |
| `cloud/cloud_acceptance_v1.md`, `cloud/cloud_cost_review_v1.md`, `cloud/cloud_environment_handoff_v1.md`, `cloud/cloud_execution_2026-11-05.md` (removed: records of the abandoned `cloud-vqa-v1` deployment of 3–5 November; the current procedure is the [AWS runbook](runbooks/aws-campaign.md)) | last versions: [acceptance](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/7f511fb/docs/cloud/cloud_acceptance_v1.md), [cost review](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/7f511fb/docs/cloud/cloud_cost_review_v1.md), [environment handoff](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/7f511fb/docs/cloud/cloud_environment_handoff_v1.md), [execution log](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/7f511fb/docs/cloud/cloud_execution_2026-11-05.md) |
| Setup, usage and CLI sections of the root README | [usage.md](usage.md) |
| `physical_unattended_campaign.md` | [runbooks/physical-campaign.md](runbooks/physical-campaign.md) |
| `physical_pair_2026-11-02.md`, `physical_experiment_baseline_v1.md`, `phase_a2_handoff.md` | [physical-testbed/hosts-and-baseline.md](physical-testbed/hosts-and-baseline.md) |
| `physical_edge_placement_v1.md`, `physical_memory_feasibility_v1.md`, `physical_cloud_memory_v1.md` | [physical-testbed/gpu-placement-and-memory.md](physical-testbed/gpu-placement-and-memory.md) |
| `physical_correctness_contract_v1.md`, `physical_cpu_gpu_divergence_v1.md`, `physical_g2_heterogeneous_correctness_v2.md` | [physical-testbed/correctness-criteria.md](physical-testbed/correctness-criteria.md) |
| `physical_network_measurement_v1.md` | [physical-testbed/network-measurement.md](physical-testbed/network-measurement.md) |
| `physical_dataset_protocol_v1.md` | [physical-testbed/vqa-dataset.md](physical-testbed/vqa-dataset.md) |
| `physical_profile_campaign_v1.md`, `timing_clock_v1.md` | [physical-testbed/profiling-protocol.md](physical-testbed/profiling-protocol.md) |
| `phase3_physical_predictor_v2.md`, `Predictor_V2_Agent_Execution_Plan.md` | [predictor-v2/specification.md](predictor-v2/specification.md) |
| `phase3_physical_g3_handoff.md` | [predictor-v2/validation-report.md](predictor-v2/validation-report.md) |
| `phase2_split_feasibility.md` | [design/split-boundary.md](design/split-boundary.md) |
| `phase2_profile_schema.md` | [design/profile-format.md](design/profile-format.md) |
| `phase3_predictor_schema.md` | [design/predictor-v1-format.md](design/predictor-v1-format.md) |
| `phase4_selector_contract.md` | [design/selector-and-oracle.md](design/selector-and-oracle.md) |
| `phase5_execution_contract.md` | [design/adaptive-execution.md](design/adaptive-execution.md) |
| `phase6_monitor_contract.md` | [design/live-monitor-and-trace-replay.md](design/live-monitor-and-trace-replay.md) |
| `phase7_evaluation_protocol.md` | [design/evaluation-protocol.md](design/evaluation-protocol.md) |
| `phase0_handoff.md` … `phase7_handoff.md` | [history/local-prototype-validation.md](history/local-prototype-validation.md) |
| `dynamic_scheduler_implementation_plan.md` (removed: superseded by the roadmap) | [last version](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/dynamic_scheduler_implementation_plan.md) |
| `Gate2_Cloud_Migration_Roadmap_v2.md` (removed: superseded by the G2 v2+ AWS roadmap) | [last version](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/Gate2_Cloud_Migration_Roadmap_v2.md) |
| `cli_refactor_handoff.md`, `phase_a1_handoff.md` (removed: past engineering notes) | [CLI note](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/cli_refactor_handoff.md), [A1 note](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/phase_a1_handoff.md) |
