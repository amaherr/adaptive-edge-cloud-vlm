# Documentation

Start with the [project README](../README.md) for setup and usage. This folder holds the plan, the physical testbed records, design specifications, and the archived local prototype evidence.

## Where things stand

| Milestone | Status | Read |
| --- | --- | --- |
| Local CPU/loopback prototype (all three placements, selector, monitor, evaluation pipeline) | Done; local predictor ranking gate failed | [Local prototype validation](history/local-prototype-validation.md) |
| Physical pair setup, correctness and network grid | Done | [Physical testbed](#physical-testbed) |
| Physical G2 baseline: 8,100 requests over nine network blocks | Done | [G2 results](../results/physical-g2-20261110/README.md) |
| Predictor V2 r1 (G3) | **Failed**: cloud error tails and edge memory coverage | [Validation report](predictor-v2/validation-report.md) |
| Edge full-model memory | Fixed by swap residency, not yet used by the campaign scripts | [GPU placement and memory](physical-testbed/gpu-placement-and-memory.md#edge-full-model-swap-residency-15-november-2026) |
| G2 v2 baseline (swap-residency edge, short-answer prompt, interleaved placements, edge-load factor) | Preparing off site | [G2 v2 preparation](g2-v2-preparation.md) |
| Selector vs measured oracle (G4) and later phases | Not started; builds on G2 v2 | [Roadmap](roadmap.md) |

## Plan and runbook

- [Roadmap](roadmap.md): the physical validation plan from repository cleanup to the final thesis matrix, with every gate.
- [G2 v2 preparation](g2-v2-preparation.md): everything to finish off site and on site before the new unattended baseline run.
- [G2 v2 protocol](protocols/g2-v2-protocol.md): the baseline's design; the prompt selection rule and dataset construction are frozen, and the rest is a draft until the pilot.
- [Physical campaign runbook](runbooks/physical-campaign.md): launching, monitoring, stopping and resuming the unattended edge/cloud campaign.

## Physical testbed

- [Hosts and source baseline](physical-testbed/hosts-and-baseline.md): edge laptop and cloud PC, first pair verification, frozen baseline tag.
- [GPU placement and memory](physical-testbed/gpu-placement-and-memory.md): stage placement, measured memory, and edge swap residency.
- [Correctness criteria](physical-testbed/correctness-criteria.md): strict contract v1, CPU/GPU divergence, approved heterogeneous-GPU criterion.
- [Network measurement](physical-testbed/network-measurement.md): shaping, independent iperf3/RTT verification, the nine-block grid.
- [Frozen VQA dataset](physical-testbed/vqa-dataset.md): the 100-question VQA v2 subset and its split.
- [Profiling protocol](physical-testbed/profiling-protocol.md): fixed-policy profiling blocks and the elapsed-time clock.

## Predictor V2

- [Specification](predictor-v2/specification.md): the physical-scope revision as executed, followed by the original specification.
- [Validation report](predictor-v2/validation-report.md): the r1 fit, single sealed validation pass and failed G3 decision.
- Training-only stage diagnosis: [results/…/predictor_v2/r1/stage_diagnosis](../results/physical-g2-20261110/predictor_v2/r1/stage_diagnosis/README.md).

## Design specifications

- [Split boundary](design/split-boundary.md): where the vision/language split is taken in the pinned model.
- [Profile format](design/profile-format.md): `profile-runs` output, schema version 1.
- [Component predictor format](design/predictor-v1-format.md): schema-1 calibration and prediction.
- [Selector and offline oracle](design/selector-and-oracle.md): selection rules, exclusions, oracle regret.
- [Adaptive execution](design/adaptive-execution.md): `adaptive-benchmark` inputs, preflight and records.
- [Live monitor and trace replay](design/live-monitor-and-trace-replay.md): bounded probes and synthetic network traces.
- [Paired evaluation protocol](design/evaluation-protocol.md): split-safe paired analysis.

## Renamed and removed documents

Sealed result files keep their original links and commands, and receipts pin the exact source commit used. Use this table to find the current location. Code paths moved in the same reorganization: predictors are now under `src/adaptive_vlm/prediction/`, offline analyses under `src/adaptive_vlm/evaluation/`, scripts under topic folders in `scripts/`, and `phase`/`chunk` configs were renamed (for example `chunk6_local_campaign.yaml` → `local_campaign.yaml`).

| Former document | Now |
| --- | --- |
| `physical_adaptive_vlm_validation_roadmap.md` | [roadmap.md](roadmap.md) |
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
| `Gate2_Cloud_Migration_Roadmap_v2.md` (removed: AWS/Vultr out of scope; see the `cloud-infrastructure` branch) | [last version](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/Gate2_Cloud_Migration_Roadmap_v2.md) |
| `cli_refactor_handoff.md`, `phase_a1_handoff.md` (removed: past engineering notes) | [CLI note](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/cli_refactor_handoff.md), [A1 note](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/035e9f0/docs/phase_a1_handoff.md) |
