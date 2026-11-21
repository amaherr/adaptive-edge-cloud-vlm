# Documentation

Start with the [project README](../README.md) for the overview and the key results. This folder documents the system and the experimental approach: the protocol, the specifications, the design of each component, the usage guide and the runbooks. The measured evidence is in [`results/`](../results/), one folder per campaign or analysis, each with its own README and checksums.

## Gates

| Gate | Result | Read |
| --- | --- | --- |
| Physical G2 baseline: 8,100 requests over nine network blocks on the laptop/PC pair | Valid | [Results](../results/physical-g2-20261110/README.md) |
| Predictor V2 r1 on the physical data | Failed G3: cloud error tails and edge memory coverage | [Validation report](predictor-v2/validation-report.md) |
| **G2 v2+** on AWS: five configurations, edge and cloud load, 43,200 requests in 196 sub-blocks | **Passed**: every sub-block valid, quality identical across configurations; the cloud wins 48 of 54 training cells and the edge 6; split inference never wins | [Campaign](../results/aws-campaign-20261119/README.md), [baseline report](../results/aws-campaign-20261119/baseline/README.md), [derived checks](../results/aws-campaign-20261119/derived/README.md) |
| **G3**: Predictor V2 r2 | **Passed**: validation P90 APE 1.4–6.0% per configuration, 98.3% top-1 agreement | [r2 record](../results/aws-campaign-20261119/predictor_r2/README.md), [specification](predictor-v2/r2-specification.md) |
| **G4**: selector against the measured oracle on unseen networks | **Passed**: median relative regret 0.00%, 42.6 ms faster than always-cloud | [G4 record](../results/aws-campaign-20261119/g4/README.md), [specification](predictor-v2/g4-specification.md) |
| **G5**: the live adaptive policy on dynamic traces | **Passed**: 118 ms (11.5%) faster than always-cloud, 2.2% median regret, quality unchanged | [G5 record](../results/g5-aws-20261120/README.md), [adaptive runtime](design/adaptive-runtime.md) |

The [final analysis](../results/final-analysis-20261121/README.md) collects the thesis tables and figures; the [archive record](../results/archive-20261121/README.md) inventories the raw data.

## Start here

- [Usage guide](usage.md): installation, every execution path, the CLI reference and running on two machines.
- [Reproducing the results](reproducibility.md): the commands that regenerate every thesis table and figure from the archived data.

## Protocol and specifications

These files are frozen: their SHA-256 hashes are pinned by the campaign grids, the freeze records or the git tags, so they are never edited. Their links to planning documents that were later removed point to the [last versions](#removed-documents).

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
- Local prototype (single machine, three configurations): [component predictor format](design/predictor-v1-format.md) (schema 1), [adaptive execution](design/adaptive-execution.md), [live monitor and trace replay](design/live-monitor-and-trace-replay.md), [paired evaluation protocol](design/evaluation-protocol.md).

## Runbooks

- [AWS campaign](runbooks/aws-campaign.md): design decisions, infrastructure, image, references, settings, pilot, sessions, recovery, archive and teardown, for G2 v2+ and G5.
- [Physical campaign](runbooks/physical-campaign.md): the unattended physical G2 campaign on the laptop/PC pair.

## Physical testbed

- [Hosts and source baseline](physical-testbed/hosts-and-baseline.md): the edge laptop and cloud PC, the first pair verification and the frozen baseline tag.
- [GPU placement and memory](physical-testbed/gpu-placement-and-memory.md): stage placement, measured memory and edge swap residency.
- [Correctness criteria](physical-testbed/correctness-criteria.md): strict contract v1, CPU/GPU divergence, the heterogeneous-GPU criterion also used on AWS.
- [Network measurement](physical-testbed/network-measurement.md): shaping, independent iperf3/RTT verification, the nine-block grid.
- [Frozen VQA dataset](physical-testbed/vqa-dataset.md): the 100-question VQA v2 subset and its split.
- [Profiling protocol](physical-testbed/profiling-protocol.md): fixed-policy profiling blocks and the elapsed-time clock.

## Predictor V2 r1 (physical data)

- [Specification](predictor-v2/specification.md): the r1 predictor as executed.
- [Validation report](predictor-v2/validation-report.md): the r1 fit, the single sealed validation pass and the failed G3 decision.
- [Training-only stage diagnosis](../results/physical-g2-20261110/predictor_v2/r1/stage_diagnosis/README.md).

## Results

| Folder | Content |
| --- | --- |
| [physical-g2-20261110](../results/physical-g2-20261110/README.md) | Physical G2: 8,100 requests, official VQA scoring, stage analysis, Predictor V2 r1 |
| [prompt-calibration-20261115](../results/prompt-calibration-20261115/README.md) | Choice of the short-answer prompt P1 |
| [edge-load-calibration-20261116](../results/edge-load-calibration-20261116/README.md) | Edge-load levels on the physical laptop |
| [g2v2-local-rehearsal-20261116](../results/g2v2-local-rehearsal-20261116/README.md) | Fault-injection rehearsal of the campaign supervisor |
| [aws-phase-e-20261118](../results/aws-phase-e-20261118/README.md) | AWS calibration: residency, loads, int8 quality, winner estimate |
| [aws-pilot-20261118](../results/aws-pilot-20261118/README.md) | G2 v2+ pilot and go decision |
| [aws-campaign-20261119](../results/aws-campaign-20261119/README.md) | G2 v2+ campaign, baseline report, derived checks, G3 (r2) and G4 |
| [g5-rehearsal-20261120](../results/g5-rehearsal-20261120/README.md) | Loopback rehearsal of the G5 runtime |
| [g5-aws-20261120](../results/g5-aws-20261120/README.md) | G5: freeze record, pilots and campaign analysis |
| [final-analysis-20261121](../results/final-analysis-20261121/README.md) | Thesis tables and figures |
| [archive-20261121](../results/archive-20261121/README.md) | Archive inventory, restore test and source tags |

## Removed documents

Plans, roadmaps and phase reports were removed so that this folder documents only the current system and approach. Sealed records may still cite them; their last versions are at commit `8e322f3`:

| Document | Content |
| --- | --- |
| [`roadmap.md`](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/8e322f3/docs/roadmap.md) | Thesis roadmap, phases 0–9 (earlier name `thesis-completion-roadmap.md`) |
| [`g2-v2-plus-plan.md`](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/8e322f3/docs/g2-v2-plus-plan.md) | G2 v2+ plan; cited by the frozen protocol |
| [`g2-v2-plus-aws-roadmap.md`](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/8e322f3/docs/g2-v2-plus-aws-roadmap.md) | The move to AWS, phases A–F; cited by the frozen protocol |
| [`history/physical-validation-roadmap.md`](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/8e322f3/docs/history/physical-validation-roadmap.md) | The original physical validation roadmap (earlier name `roadmap.md`) |
| [`history/g2-v2-physical-preparation.md`](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/8e322f3/docs/history/g2-v2-physical-preparation.md) | Preparation, runbook and checklist of the never-run physical G2 v2 |
| [`history/local-prototype-validation.md`](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/8e322f3/docs/history/local-prototype-validation.md) | Local prototype phase reports (earlier `phase0_handoff.md` … `phase7_handoff.md`) |
| [`cloud/*.md`](https://github.com/amaherr/adaptive-edge-cloud-vlm/tree/7f511fb/docs/cloud) | The abandoned `cloud-vqa-v1` deployment; the container contracts are merged into [design/container-runtime.md](design/container-runtime.md) |
| [`predictor-v2/specification.md`, original part](https://github.com/amaherr/adaptive-edge-cloud-vlm/blob/8e322f3/docs/predictor-v2/specification.md#original-execution-specification-as-received) | The r1 execution plan as received |

Older renames, from the October reorganisation, map flat file names to their topic folders: `physical_*` → [physical-testbed/](physical-testbed/hosts-and-baseline.md) and [runbooks/physical-campaign.md](runbooks/physical-campaign.md); `phase2_split_feasibility` → [split-boundary](design/split-boundary.md); `phase2_profile_schema` → [profile-format](design/profile-format.md); `phase3_predictor_schema` → [predictor-v1-format](design/predictor-v1-format.md); `phase4_selector_contract` → [selector-and-oracle](design/selector-and-oracle.md); `phase5_execution_contract` → [adaptive-execution](design/adaptive-execution.md); `phase6_monitor_contract` → [live-monitor-and-trace-replay](design/live-monitor-and-trace-replay.md); `phase7_evaluation_protocol` → [evaluation-protocol](design/evaluation-protocol.md); `phase3_physical_predictor_v2`, `Predictor_V2_Agent_Execution_Plan` → [predictor-v2/specification](predictor-v2/specification.md); `phase3_physical_g3_handoff` → [validation-report](predictor-v2/validation-report.md).
