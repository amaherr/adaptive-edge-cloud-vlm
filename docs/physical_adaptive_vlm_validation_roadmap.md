# Physical Adaptive VLM Validation Roadmap

**Implementation checkpoint (2026-10-27):** Phase A1 passes with 185 tests, Ruff, mypy, and 80.54% local coverage under the unchanged 80% threshold. [GitHub CI passed](https://github.com/amaherr/adaptive-edge-cloud-vlm/actions/runs/40403349052) on the A1 head. The [A1 handoff](phase_a1_handoff.md) records the replay regression fix and the first CLI extraction. Phase A2 freezes the source in the [Physical Experiment Baseline v1](physical_experiment_baseline_v1.md) commit and tag after its own CI run passes; physical-host work has not started.

Since access to the second machine is imminent, the project should now move from **framework-development mode** to **physical experimental-validation mode**.

The key principle is:

> **Do not add more scheduler sophistication until the current three-configuration scheduler is proven on two physical machines under controlled network conditions.**

The software pipeline is already mature enough. The next major risk is experimental validity, not missing architecture.

---

# Target End State

By the end of this roadmap, the project should support a defensible claim like:

> Under varying network and task conditions, the proposed system observes the current system/task state, predicts the latency of multiple VLM execution configurations, selects a feasible configuration dynamically, and achieves performance close to the measured oracle while outperforming appropriate fixed-placement baselines under conditions where adaptation is beneficial.

The final runtime should look approximately like:

```text
                         REQUEST
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
       Task Analyzer                 System Monitor
   image dimensions                  uplink goodput
   image bytes                       downlink goodput
   prompt tokens                     RTT
   task type                         edge memory
   output estimate                   edge/cloud load
             │                             │
             └──────────────┬──────────────┘
                            ▼
                    Performance Predictor
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
      Edge/full        Cloud/raw       Distributed
         L̂1               L̂2               L̂3
           └────────────────┼────────────────┘
                            ▼
                         Selector
                            │
                            ▼
                   Selected configuration
                            │
                            ▼
                       VLM execution
                            │
                            ▼
                  observed metrics + logs
```

---

# Phase A — Clean the Repository Before Touching the Second Machine

Do this first because the physical campaign should start from a clean, reproducible commit.

## A1. Fix the Failed CI

At the start of A1, all 171 tests passed, Ruff and mypy passed, but CI failed because coverage was **79.15%** while the threshold was 80%.

Do **not** lower the threshold.

Add tests to the lowest-coverage areas, especially:

```text
adaptive_runner.py
network_trace.py
adaptive_analysis.py
thesis_eval.py
monitor_eval.py
```

Only a small increase is needed.

### Gate A1

Require:

```text
ruff        PASS
mypy        PASS
pytest      PASS
coverage    >= 80%
GitHub CI   GREEN
```

Do not start the publication-quality physical campaign from a red commit.

---

## A2. Freeze a Physical-Experiment Starting Commit

Once CI is green:

```bash
git status
git add .
git commit -m "chore: freeze pre-physical adaptive VLM baseline"
git push
```

Record the commit SHA.

This commit becomes:

> **Physical Experiment Baseline v1**

Every future experiment should identify exactly which Git commit generated it.

---

# Phase B — Prepare the Two-Machine Environment

When the second machine is available, define fixed roles:

```text
Machine E = Edge

Machine C = Cloud
```

Do not casually switch their roles between experiments.

---

# Phase C — Hardware Inventory

Record exact hardware on both machines.

## Edge Machine

Record:

```text
CPU
RAM
GPU
GPU VRAM
CUDA version
NVIDIA driver
OS
Python version
PyTorch version
network interface
```

For example:

```text
Edge:
CPU: ...
RAM: 16 GB
GPU: RTX ....
VRAM: 8 GB
CUDA: ...
```

## Cloud Machine

Record the same information.

Also record:

```bash
nvidia-smi
python --version
pip freeze
```

and ideally:

```bash
lscpu
free -h
uname -a
```

or Windows equivalents.

### Deliverable

Create something like:

```text
artifacts/physical/environment/
    edge_environment.json
    cloud_environment.json
```

---

# Phase D — Reproduce the Model on Both Machines

Before benchmarking anything, ensure the exact same model revision works on both hosts.

Keep the current pinned Moondream configuration unchanged initially.

Verify:

```text
model ID
model revision
tokenizer
preprocessing
precision
max_new_tokens
greedy decoding
model fingerprint
```

The most important rule:

> **Do not change the model while validating the scheduler.**

Otherwise scheduler effects cannot be separated from model changes.

---

# Phase E — Establish Actual Network Connectivity

Connect:

```text
EDGE ───────── network ───────── CLOUD
```

Verify:

```text
ping
TCP connection
adaptive server health
model fingerprint
server capabilities
raw-image path
projected-feature path
```

The cloud server should simultaneously support:

```text
cloud/raw_image

distributed/projected/synchronous
```

without restarting between requests.

---

# Phase F — Re-run Physical Correctness First

Do **not** begin latency benchmarking yet.

First prove that:

```text
edge/full

cloud/raw_image

distributed/projected/synchronous
```

produce equivalent outputs on the two-machine deployment.

Use the frozen correctness manifest.

For each request compare:

```text
output token IDs
decoded answer
model fingerprint
prompt token count
generation count
```

### Required Outcome

Ideally:

```text
Edge vs Cloud           100% token agreement
Edge vs Distributed     100% token agreement
Cloud vs Distributed    100% token agreement
```

If numerical hardware differences create tiny deviations, document them rather than hiding them.

### Gate G0-physical

No performance experiment until the physical correctness gate passes.

---

# Phase G — Measure Actual Edge Memory

This is currently one of the biggest missing safety measurements.

Do not rely on process RSS.

Measure CUDA memory properly.

For each configuration record:

```text
peak allocated VRAM
peak reserved VRAM
resident memory
available VRAM before request
available VRAM after request
```

You want something like:

```text
Configuration                  Edge peak VRAM
------------------------------------------------
edge/full                      6.4 GB
cloud/raw_image                0.2 GB
distributed/projected          2.1 GB
```

Then define safe headroom.

For example:

\[
M_{required}
=
M_{peak}
	imes 1.10
\]

or another predeclared safety margin.

Then the selector can correctly evaluate:

\[
M_{required}(C)
\le
M_{available}
\]

---

# Phase H — Build the Controlled Network Environment

This is probably the most important experimental infrastructure step.

Control:

## Bandwidth

For example:

```text
1 Mbps
2 Mbps
5 Mbps
10 Mbps
20 Mbps
50 Mbps
100 Mbps
```

## RTT

For example:

```text
5 ms
20 ms
50 ms
100 ms
```

A manageable initial grid could be:

| Network ID | Bandwidth | RTT |
|---|---:|---:|
| N1 | 2 Mbps | 20 ms |
| N2 | 5 Mbps | 20 ms |
| N3 | 10 Mbps | 20 ms |
| N4 | 25 Mbps | 20 ms |
| N5 | 50 Mbps | 20 ms |
| N6 | 100 Mbps | 20 ms |
| N7 | 10 Mbps | 5 ms |
| N8 | 10 Mbps | 50 ms |
| N9 | 10 Mbps | 100 ms |

This is enough to start seeing crossovers.

---

# Phase I — Verify the Network Condition Independently

This is critical.

Do not simply configure:

```text
10 Mbps
```

and assume the connection is 10 Mbps.

Measure:

```text
actual uplink goodput
actual downlink goodput
actual RTT
```

using independent network tools.

For every network block archive:

```text
configured bandwidth
measured bandwidth
configured RTT
measured RTT
timestamp
```

Do not substitute application socket-send rate for physical-path throughput.

---

# Phase J — Freeze the Physical Task Dataset

Stop using the 4×4 synthetic fixture for performance experiments.

Keep it for correctness testing.

For real experiments create a frozen task set.

## Task Group 1 — Short-Answer VQA

Example:

```text
image
question
short answer
max_new_tokens = 32
```

## Task Group 2 — Descriptive / Caption Task

Example:

```text
image
describe the image
longer output
```

If changing the output-token contract complicates calibration, keep this as a later campaign.

For the first physical scheduler validation, **short-answer VQA alone is enough**.

---

# Phase K — Import Official VQA Data

The required tooling already exists.

Create a fixed VQA subset.

For example:

```text
100–500 questions
```

depending on runtime.

Freeze:

```text
request IDs
images
questions
annotations
seed
dataset revision
```

Do not repeatedly resample during development.

---

# Phase L — Design the Data Split Before Profiling

Do **not** profile everything and then decide which samples are test samples.

Freeze the split first.

For example:

```text
Training requests     60%
Validation requests   20%
Evaluation requests   20%
```

Also split network conditions.

Example:

```text
Training bandwidth:
2, 10, 50, 100 Mbps

Held-out network:
5, 25 Mbps
```

This allows interpolation/generalization testing.

The predictor should never train on the final evaluation conditions.

---

# Phase M — Run the Physical Phase 2 Profiling Campaign

Execute every fixed configuration:

```text
edge/full
cloud/raw_image
distributed/projected/synchronous
```

across the controlled network and task grid.

Conceptually:

\[
Configuration
	imes
Task
	imes
NetworkState
	imes
Repetition
\]

## Minimum Repetitions

For publication-quality analysis, preferably aim for:

```text
5–10 repetitions/request/network state
```

depending on total runtime.

At minimum:

```text
>= 3 distinct request IDs
>= 10 samples/configuration/bucket
```

---

# Phase N — Randomize / Interleave Configurations

Do not run:

```text
100 edge requests
then
100 cloud
then
100 split
```

because temperature/load/system drift can bias results.

Prefer:

```text
Request 1:
 edge
 cloud
 split

Request 2:
 split
 edge
 cloud

Request 3:
 cloud
 split
 edge
```

Rotate order.

Even better:

```text
randomized block design
```

with a fixed seed.

---

# Phase O — Capture All Measurements

For every trial, keep:

## Task Features

```text
image bytes
image width
image height
prompt tokens
task type
max output tokens
```

## System State

```text
uplink goodput
downlink goodput
RTT
edge memory
edge GPU utilization
cloud GPU utilization
cloud queue/load if available
```

## Execution

```text
edge compute
serialization
transfer
cloud compute
decode
total latency
```

## Communication

```text
payload bytes
request wire bytes
response bytes
```

## Output

```text
generated tokens
output hash
quality score
```

## Provenance

```text
Git SHA
model SHA
dataset SHA
host pair
network block
timestamp
```

---

# Phase P — Add Real GPU / System-State Monitoring

Once the basic physical profiler works, add:

\[
S =
[
B_{up},
B_{down},
RTT,
U_e,
M_e,
U_c,
Q_c
]
\]

where:

- \(U_e\) = edge GPU utilization
- \(M_e\) = available edge memory
- \(U_c\) = cloud GPU utilization
- \(Q_c\) = cloud queue/backlog

Not all of these need to be used by Predictor V1.

But they should be **recorded** now.

They may become useful during analysis.

---

# Phase Q — Complete Physical G2

At this point you should have:

```text
real edge host
real cloud host
controlled bandwidth
controlled RTT
real images
real prompts
multiple repetitions
real memory measurements
verified byte accounting
stage timing
exact output equivalence
```

Then generate the new physical profile.

Do not reuse the old localhost calibration.

Create something like:

```text
configs/profiles/phase2_physical_v1.json

artifacts/phase2/physical-v1/
```

### G2 Must Show

- sufficient samples,
- zero unexplained negative residuals,
- consistent units,
- task coverage,
- network coverage,
- memory measurements,
- equality/correctness evidence,
- independent network verification.

---

# Phase R — Fit Predictor V2 on Physical Data

Recalibrate using only:

```text
training request IDs

training network blocks
```

The predictor equation can remain simple.

Do not switch to ML yet.

For example:

\[
\hat L(C)
=
L_{compute}(C,T)
+
L_{network}(C,S)
\]

with:

\[
L_{network}
=
lpha
+
eta RTT
+
rac{8D_{up}}{B_{up}}
+
rac{8D_{down}}{B_{down}}
\]

---

# Phase S — Evaluate Latency Prediction

For held-out requests/network states compute:

## Median APE

\[
Median\left(
rac{|\hat L-L|}{L}
ight)
\]

## p90 APE

## Bias

\[
\hat L-L
\]

## Interval Coverage

These are important, but secondary to ranking.

---

# Phase T — Evaluate Ranking Accuracy

This is the crucial metric.

For each condition:

\[
C_{predicted}
=
rg\min_C\hat L(C)
\]

Compare with:

\[
C_{measured}
=
rg\min_C L(C)
\]

Measure:

\[
Top1Accuracy
=
P(C_{predicted}=C_{measured})
\]

The real goal of Predictor V2 should be:

> **reliably rank configurations.**

---

# Phase U — Compare Against a Simple Baseline Predictor

Keep the constant-median predictor.

Compare:

```text
Component predictor

vs

Constant median
```

The new predictor should be meaningfully better at configuration ranking.

If it is not, investigate rather than hiding the result.

---

# Phase V — Pass Physical G3

The current proposed criteria are good:

```text
median APE ≤ 15%

p90 APE ≤ 30%

ranking accuracy >= constant-median baseline

out-of-coverage handling correct
```

Once G3 passes, you finally have evidence that the scheduler's performance model is useful.

---

# Phase W — Evaluate Selector V1 Against the Measured Oracle

Take held-out conditions.

For every:

```text
request
network state
```

collect observed latency for all three fixed configurations.

The measured oracle is:

\[
C_{oracle}
=
rg\min_C L(C)
\]

The scheduler chooses:

\[
C_{scheduler}
=
rg\min_C \hat L(C)
\]

Then calculate:

## Top-1 Agreement

\[
C_{scheduler}=C_{oracle}
\]

## Absolute Regret

\[
R =
L(C_{scheduler})
-
L(C_{oracle})
\]

## Relative Regret

\[
R_\%
=
rac{L(C_{scheduler})-L(C_{oracle})}
{L(C_{oracle})}
\]

---

# Phase X — Pass Physical G4

A reasonable target is:

```text
median regret <= 10%

>= 10 complete paired blocks

network holdout included
```

Also report:

```text
no-feasible decisions
SLA violations
unsafe selections
out-of-coverage states
```

---

# Phase Y — Enable Automatic Performance Selection

Only after G3 and G4 pass.

At that point you should be able to remove:

```text
--diagnostic-allow-unpassed-gates
```

for the physical adaptive experiment.

Now the scheduler can legitimately select configurations automatically.

---

# Phase Z — Run Real Adaptive Network Traces

Create realistic changing-network traces.

For example:

```text
time       bandwidth
--------------------
0–20s      100 Mbps
20–40s      10 Mbps
40–60s       2 Mbps
60–80s      25 Mbps
80–100s    100 Mbps
```

and optionally vary RTT too.

Then observe:

```text
selected configuration
predicted latency
observed latency
switch count
selection overhead
monitoring overhead
```

---

# Phase AA — Evaluate Switching Stability

A scheduler should not oscillate:

```text
edge
cloud
edge
cloud
edge
cloud
```

because bandwidth fluctuates slightly.

Test noisy conditions around the decision boundary.

If oscillation occurs, introduce hysteresis.

For example:

\[
L_{new}
<
0.9L_{current}
\]

before switching.

Do **not** introduce hysteresis unless a real problem is observed.

---

# Phase AB — Run the Full Phase 7 Experiment

You should have four policies:

```text
1. Edge fixed
2. Cloud fixed
3. Distributed fixed
4. Adaptive
```

For every network block use the same:

```text
requests
warmups
repetitions
model
generation parameters
```

Then compare them.

---

# Phase AC — Select the Best Static Baseline Using Validation Only

Do not select the best fixed policy after looking at the evaluation set.

Use validation:

```text
Edge
Cloud
Distributed
```

Choose best static.

Then freeze it.

---

# Phase AD — Final Latency Metrics

Report at least:

```text
median latency
p95 latency
SLA success
failure rate
```

for:

```text
adaptive
best-static
edge
cloud
distributed
oracle
```

---

# Phase AE — Final Scheduler Metrics

Also report:

```text
top-1 oracle agreement

absolute regret

relative regret

configuration distribution

number of switches

selection overhead

monitoring overhead
```

These are some of the most important results for the thesis.

---

# Phase AF — Communication Evaluation

Report:

```text
mean uplink bytes/request
mean total wire bytes/request
```

for each policy.

For example:

```text
Edge:
0 network bytes

Cloud:
image bytes

Projected:
~3 MB features
```

At low bandwidth, the projected split may be worse than uploading a compressed image.

That could become an important thesis result.

---

# Phase AG — Run Official VQA Evaluation

After physical VQA runs, export:

```text
edge predictions
cloud predictions
distributed predictions
adaptive predictions
```

Then run the official VQA evaluator.

Report:

```text
official VQA accuracy
```

per policy.

Ideally:

\[
Accuracy_{edge}
pprox
Accuracy_{cloud}
pprox
Accuracy_{distributed}
pprox
Accuracy_{adaptive}
\]

This establishes:

> placement changes performance, not model semantics.

---

# Phase AH — Make Task Awareness Stronger

Only after the basic physical system works.

Right now the task state includes:

```text
image dimensions
image bytes
prompt tokens
task bucket
expected output tokens
```

Next expand to:

```text
VQA
captioning
reasoning
```

Then analyze whether the best configuration changes by task.

For example:

```text
short VQA
   ↓
LLM work small
   ↓
vision/network trade-off matters more

long reasoning
   ↓
LLM dominates
   ↓
cloud execution potentially dominates
```

---

# Phase AI — Improve Expected-Output-Length Prediction

Currently the predictor often uses:

```text
training median output length
```

which is safe but simplistic.

Eventually use task-specific estimates:

```text
VQA:
~5 tokens

caption:
~30 tokens

description:
~100 tokens
```

or a lightweight estimator.

Do not use the actual generated output length because that leaks future information into the decision.

---

# Phase AJ — Add Task-Aware Analysis

The final system should ideally be able to produce results like:

| Task | Network | Best configuration |
|---|---|---|
| VQA | slow | Edge |
| VQA | fast | Cloud |
| Caption | slow | Edge / Split |
| Caption | fast | Cloud |
| Reasoning | most states | Cloud |

That type of result would be powerful in the final thesis.

---

# Phase AK — Add Cloud / Edge Load Experiments

After network adaptation works, introduce compute contention.

For example:

```text
Cloud load:
0%
25%
50%
75%

Edge load:
0%
25%
50%
75%
```

Then test whether the scheduler correctly changes configuration.

Ultimately:

\[
C^*
=
F(
Bandwidth,
RTT,
EdgeLoad,
CloudLoad,
Task
)
\]

rather than merely:

\[
C^*=F(Bandwidth)
\]

---

# Phase AL — Add Queue Awareness

If multiple cloud requests are possible, track:

```text
queue depth
request backlog
```

This becomes especially useful because:

```text
fast network + overloaded cloud
```

may make edge execution preferable.

This is optional for the first thesis version but valuable if time permits.

---

# Phase AM — Only Then Add Another Internal Split

Once the three-config scheduler is validated, return to dynamic partitioning.

Prioritize a **vision-encoder split**, not an LLM split.

LLM splitting is much more complex because of:

```text
per-token communication
KV cache ownership
session state
```

A more realistic extension is:

```text
Vision Encoder
─────────────────────────

blocks 0 ... k
      EDGE
        │
        ▼
activation
        │
        ▼
blocks k+1 ... 26
      CLOUD
        │
        ▼
projector
        │
        ▼
LLM
```

---

# Phase AN — Profile Candidate Internal Boundaries

For each candidate \(k\), record:

```text
activation shape
dtype
bytes
edge compute
cloud compute
edge memory
cloud memory
```

Do not assume deeper edge execution always sends less data.

Transformer hidden-state dimensions may remain constant across blocks.

---

# Phase AO — Implement Only the Best 1–2 Internal Candidates

Do not add 27 splits just because there are 27 blocks.

After analysis, perhaps choose:

```text
Vision split early

Vision split late
```

plus the current projector split.

The candidate space might become:

```text
edge/full

cloud/raw_image

vision/block-8

vision/block-20

distributed/projected
```

That is already enough for a strong paper.

---

# Phase AP — Recalibrate Everything

New configurations mean:

> new profiling.

Do **not** reuse the old predictor.

Repeat:

```text
profile
↓
calibrate
↓
G3
↓
G4
↓
adaptive evaluation
```

for the expanded configuration space.

---

# Phase AQ — Optional ML Scheduler

Only after a large profiling dataset exists.

Possible comparisons:

## Analytical Predictor

\[
f(S,T,C)ightarrow latency
\]

vs.

## Regression Model

```text
Random Forest
XGBoost
```

or classification:

\[
f(S,T)ightarrow C^*
\]

But this should be an **extension experiment**, not the core framework.

The deterministic cost model is already appropriate for the thesis.

---

# Phase AR — Optional Energy Experiment

Only do this if proper measurement hardware/software exists.

You would need:

```text
sensor
sampling frequency
idle baseline
measurement boundary
edge/cloud accounting
```

Otherwise omit energy entirely.

---

# Final Experimental Matrix

By the end, the core evaluation should look roughly like:

\[
Network
	imes
Task
	imes
Policy
	imes
Repetition
\]

For example:

```text
6 network states
×
100 VQA requests
×
4 policies
×
3 repetitions
```

which gives:

\[
6	imes100	imes4	imes3
=
7200
\]

trials.

You do not necessarily need that exact number, but it illustrates why the experimental design must be frozen before running.

---

# Final Research Comparisons

The thesis should ultimately compare:

| Policy | Purpose |
|---|---|
| Edge-only | Baseline |
| Cloud-only | Baseline |
| Fixed distributed | Distributed VLM baseline |
| Best static | Strong static baseline |
| Adaptive | Proposed framework |
| Measured oracle | Upper reference |

And evaluate:

| Metric |
|---|
| Median latency |
| p95 latency |
| SLA success |
| Communication bytes |
| Failure rate |
| VQA accuracy |
| Oracle agreement |
| Regret |
| Switching frequency |
| Scheduler overhead |
| Monitor overhead |

---

# Mandatory vs Optional Work

## Must Have for a Strong Bachelor Thesis

1. Green CI.
2. Two physical machines.
3. Verified model equivalence.
4. Measured GPU memory.
5. Controlled physical bandwidth/RTT.
6. Real task dataset.
7. Physical profiling.
8. Fresh physical predictor calibration.
9. Held-out request + network evaluation.
10. G3 predictor validation.
11. G4 scheduler validation.
12. Adaptive physical execution.
13. Fixed baselines.
14. Measured oracle.
15. Regret analysis.
16. Phase 7 paired evaluation.
17. Official VQA scoring.
18. Statistical uncertainty / repetitions.
19. Reproducible artifacts.
20. Honest negative results if adaptation does not win.

## Strong Additions

- task-aware scheduling,
- edge/cloud load,
- cloud queue,
- dynamic network traces,
- hysteresis,
- one or two new internal vision splits.

## Optional / Probably Unnecessary

- RL,
- neural scheduler,
- pruning,
- quantization,
- energy,
- LLM-layer splitting,
- multiple different VLMs.

---

# Immediate Sequence When the Second Machine Is Available

If everything above is reduced to the concrete first steps, follow this order:

1. **Fix CI and freeze a clean Git commit.**
2. **Inventory both machines.**
3. **Install identical software/model artifacts.**
4. **Start the cloud server.**
5. **Verify health + capability paths.**
6. **Run edge/cloud/distributed correctness.**
7. **Verify token equality.**
8. **Measure real edge/cloud GPU memory.**
9. **Set up bandwidth + RTT shaping.**
10. **Verify actual goodput/RTT independently.**
11. **Freeze real VQA/task manifests.**
12. **Freeze train/validation/test request IDs and network blocks.**
13. **Run physical fixed-policy profiling.**
14. **Generate the new Phase 2 profile.**
15. **Fit a completely new physical predictor.**
16. **Evaluate ranking + latency error on untouched data.**
17. **Pass G3.**
18. **Evaluate scheduler vs measured oracle.**
19. **Pass G4.**
20. **Run automatic adaptive execution on new held-out conditions.**
21. **Run best-static and all fixed baselines.**
22. **Run Phase 7 paired analysis.**
23. **Run official VQA evaluation.**
24. **Only then add stronger task awareness / internal splits.**

---

# Recommended Strategy Going Forward

The most important change in strategy now is:

> **Stop building horizontally and start validating vertically.**

The project already has enough components.

The next milestone is a complete, controlled path from:

> **real network state → prediction → selection → execution → measured oracle comparison**

Once that works, every additional feature—task awareness, compute load, queue awareness, or new split points—becomes a clean research extension rather than another unvalidated subsystem.
