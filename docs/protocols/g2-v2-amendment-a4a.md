# G2 v2 protocol, amendment A4a: the edge-probe shape after the pilot no-go

**Dated 20 November 2026.** Written after the A4 section 5 calibration returned a no-go, and before any G5 pilot or campaign data was collected. [A4](g2-v2-amendment-a4.md) requires a dated amendment in this case. Everything in A4 other than the probe shape is unchanged.

## 1. The no-go

**What was run.** The A4 calibration ran inside the G5 image (`sha256:0cf64e17…`, commit `aa4459c`) on the edge:
- probe: five 2048 × 2048 bf16 multiplies with one synchronise at the end;
- 60 probes at L0 and 60 under L1.

**Result:**

| | p05 | Median | p95 |
| --- | ---: | ---: | ---: |
| L0 | 1.39 ms | 7.78 ms | 7.83 ms |
| L1 | 1.35 ms | 7.74 ms | 8.65 ms |

The threshold, √(7.78 × 7.74) ≈ 7.76 ms, does not lie between p95(L0) and p05(L1). This is a **no-go**.

**Why the probe missed the load.**
- On AWS, L1 is a light competitor: a 1 ms kernel every 11 ms, on its own CUDA stream (A2a-3, A2b).
- A single long burst of large multiplies barely overlaps it.
- The model still slows 1.4× under L1, because inference issues many short kernels and synchronises often, and each synchronise can wait behind a competitor kernel.
- The L4-6Q is also a time-sliced vGPU. Short probe timings jump between about 1 ms and 7–8 ms with the scheduler's slices, whatever the load.

## 2. The new probe shape (calibration data only)

**Candidates.** Five shapes, measured right after the no-go with a sweep script in the same image. They are numbered so the selection rule can refer to them:

| # | Shape |
| --- | --- |
| 1 | 2048 × 2048 × 5, one synchronise (the A4 probe) |
| 2 | 1024 × 1024 × 10, synchronise after each |
| 3 | 512 × 512 × 20, synchronise after each |
| 4 | 256 × 256 × 40, synchronise after each |
| 5 | 128 × 128 × 80, synchronise after each |

Each shape had 40 probes at L0 and 40 under L1, interleaved across shapes with 0.1 s gaps.

**Selection rule:** choose the shape with the largest separation p05(L1) ÷ p95(L0), subject to a median L0 probe time of at most 20 ms.

**Sweep results:**

| Shape | L0 p05 / median / p95 (ms) | L1 p05 / median / p95 (ms) | p05(L1) ÷ p95(L0) |
| --- | --- | --- | ---: |
| 2048 × 5 | 7.33 / 7.74 / 7.82 | 7.63 / 8.11 / 15.03 | 0.98 |
| 1024 × 10, sync | 0.96 / 0.99 / 7.31 | 7.33 / 11.63 / 15.98 | 1.00 |
| 512 × 20, sync | 0.89 / 4.07 / 7.31 | 0.92 / 0.94 / 0.97 | 0.13 |
| 256 × 40, sync | 1.89 / 8.22 / 8.30 | 8.25 / 12.62 / 16.94 | 1.00 |
| **128 × 80, sync** | **10.08 / 16.37 / 16.46** | **25.08 / 25.15 / 25.25** | **1.52** |

**Selected: 128 × 128 bf16, 80 repetitions, synchronise after each.** It's the only shape that separates the two states. Its L0 median is 16.4 ms, under the 20 ms limit.

**The A4 rule then runs unchanged on fresh samples.** That is 60 probes per state with the new shape:
- threshold = √(median L0 × median L1);
- go iff p95(L0) < threshold < p05(L1).

**On a second no-go,** edge-load sensing is dropped: the adaptive policy takes the edge as L0, and that limitation is reported. This is decided now, to avoid a third iteration.

**Overhead.** The new probe costs about 16–25 ms per decision. A4's pilot criterion (median adaptive overhead ≤ 50 ms) still applies.

## 3. Changes

- **Code:** `SensingConfig` gains `edge_probe_synchronize_each`, and `torch_edge_probe` synchronises after each multiply when it is set.
  - **Consequence:** this changes `src/`. The image is rebuilt, and the reference bundles are rebuilt for it, before the calibration and the pilot.
- **Template:** `configs/g5/sensing_template.json` takes the selected shape.
- **Evidence:** the sweep data stays on the edge data volume (`/artifacts/g5/probe_sweep.json`) and is mirrored with the G5 data.
