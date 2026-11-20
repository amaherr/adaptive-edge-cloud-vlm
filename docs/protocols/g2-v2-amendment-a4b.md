# G2 v2 protocol, amendment A4b: cloud sensing off the decision path after the pilot no-go

**Dated 20 November 2026.** Written after the first G5 pilot returned a no-go, and before any G5 campaign data was collected. A4 and A4a are unchanged except as stated here.

## 1. The no-go

**What was run.** The pilot ran on 20 November, 18:14–18:28 UTC:
- image `sha256:f54c773e…`, commit `b3296cc`;
- two segments, E01 L0/C0 and E01 L1/C2, with 30 evaluation IDs each.

**Pilot criteria (A4 section 5):**

| Criterion | Result |
| --- | --- |
| Both segments valid | **pass**: 360 of 360 measured attempts |
| Cloud sensing accuracy ≥ 99% | **pass**: 100% |
| Edge sensing accuracy ≥ 95% (A4a probe, threshold 20.65 ms) | **pass**: 98.3% |
| Every adaptive answer equals the chosen configuration's | **pass**: 60 of 60 |
| Median adaptive overhead ≤ 50 ms | **fail**: 126 ms (p95 139 ms) |

**Where the overhead came from:**
- **Edge probe:** 15 ms at L0, 25 ms at L1.
- **Cloud probe:** about 100 ms, because each decision opened a fresh TCP connection for a health exchange with the server, about two round trips on E01's 50 ms link.

The selector's choices were right: 98.3% top-1 agreement with the per-request oracle. The overhead alone pushed the median relative regret to 19%.

## 2. Change

**Cloud load is sensed in the background** (`BackgroundCloudState`):
- **Polling:** a daemon thread polls the server's health response, which carries its cached GPU state, every **1 s** (`cloud_poll_interval_s`), on its own connection.
- **Reuse:** every remote response's `server_state_*` fields update the same cache.
- **Reading:** each decision reads the newest cached sample. The C2 rule (GPU power ≥ 58 W) is unchanged.
- **Age:** the sample's age is recorded with each decision. It is the server's own sample age plus the time since the sample arrived.

**The edge probe stays on the decision path.** It runs on the GPU, so a background probe would compete with inference.

**Expected overhead:** about 15–25 ms per decision.

**Trade-off and disclosure.** The background poll adds a health exchange once a second on the shaped link: a few hundred bytes, on its own connection. Its effect on the measured fixed configurations is negligible but not zero, and the thesis states it.

## 3. Then

- **Rebuild.** The change is in `src/`, so the image is rebuilt and the references are rebuilt for it.
- **Calibration.** The A4a calibration stands: the probe code is unchanged and so is its threshold. `sensing.json` (SHA-256 `4fb9a112…`) is kept, and its poll interval takes the declared default of 1 s.
- **Pilot.** The pilot reruns with the same grid into a new output directory (`/artifacts/g5/pilot2`), against the same go/no-go criteria.
- **On a second overhead no-go,** the campaign does not start; the result is reported instead. This is decided now.
