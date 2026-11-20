# G5 loopback rehearsal (20 November 2026)

This is the roadmap 5.4 rehearsal of the G5 trace campaign ([amendment A4](../../docs/protocols/g2-v2-amendment-a4.md)) on loopback. It ran with the fake model, the real supervisor, a real edge-load worker and a real cloud-load worker on the local GPU, and injected faults. It is **not a measurement**: the fake model's latencies and the fixed sensing readings mean nothing.

**Command:**

```bash
.venv-model/bin/python scripts/campaign/rehearse_g2v2_faults.py \
  --settings configs/campaigns/g5_loopback_rehearsal.json --fresh
```

**Grid:** [rehearsal_grid_g5.json](../../configs/profiles/rehearsal_grid_g5.json). One trace on E01: L0/C0 → L1/C1 → L0/C0. Each segment runs the five fixed configurations plus the adaptive placement on 3 request IDs, with one warmup each, which makes 24 attempts per segment.

**Faults injected ([faults-report.json](faults-report.json)). Every check passed:**

| Fault or check | Result |
| --- | --- |
| Supervisor hard-killed (SIGKILL) in the first segment | Its benchmark and load children exited on their own; the restarted campaign resumed the segment |
| Edge-load worker killed in the L1 segment | The attempt ended with exit code 76 and resumed automatically |
| Cloud-load worker killed in the C1 segment | The cloud-load gate invalidated the segment, and the supervisor reran it |
| Journals | No gaps or duplicates; interleave positions contiguous across all six placements |
| Campaign | Completed, all 3 segments valid, all attempts recorded, no stray processes |
| G5 analysis | `analyze_g5.py` built [g5_summary.json](g5_summary.json) |

**What the analysis shows here.** It confirms the mechanics only:
- 9 of 9 adaptive requests succeeded;
- every adaptive answer equals its chosen configuration's answer;
- median sensing and selection overhead is 0.76 ms.

**Sensing is not tested here.** The rehearsal uses a fixed edge-probe reading (5 ms, so always L0) and the laptop GPU's power (under 58 W, so always C0). The 66.7% sensing accuracy and the gate result are therefore meaningless. Sensing is calibrated and checked in the AWS pilot (A4 section 5).
