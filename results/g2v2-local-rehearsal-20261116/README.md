# G2 v2 local rehearsal and edge pilot (16 November 2026)

This is the local part of W8 in the [G2 v2 preparation](../../docs/g2-v2-preparation.md). The on-site pilot with the cloud PC is still to come; see that document's on-site checklist.

## Fault-injection rehearsal

[`rehearse_g2v2_faults.py`](../../scripts/campaign/rehearse_g2v2_faults.py) ran the real campaign script in loopback mode on this laptop.
- **Setup:** fake model on both sides, a real `adaptive-server --report-resource-state`, the real GPU load worker for the loaded sub-block, and a 400 ms delay per model call so faults land inside a sub-block.
- **Grid:** [rehearsal grid](../../configs/profiles/rehearsal_grid_v2.json), with T01 at L0 and L1, plus V01 and E01.

| Step | What happened |
| --- | --- |
| Hard kill | `SIGKILL` to the supervisor after 4 committed attempts of T01-L0-r0. Its benchmark child exited by itself (parent-death signal), so no stray process was left and the sweep had nothing to do. |
| Restart | `start` again; T01-L0-r0 resumed and finished with all 21 attempts. No torn journal tail, no duplicates, contiguous interleave positions. |
| Competitor failure | `SIGKILL` to the L1 load worker after 4 attempts of T01-L1-r0. The next pre-attempt check failed, the benchmark exited with code 76, and the supervisor logged the restart in `restarts.jsonl`, started a new generator and finished the sub-block. |
| Completion | 54/54 measured attempts, all four sub-blocks valid, none moved to `invalid/`. |
| Derived tables | `derive_g2v2_tables.py all` built 36 training rows (L0 and L1), 9 sealed validation inputs and the training oracle table. The evaluation sub-block was not opened. |

All eight checks in [faults_report.json](faults_report.json) passed.

**Gap found and fixed during W8.** Before this rehearsal, a supervisor killed outright would have left its benchmark and GPU load worker running. Each runs in its own session, and SIGTERM skipped the cleanup that stops the worker. An orphaned load worker would have loaded every later L0 sub-block, and no gate would have noticed. The fix has four parts:
1. Each child asks the kernel to stop it when its parent dies (`exit_with_parent`).
2. SIGTERM goes through the normal cleanup.
3. A run directory accepts only one writer (`flock`).
4. The supervisor stops any leftover campaign process before each sub-block and logs it in `stray_processes.jsonl`.

## Edge pilot (real model)

The real Moondream model on the RTX 3050, with swap residency and prompt P1. The edge half of distributed is the vision encoder and projector, with no network. Twenty prompt-calibration questions per level (not G2 v2 data), after one warmup per placement; the generator was checked before every request.

| Level | Edge/full median (p90) | Distributed edge half median (p90) | Peak Windows dedicated | Edge answers vs. P1 calibration |
| --- | ---: | ---: | ---: | --- |
| L0 | 1,153 ms (1,185) | 518 ms (521) | 3,271 MiB | 20/20 identical |
| L1 | 1,616 ms (1,643) | 821 ms (826) | 3,862 MiB | 20/20 identical |
| L2 | 2,425 ms (2,497) | 1,358 ms (1,363) | 3,862 MiB | 20/20 identical |

- **No spill at L0:** peak dedicated usage is 3,271 MiB, under the protocol's 3,832 MiB limit.
- **Answers don't depend on load:** edge outputs match the calibration run token for token at every level.
- **Distributed is also slowed by edge load:** its vision half takes 2.6 times longer at L2 than at L0. Distributed is therefore not load-independent, and the network still decides cloud against distributed.
- These questions are not the ones used in the W5 calibration, so the absolute values differ slightly from W5's (1.27, 1.85, 2.81 s); the ratios between levels are similar.

Files: [edge_pilot_summary.json](edge_pilot_summary.json) and [edge_pilot.jsonl](edge_pilot.jsonl), with one row per request including clocks, temperature and Windows memory counters.
