# Physical testbed: profiling protocol

How fixed-policy profiling blocks are run and timed on the physical pair.

## Fixed-policy profiling campaign

Profiling uses the [frozen physical VQA manifest and split](vqa-dataset.md) and the [measured nine-block network grid](network-measurement.md). The edge orchestrates all three fixed placements using `scripts/campaign/run_physical_profile_block.py`; the cloud keeps the RTX 2070 SUPER GPU server on TCP 5050 and the independent `iperf3` and echo probes on 5051/5052. The edge runs full and vision-only stages on its RTX 3050 GPU. The full edge model succeeded on one real 427×640 COCO image in 17.5 seconds, with 4,450 MiB peak reserved memory reported despite a 4,096 MiB device. Treat edge/full as conditional: record OOM and other failures, and never drop them from denominators.

The frozen block settings are in `configs/profiles/physical_network_blocks_v1.json`. For each block, apply its rate and egress delay to **both** WSL `eth0` queues. Each host contributes one egress delay, so the configured round-trip addition is twice the per-host delay. Cloud shaping is applied by the cloud agent; edge shaping is applied with `sudo tc qdisc replace dev eth0 root netem delay <delay> rate <rate>mbit`. The edge runner checks its qdisc and measures both directions with `iperf3` plus full-path TCP echo before inference. It repeats those measurements after all three placements. It rejects a rate outside 80–120% of target or a median RTT outside the declared tolerance. Independent measurements and raw output are retained in each block directory.

For a shaped block such as N1, run on edge WSL after the cloud agent confirms the matching cloud queue:

```bash
sudo tc qdisc replace dev eth0 root netem delay 10ms rate 2mbit
.venv-model/bin/python scripts/campaign/run_physical_profile_block.py --block N1
```

The runner requires a clean Git checkout and exact frozen dataset and model hashes. It uses one warmup and three measured repetitions per policy for all 100 requests. The request schedule is seeded and shared across policies. Policy order rotates through six permutations across the nine network blocks; each placement runs in a separate process so its GPU allocation is released before the next placement. This controls order drift at the block level; it does not interleave placements per request. The runner supports resuming an incomplete block without changing its Git/data plan. It samples edge GPU utilization and free/used VRAM with `nvidia-smi` during the policy runs.

Each block produces `plan.json`, `environment.json`, `network_pre.json`, `network_post.json`, `network_evidence.json`, three policy run directories, their logs, `edge_gpu_samples.csv`, `profile/profile.jsonl`, `profile/summary.json`, and `validation_report.json` under `artifacts/phase2/physical-v1/N*`. The profile converter validates the seeded schedules, model/data/config hashes, token counts, byte accounting, residuals, and hashed independent network evidence. VQA rows now carry `task_type=vqa_short_answer`. The in-run VQA quality score is provisional until the official evaluator is run. All raw artifacts are ignored by Git and must be archived separately.

The declared full development matrix is 9 network blocks × 100 requests × 3 placements × 3 repetitions = 8,100 measured attempts, plus 27 warmups. A representative edge/full real-image attempt took 17.5 seconds, so this is a long campaign. Do not mark Physical G2/P7 complete until every block has complete, validated rows, failure accounting, cloud and edge memory evidence, and a reviewed aggregate report. After the last block, remove both temporary qdiscs with `sudo tc qdisc del dev eth0 root` on edge and `tc qdisc del dev eth0 root` on cloud, then verify each shows `qdisc mq`.

## Elapsed-time clock

The local 100-request edge diagnostic on 2026-11-08 completed all 100 measured attempts plus one warmup without inference failures or OOM. GPU reservation stabilized at 4,570 MiB and the sampled GPU temperature peaked at 73°C. Raw diagnostic evidence is retained under `artifacts/unattended/edge-100-validation-20261108/` and remains separate from the official rotated campaign.

That diagnostic also exposed a timing problem: UTC stepped backward by approximately 2.7 seconds in a 30-second trace. An independent Windows .NET Stopwatch comparison measured the Linux adjusted monotonic clock at approximately 1.1001 times Windows elapsed time, while `CLOCK_MONOTONIC_RAW` agreed at approximately 1.0001. **The pre-fix diagnostic supports functional and memory validation; its latency statistics are provisional and must not be treated as a calibrated physical performance baseline.** Do not retrospectively rescale its rows with a single correction factor: the adjusted clock rate varied over time.

`adaptive_vlm.timing` now supplies a common nanosecond/second duration clock. Linux uses `clock_gettime_ns(CLOCK_MONOTONIC_RAW)`; other platforms use Python's performance counter. Model stages, benchmark attempt duration, transport/client/server timing, scheduler measurements, and TCP echo measurements use this common clock. UTC remains a provenance timestamp and is never subtracted to calculate inference duration. Application watchdog and cache-age calculations also use the common clock; operating-system socket and subprocess waits retain their native timeout behavior.

Every newly collected environment records `duration_clock`. Fixed and adaptive benchmark resume reject a changed duration clock; profile conversion rejects mixed clock metadata. Historical runs with no clock metadata can still be analyzed together as historical records, but cannot be resumed by the new CLI into raw-clock measurements. Use fresh run directories and keep old evidence intact.

### Host synchronization

[Ubuntu's WSL documentation](https://ubuntu.com/wsl/docs/stable/explanation/time-sync/) explains that a guest NTP client can conflict with Hyper-V implicit time synchronization. On this edge, the redundant `systemd-timesyncd.service` was stopped and disabled, while Hyper-V synchronization and the kernel clock source were left unchanged. The first post-change measurements still showed adjusted-clock distortion; disabling the guest service alone is not proof of calibrated timing. Independent Windows-reference checks and use of the raw clock address the measured-duration problem.

Windows `w32tm /query /status` also reported unsynchronized status, and a resync request required an elevated Windows shell. Before leaving the university devices, verify Windows time synchronization from an Administrator PowerShell with `w32tm /resync /rediscover`, then confirm its status. This does not substitute for checking each host's elapsed-time clock. Validate cloud/edge clock behavior during the site preflight; timestamps from different machines are not used as one-way latency measurements.

### Before the official campaign

Check out the same clean timing-fix commit on both hosts and **restart the cloud GPU server after updating source**, so both client and server use the same timing implementation. Keep the pinned model, BF16 precision, preprocessing and 32-token budget unchanged. Use a fresh campaign output root for this measurement revision, run the full three-placement correctness preflight and live achieved-network gates, and keep the existing rotating placement order. Record clock metadata with the raw artifacts. Re-estimate remaining runtime after the first complete controlled block.

The [Linux clock manual](https://www.man7.org/linux/man-pages/man2/clock_gettime.2.html) documents that `CLOCK_MONOTONIC` can be frequency-adjusted, whereas `CLOCK_MONOTONIC_RAW` excludes those adjustments. A raw clock is still host-local: its agreement with an independent reference should be checked rather than assumed across arbitrary environments.
