# Elapsed-time clock for physical profiling

The local 100-request edge diagnostic on 2026-11-08 completed all 100 measured attempts plus one warmup without inference failures or OOM. GPU reservation stabilized at 4,570 MiB and the sampled GPU temperature peaked at 73°C. Raw diagnostic evidence is retained under `artifacts/unattended/edge-100-validation-20261108/` and remains separate from the official rotated campaign.

That diagnostic also exposed a timing problem: UTC stepped backward by approximately 2.7 seconds in a 30-second trace. An independent Windows .NET Stopwatch comparison measured the Linux adjusted monotonic clock at approximately 1.1001 times Windows elapsed time, while `CLOCK_MONOTONIC_RAW` agreed at approximately 1.0001. **The pre-fix diagnostic supports functional and memory validation; its latency statistics are provisional and must not be treated as a calibrated physical performance baseline.** Do not retrospectively rescale its rows with a single correction factor: the adjusted clock rate varied over time.

`adaptive_vlm.timing` now supplies a common nanosecond/second duration clock. Linux uses `clock_gettime_ns(CLOCK_MONOTONIC_RAW)`; other platforms use Python's performance counter. Model stages, benchmark attempt duration, transport/client/server timing, scheduler measurements, and TCP echo measurements use this common clock. UTC remains a provenance timestamp and is never subtracted to calculate inference duration. Application watchdog and cache-age calculations also use the common clock; operating-system socket and subprocess waits retain their native timeout behavior.

Every newly collected environment records `duration_clock`. Fixed and adaptive benchmark resume reject a changed duration clock; profile conversion rejects mixed clock metadata. Historical runs with no clock metadata can still be analyzed together as historical records, but cannot be resumed by the new CLI into raw-clock measurements. Use fresh run directories and keep old evidence intact.

## Host synchronization

[Ubuntu's WSL documentation](https://ubuntu.com/wsl/docs/stable/explanation/time-sync/) explains that a guest NTP client can conflict with Hyper-V implicit time synchronization. On this edge, the redundant `systemd-timesyncd.service` was stopped and disabled, while Hyper-V synchronization and the kernel clock source were left unchanged. The first post-change measurements still showed adjusted-clock distortion; disabling the guest service alone is not proof of calibrated timing. Independent Windows-reference checks and use of the raw clock address the measured-duration problem.

Windows `w32tm /query /status` also reported unsynchronized status, and a resync request required an elevated Windows shell. Before leaving the university devices, verify Windows time synchronization from an Administrator PowerShell with `w32tm /resync /rediscover`, then confirm its status. This does not substitute for checking each host's elapsed-time clock. Validate cloud/edge clock behavior during the site preflight; timestamps from different machines are not used as one-way latency measurements.

## Before the official campaign

Check out the same clean timing-fix commit on both hosts and **restart the cloud GPU server after updating source**, so both client and server use the same timing implementation. Keep the pinned model, BF16 precision, preprocessing and 32-token budget unchanged. Use a fresh campaign output root for this measurement revision, run the full three-placement correctness preflight and live achieved-network gates, and keep the existing rotating placement order. Record clock metadata with the raw artifacts. Re-estimate remaining runtime after the first complete controlled block.

The [Linux clock manual](https://www.man7.org/linux/man-pages/man2/clock_gettime.2.html) documents that `CLOCK_MONOTONIC` can be frequency-adjusted, whereas `CLOCK_MONOTONIC_RAW` excludes those adjustments. A raw clock is still host-local: its agreement with an independent reference should be checked rather than assumed across arbitrary environments.
