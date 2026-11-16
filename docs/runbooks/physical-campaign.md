# Unattended physical campaign runbook

> **Running G2 v2? Go straight to [G2 v2 campaign](#g2-v2-campaign)** and print the [on-site checklist](g2v2-onsite-checklist.md). The sections before it describe the G2 v1 campaign (`run_physical_campaign.py`), kept so that G2 can be reproduced. The cloud service unit in `configs/systemd/` now serves the v2 config; for G2 v1, change it back as noted in the unit.

This runs on the **edge laptop at university**, beside the physical cloud PC. The edge schedules all nine frozen network blocks, all three placements, 100 requests and three repetitions: **8,100 measured attempts and 27 excluded warmups**. Connect from home over SSH/Tailscale to monitor the edge. The inference and measurement addresses remain the university LAN addresses; the third device is outside the experiment path.

The entry point is `scripts/campaign/run_physical_campaign.py`. It runs software/host checks and a separate three-placement GPU correctness campaign before any baseline work. It then applies each network setting to both Linux hosts, invokes the existing block runner, and requires its controlled profile gate to pass. A block failure stops the campaign. Completion means all nine profiles passed; scientific G2 closure still needs the aggregate report, official accuracy evaluation and reviewed cloud/edge memory evidence described in the existing physical protocol.

The explicitly selected [heterogeneous-GPU correctness v2 protocol](../physical-testbed/correctness-criteria.md#heterogeneous-gpu-correctness-v2-approved) keeps exact transport and same-device path checks, while auditing cross-GPU wording and requiring per-request score agreement. The original strict gate remains the default and historical reports stay unchanged.

## Test the edge without a cloud device

Run `.venv-model/bin/python -u scripts/campaign/run_physical_campaign.py edge-smoke --requests 3` from the repository root. This runs one warmup and three real VQA requests locally on the edge GPU, logs progress to `artifacts/unattended/edge-smoke/smoke.log`, and saves resumable attempts under its `run/` directory. It needs no `--settings`, SSH, cloud server or shaping. Repeat the same command after an interruption to resume, or supply a new `--smoke-dir` for a fresh test. See the final README section for software checks and full usage. These diagnostic rows are separate from the 8,100-attempt baseline and do not validate cloud/network readiness.

## Prepare at university

1. Check out the **same clean commit** on both machines. Generated local Terraform settings, caches, state and plans are ignored; experiment source changes must be committed before the clean-tree gate will pass. Keep the pinned offline model cache and frozen image manifest/files on their required hosts. Install the project's existing model and development environments. The edge needs `ssh`, `sudo`, `tc`, `iperf3` and `nvidia-smi`.
2. Verify the current LAN IPs, WSL portproxy destinations and edge-restricted Windows firewall rules for TCP **5050, 5051, 5052**. If addresses changed, update the experiment YAML, correctness-smoke YAML and network grid consistently, then commit those changes on both hosts. Re-measure the unshaped echo RTT if the physical path changed, and identify a new environment revision rather than silently reusing the old network baseline. Defaults still describe the previously measured `192.168.8.74` cloud.
3. Configure key-based SSH from edge to **cloud Linux/WSL**, using an SSH config alias if a nonstandard port or jump host is necessary. Accept and verify the host key interactively first. `cloud_ssh` must land in Linux where `tc` controls the cloud experiment interface, rather than a Windows PowerShell session. Verify `ssh CLOUD_ALIAS true`. Both hosts need passwordless, narrowly scoped `sudo -n tc` for qdisc show/replace/delete on the experiment interface; have the administrator configure this. A cached interactive sudo password will expire during this run.
4. Plug both machines into AC. In Windows power settings, disable sleep/hibernate while plugged in, including the laptop lid action, and arrange updates/reboots outside the campaign. Keep ventilation clear. WSL processes and detached sessions cannot prevent Windows sleep or resume themselves after Windows terminates WSL.
5. Set up SSH/Tailscale access to the edge from the third device and test it **before leaving**. Keep a shell open while testing a shaped low-bandwidth block. Root netem shapes all traffic on that Linux interface, including management traffic that uses it; latency/bandwidth limits can affect remote monitoring. Keep physical access until preflight and initial progress have been verified.

## Keep cloud services running

If the cloud already has supervised inference and probe services, reuse them; do not create duplicate listeners. Otherwise use the user-service templates in `configs/systemd/` on **cloud Linux**, where systemd user services are available:

```bash
mkdir -p ~/.config/systemd/user
cp configs/systemd/adaptive-vlm-*.service ~/.config/systemd/user/
```

Edit the copied GPU and echo units: replace every `/REPLACE/absolute/repo/path` with the cloud checkout's absolute Linux path. Check the `iperf3` executable path. Then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now adaptive-vlm-cloud adaptive-vlm-iperf adaptive-vlm-echo
systemctl --user status adaptive-vlm-cloud adaptive-vlm-iperf adaptive-vlm-echo
journalctl --user -u adaptive-vlm-cloud -f
```

An administrator can enable user lingering with `sudo loginctl enable-linger "$USER"` so these units survive logout. Validate logout/reconnect before relying on it. The GPU unit uses the existing WSL NVML workaround and the dual-path `adaptive-server`, so one GPU server handles both raw images and projected tensors. Restart is bounded to avoid an endless GPU load/crash loop. Enabling these Linux units does **not** boot WSL after a Windows reboot; start the distro/services again before campaign resume. Save the cloud service journal and GPU/memory evidence alongside the final artifacts.

## Configure the edge launcher

Keep machine-specific settings in an ignored artifact directory:

```bash
mkdir -p artifacts/unattended
cp configs/campaigns/physical_unattended.example.json artifacts/unattended/settings.json
```

Edit `cloud_ssh`, `cloud_repo`, and `cloud_python` with the cloud's real Linux values. Check `cloud_interface`, executable paths and `min_free_gb`. Paths in the settings resolve relative to the edge repository, except the cloud paths, which must be absolute on cloud. All settings, source commit and input hashes are frozen in `campaign_plan.json`; use the same settings and output root for resume. Use a separate output root for a different experiment. Do not point this launcher at earlier manually collected block directories without their new validation hashes.

Run the foreground preflight, while both devices are accessible:

```bash
.venv-model/bin/python scripts/campaign/run_physical_campaign.py preflight \
  --settings artifacts/unattended/settings.json
```

The checks require matching clean source commits, local and remote GPUs, passwordless shaping permissions, at least the configured free disk space on both hosts, all probe listeners, server health/model agreement, the frozen 100-image manifest and split, Ruff, mypy, and the **entire test suite with zero skipped tests**. Then all three policies run the correctness manifest with the pinned offline GPU model; the default strict protocol requires matching output tokens/text across placements; explicitly selected v2 instead checks each placement against its hash-bound reference, repeated outputs and per-request scores. Smoke artifacts live under `preflights/`, outside the measured 8,100 rows. This must run on the actual hosts with sockets/GPU access; a sandbox-skipped socket test does not pass the launch gate.

Start the real campaign after that succeeds:

```bash
.venv-model/bin/python scripts/campaign/run_physical_campaign.py start \
  --settings artifacts/unattended/settings.json
```

`start` detaches from SSH, redirects output to `campaign.log`, and uses a one-writer lock. It **repeats preflight and GPU correctness checks** each time, including on resume. Watch until at least the first real requests commit before leaving. An alternative `run` action stays in the foreground for use inside your existing service manager or tmux. Do not launch the block runner separately while the supervisor owns the campaign.

## Monitor from home

SSH into the edge, enter the checkout, and use:

```bash
tail -F artifacts/phase2/physical-v1-unattended/campaign.log
```

In a second shell:

```bash
.venv-model/bin/python scripts/campaign/run_physical_campaign.py status \
  --settings artifacts/unattended/settings.json
```

The timestamped heartbeat is emitted every 30 seconds while a child runs. It reports the active stage and block, saved measured attempts out of 8,100, percent, warmups, and status counts. `status` also checks the live writer lock and reports `supervisor_running`; a stale `state=running` file after a crash is insufficient evidence of a live process. Read `updated_at_utc` and `last_journal_write_unix` together: an active heartbeat with no new rows can mean model loading or a slow/hung request. A 900-second absence of new measured rows during a block stops the process tree and preserves its journals. The edge disk reserve is checked during heartbeats as well as startup; cloud free space is checked at startup. Preflight subprocesses have a one-hour limit.

Detailed per-request/model logs are in `N1/edge.log`, `N1/cloud.log`, `N1/distributed.log`, and equivalent directories for later blocks. For example:

```bash
tail -F artifacts/phase2/physical-v1-unattended/N1/edge.log
```

`status.json` also provides machine-readable progress, last update, failure reason and network cleanup errors. Tests and GPU smoke progress appear directly in `campaign.log` while preflights run.

## Stop, recover and resume

```bash
.venv-model/bin/python scripts/campaign/run_physical_campaign.py stop \
  --settings artifacts/unattended/settings.json
```

Wait for `supervisor_running=false` and inspect `cleanup_errors`. On graceful stop/failure the supervisor terminates its benchmark process group and attempts to remove temporary netem on **both** peers. A hard kill, power failure or inaccessible cloud may leave shaping behind; inspect `tc qdisc show dev eth0` on both hosts before recovery. A saved `shaping_active.json` identifies a prior interrupted shaping session. Do not delete artifacts, lock files or that marker to force a second writer to start. A lock file may remain on disk after exit; the kernel releases its lock automatically.

Fix the operational cause (cloud service, network, power, storage), restart cloud services if needed, and run the **same `start` command** with the same settings. Resume rechecks live network throughput/RTT before inference, preserves every earlier network preflight, and skips saved trial IDs. Fully validated blocks are skipped only after their profile, network evidence and raw-input hashes verify. A changed Git/data/model/config contract is rejected.

Every completed attempt is flushed and fsynced to `requests.jsonl` before advancing. A torn final append is copied to `requests.torn-*.bin`, then truncated to the last newline before resume. Corrupt committed lines fail closed. At most an interrupted, uncommitted request is repeated; its earlier successful execution on cloud, if any, was not an authoritative saved measurement. This is a local persistent-storage checkpoint, so destroyed/lost disks still require a separate artifact backup.

Recorded failures remain in the denominator and are **not retried or erased**. Three consecutive inference failures stop further work after saving those rows; process failures also stop immediately. This avoids consuming the remaining schedule during an outage. There is no unattended retry loop that could conceal an infrastructure problem. Resume needs an operator to repair the cause and relaunch; it retains all completed work. A failed controlled-profile gate also stops the next block; inspect the report rather than deleting rows to manufacture a passing result.

The final `state=completed` requires all nine validated blocks and exactly 8,100 measured attempts. Preserve the whole output directory, both host/service logs, model/data provenance and memory evidence in a separate backup before any cleanup or machine move. Review the aggregate results before declaring physical G2 complete. A post-run qdisc cleanup failure reports `state=failed` even if all inference rows were collected.

## G2 v2 campaign

G2 v2 runs every request on all three placements back to back, under 22 shaped network conditions and three edge-load levels.
- **Size:** 34 cells as 102 sub-blocks, giving 14,580 measured attempts and 306 warmups. The estimate is about 16–19 h; the on-site pilot replaces it with measured numbers.
- **Design:** the [protocol](../protocols/g2-v2-protocol.md), sections 5–9.
- **Status and decisions:** the [preparation notes](../g2-v2-preparation.md).
- **Driver:** `scripts/campaign/run_g2v2_campaign.py`, on the **edge laptop**. It shapes both hosts, runs one sub-block at a time, validates each one, and reruns a sub-block that fails a check.

Every command below runs from the repository root. Edge commands run in the edge's WSL and cloud commands in the cloud's WSL, unless the step says otherwise. The [on-site checklist](g2v2-onsite-checklist.md) is the same sequence as tick boxes.

### 1. Freeze the code

The correctness reference binds the source hashes, and preflight requires the same clean commit on both hosts. Finish all code changes first; any later change to `src/` means rebuilding the reference.

On the edge, check the commit you will take and rehearse the full supervisor with injected faults:

```bash
.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q
```

```bash
.venv-model/bin/python scripts/campaign/rehearse_g2v2_faults.py --fresh
```

The rehearsal must end with every check `true`. It covers a hard kill and resume, a killed load generator, and the derived tables. Keep at least 15 GB free on the Windows C: drive; a crash dump once filled it.

### 2. Cloud PC

1. **Same commit:** pull the commit and set up `.venv-model`. The model cache must be the pinned offline cache, and the server refuses an unverified one.
2. **Edge-to-cloud access:** key-based SSH from the edge must land in the cloud's **Linux/WSL**, not Windows, and `ssh CLOUD_ALIAS true` must work. Both hosts need passwordless `sudo -n tc` for qdisc show, replace and delete on the experiment interface.
3. **Services:** use the units in `configs/systemd/`, as in [Keep cloud services running](#keep-cloud-services-running).
   - `adaptive-vlm-cloud` now serves `physical_vqa_v2.yaml` with `--report-resource-state`, so each cloud and distributed response carries the cloud GPU state.
   - This host's WSL NVML has crashed before. The server probes NVML in a separate process first; if the probe fails, it reports host memory only, with `nvml_available: false`, instead of crashing.
   - If the units were installed for G2, copy the updated `adaptive-vlm-cloud.service` again, then run:

```bash
systemctl --user daemon-reload && systemctl --user restart adaptive-vlm-cloud adaptive-vlm-iperf adaptive-vlm-echo
```

```bash
journalctl --user -u adaptive-vlm-cloud -n 50
```

   The journal must show `"reports_resource_state": true` and both `raw_image` and `projected_synchronous` in `supported_paths`.
4. **Keepalive:** keep a `sleep infinity` open in a cloud WSL terminal. Preflight checks for it when `require_cloud_wsl_keepalive` is set.

### 3. Network

- **Addresses and ports:** check the LAN IPs, and the Windows portproxy and firewall rules for TCP 5050 (inference), 5051 (iperf3) and 5052 (echo). If the cloud address changed, update the address in three places, then commit, push and pull on the cloud:
  - `cloud_host` in `configs/profiles/physical_network_grid_v2.json`;
  - `cloud_host` in `configs/profiles/pilot_grid_v2.json`;
  - `transport.host` in `configs/experiments/physical_vqa_v2.yaml`.
- **Top rate:** check that the uplink sustains 200 Mb/s in both directions, unshaped. If either direction falls outside ±20%, set `"rate_fallback": true` in both settings files (step 5). That applies the protocol's predeclared T08 → 150/5 and E04 → 140/50.

```bash
iperf3 -c CLOUD_IP -p 5051 -t 8 && iperf3 -c CLOUD_IP -p 5051 -t 8 -R
```

- **Baseline RTT:** nothing to record by hand. Preflight measures the unshaped echo RTT once, stores it in `network_baseline.json`, and reuses it on resume.

### 4. Correctness reference

Preflight and the profiles need a same-device reference for the v2 manifest, made with `scripts/campaign/build_g2v2_reference.py` on the frozen commit. The G2 bundle doesn't work: the source and the prompt changed.

| Where | Command | Time |
| --- | --- | --- |
| edge | `.venv-model/bin/python scripts/campaign/build_g2v2_reference.py same-device --output artifacts/g2v2/reference/edge` | ~1 h (resident weights spill on 4 GiB; can run before travelling if the commit is final) |
| edge | `.venv-model/bin/python scripts/campaign/build_g2v2_reference.py runtime --output artifacts/g2v2/reference/edge-runtime.json` | seconds |
| cloud | `.venv-model/bin/python scripts/campaign/build_g2v2_reference.py same-device --output artifacts/g2v2/reference/cloud` | ~10–15 min (stop `adaptive-vlm-cloud` first to free the GPU, then restart it) |
| cloud | `.venv-model/bin/python scripts/campaign/build_g2v2_reference.py runtime --output artifacts/g2v2/reference/cloud-runtime.json` | seconds |
| edge | `.venv-model/bin/python scripts/campaign/build_g2v2_reference.py control --output artifacts/g2v2/reference/control` (cloud service running, no shaping) | ~15–20 min |

Copy the cloud's `artifacts/g2v2/reference/cloud` folder and `cloud-runtime.json` to the same paths on the edge, with `scp`. Then bind and verify everything:

```bash
.venv-model/bin/python scripts/campaign/build_g2v2_reference.py bundle --output artifacts/g2v2/reference/bundle --edge artifacts/g2v2/reference/edge --cloud artifacts/g2v2/reference/cloud --control artifacts/g2v2/reference/control --edge-runtime artifacts/g2v2/reference/edge-runtime.json --cloud-runtime artifacts/g2v2/reference/cloud-runtime.json
```

The bundle step verifies everything exactly as preflight will, and prints the index path. Any same-device mismatch stops it. Don't edit the evidence; find the cause.

Two places point at the bundle:
- the settings (step 5), in `same_device_reference_evidence`;
- `configs/predictor/g2v2_dataset_contract.json`, in `same_device_reference_evidence`. This is used by the derived tables. Commit it, push, and pull on the cloud.

### 5. Settings files

Keep machine-specific settings in ignored files:

```bash
mkdir -p artifacts/g2v2 && cp configs/campaigns/g2v2_pilot.example.json artifacts/g2v2/pilot-settings.json && cp configs/campaigns/g2v2_physical.example.json artifacts/g2v2/settings.json
```

In both files, set:
- `cloud_ssh`, `cloud_repo` and `cloud_python`: absolute Linux paths on the cloud;
- `cloud_interface`;
- `same_device_reference_evidence`: the bundle index from step 4;
- `rate_fallback`: from step 3.

Leave the rest unless there is a reason. The main fields are:

| Field | Meaning |
| --- | --- |
| `edge_residency` | `swap`, so edge/full fits in 4 GiB. Preflight requires 3,400 MiB of free VRAM. |
| `max_restarts_per_subblock` | How many times a sub-block is resumed after a CUDA context failure (exit 75) or a dead load generator (exit 76). Default 3. |
| `max_subblock_attempts` | How many times an *invalid* sub-block is moved to `invalid/` and rerun before the campaign stops. Default 2. |
| `stall_seconds` | The child is stopped if no attempt is committed for this long. Allow for an edge model load of about 90 s. |
| `min_free_gb`, `min_*_host_available_ram_mib` | Disk and host-memory reserves, checked at startup and during the run. |
| `edge_dedicated_limit_mib`, `spill_margin_mib` | The L0 no-spill check: Windows dedicated usage must stay 128 MiB below 3,960 MiB. |

`allow_unfrozen_protocol` is `true` only in the pilot settings. The real campaign refuses to start until the protocol is frozen (step 7).

### 6. Pilot, then discard it

The pilot grid runs 100/5, 100/100, 150/150, 35/20 and 200/5, each at L0 and L2, with one repetition over 10 training IDs. That is 10 sub-blocks and 300 attempts, about 1–1.5 h. It never measures validation or evaluation IDs.

```bash
.venv-model/bin/python scripts/campaign/run_g2v2_campaign.py preflight --settings artifacts/g2v2/pilot-settings.json
```

```bash
.venv-model/bin/python scripts/campaign/run_g2v2_campaign.py start --settings artifacts/g2v2/pilot-settings.json
```

When `status` reports `completed`, build the pilot's oracle table:

```bash
.venv-model/bin/python scripts/campaign/derive_g2v2_tables.py profiles --campaign artifacts/g2v2/pilot --contract configs/predictor/g2v2_dataset_contract.json --output artifacts/g2v2/pilot-derived
```

```bash
.venv-model/bin/python scripts/campaign/derive_g2v2_tables.py oracle --campaign artifacts/g2v2/pilot --contract configs/predictor/g2v2_dataset_contract.json --output artifacts/g2v2/pilot-derived
```

**Go/no-go.** If any of these fails, stop. Fix the cause and record a dated amendment in the protocol before collecting anything.
- **Valid sub-blocks:** every pilot sub-block is valid (`subblocks/*/validation_report.json`), which covers zero netem drops, probes within tolerance and no L0 spill.
- **Every placement can win:** each of the three placements wins at least one pilot cell (`oracle-training/oracle_cells.csv`).
- **L1 position:** the median cloud latency at 35/20 (cell `P04-L0`) is between 1.6 and 2.6 s. Otherwise, re-choose L1 and L2 from the [calibration curve](../../results/edge-load-calibration-20261116/README.md) (protocol section 6).
- **Cloud state:** the cloud resource state is present. A cloud row in `subblocks/*/run/cloud/requests.jsonl` has `server_state_gpu_temperature_c`, or `server_state_nvml_available: false` if the probe failed.
- **Timing:** record the per-sub-block wall time, the edge model load time (`Loading verified shared edge model … completed in` in `campaign.log`), and the probe and shaping time.

### 7. Freeze the protocol

In `docs/protocols/g2-v2-protocol.md`:
1. Replace the run-length estimate with the pilot's measurements.
2. Decide the open section 9 choice (lowercased primary VQA score).
3. Change the status to frozen.

Then pin its hash in the grid, and commit, push and pull on the cloud:

```bash
sha256sum docs/protocols/g2-v2-protocol.md
```

Put the hash in `protocol_sha256` in `configs/profiles/physical_network_grid_v2.json`. Preflight refuses the physical grid until this matches.

### 8. Launch

Before launching:
- AC power on both hosts, with sleep and hibernate off, the lid action set to do nothing, and Windows Update paused.
- Heavy GPU applications closed.
- Remote access from the third device tested.

Then:

```bash
.venv-model/bin/python scripts/campaign/run_g2v2_campaign.py preflight --settings artifacts/g2v2/settings.json
```

```bash
.venv-model/bin/python scripts/campaign/run_g2v2_campaign.py start --settings artifacts/g2v2/settings.json
```

Preflight runs each time, including on resume. It checks:
- the same clean commit on both hosts, `sudo tc`, the cloud ports and health;
- free disk and VRAM, the reference evidence and runtime versions, and the frozen protocol hash;
- the full test suite with zero skips, and the three-placement GPU correctness smoke.

`start` detaches and logs to `artifacts/g2v2/physical-unattended/campaign.log`. Watch the first two sub-blocks finish and their `validation_report.json` show `"valid": true` before leaving.

### 9. Monitor

```bash
.venv-model/bin/python scripts/campaign/run_g2v2_campaign.py status --settings artifacts/g2v2/settings.json
```

```bash
tail -F artifacts/g2v2/physical-unattended/campaign.log
```

`status` reports:
- `state` and `stage` (`sub-block i/102 attempt n`), and the current `subblock`;
- measured attempts out of 14,580, and status counts;
- `valid_subblocks`;
- `supervisor_running`, read from the live lock rather than from the status file.

A heartbeat line appears every 30 s while a sub-block runs. Each finished sub-block writes `subblocks/<id>/validation_report.json`.

### 10. Stop, recover and resume

```bash
.venv-model/bin/python scripts/campaign/run_g2v2_campaign.py stop --settings artifacts/g2v2/settings.json
```

The supervisor stops its benchmark (which stops the load generator) and removes netem from both hosts. Wait for `supervisor_running: false`, then check `cleanup_errors`. To resume, rerun the same `start` command: finished sub-blocks are re-validated against their hashes and skipped, and the interrupted one resumes from its last committed attempt.

What happens on its own, and what needs you:

| Event | Automatic handling | Your action |
| --- | --- | --- |
| CUDA context failure on the edge | Attempt recorded; benchmark exits 75; sub-block resumed (up to `max_restarts_per_subblock`) | None, unless restarts run out |
| Load generator died or stalled | Benchmark exits 76 before the next attempt; resumed with a new generator | None, unless restarts run out |
| Sub-block fails a check (drops, probes, spill, failures > 1%, dead generator) | Moved to `invalid/<id>-<time>` and rerun whole | If it fails twice the campaign stops; read `reasons` in the report |
| No committed attempt for `stall_seconds` | Child process tree stopped; campaign fails | Check the cloud service and network, then `start` |
| Supervisor killed (OOM, `kill -9`) | Its children exit with it; any leftover process is stopped (`stray_processes.jsonl`) | `start` |
| Power loss or reboot | Nothing runs | Restart cloud WSL and services, check `tc qdisc show dev eth0` on both hosts, then `start` |

Netem left by a hard kill is replaced on resume, because `shaping_active.json` marks it as the campaign's own. A qdisc the campaign didn't create is never overwritten; remove it with `sudo tc qdisc del dev eth0 root` after checking what it is. Don't delete lock files, `shaping_active.json` or any journal to force progress.

### 11. After the run

1. Build the Predictor V2 partitions and the training measured-oracle table:

```bash
.venv-model/bin/python scripts/campaign/derive_g2v2_tables.py all --campaign artifacts/g2v2/physical-unattended --contract configs/predictor/g2v2_dataset_contract.json --output artifacts/g2v2/derived
```

   - Validation targets are sealed under `/tmp` with no read permission.
   - Evaluation cells stay closed until the selector is frozen; then use `--splits evaluation --unseal-evaluation`.
2. Archive the campaign folder, the reference bundle and the cloud service journal under `results/physical-g2v2-<date>/` with `SHA256SUMS`, following the G2 layout.
3. Then follow "After the run" in the [preparation notes](../g2-v2-preparation.md#after-the-run).

### Output layout

`artifacts/g2v2/physical-unattended/` contains:
- `campaign_plan.json`: the frozen contract, including the sub-block order. A resume with changed inputs is refused.
- `network_baseline.json`, `host_preflight.json` and `preflights/`: the correctness smoke.
- `subblocks/<cell>-r<rep>/`, holding:
  - `plan.json`;
  - `network_pre.json` and `network_post.json`;
  - `qdisc_before.json` and `qdisc_after.json`, with the drop counters;
  - `run/`, with `interleave.json` and one folder per placement (`plan.json`, `run_manifest.json`, `requests.jsonl`, `summary.json`);
  - `.run-edge-load/`, the load generator's heartbeat and sessions;
  - `restarts.jsonl` and `validation_report.json`.
- `invalid/`, `stray_processes.jsonl`, `status.json` and `campaign.log`.
