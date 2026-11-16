# Unattended physical campaign runbook

> **G2 v2** uses its own supervisor, `scripts/campaign/run_g2v2_campaign.py` (see [G2 v2 campaign](#g2-v2-campaign) at the end). The sections before it describe the G2 v1 campaign, kept so that G2 can be reproduced.

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

The G2 v2 run is driven by `scripts/campaign/run_g2v2_campaign.py` with a settings file. Start from [g2v2_physical.example.json](../../configs/campaigns/g2v2_physical.example.json). The design is in the [protocol](../protocols/g2-v2-protocol.md), sections 5–8. W9 will finish this section with the on-site checklist.

**Rehearse on one machine first** (fake model, no shaping, real GPU load worker):

```bash
.venv-model/bin/adaptive-vlm adaptive-server --config configs/experiments/fake_g2v2_rehearsal.yaml --artifact-manifest configs/model/moondream2_artifact.yaml --bind-host 127.0.0.1 --report-resource-state
```

```bash
.venv-model/bin/python scripts/campaign/run_g2v2_campaign.py run --settings configs/campaigns/g2v2_loopback_rehearsal.json
```

**Physical run.** Copy the example settings, fill in the cloud SSH target, repository, Python, interface and the regenerated reference evidence, and decide `rate_fallback` from the pilot. Then:

```bash
.venv-model/bin/python scripts/campaign/run_g2v2_campaign.py preflight --settings SETTINGS.json
```

```bash
.venv-model/bin/python scripts/campaign/run_g2v2_campaign.py start --settings SETTINGS.json
```

```bash
.venv-model/bin/python scripts/campaign/run_g2v2_campaign.py status --settings SETTINGS.json
```

`stop` sends SIGTERM to the supervisor, which removes netem from both hosts before exiting. Rerunning `start` resumes: finished sub-blocks are re-validated against their recorded hashes and skipped.

The output folder (`output_root`) contains:
- `campaign_plan.json`: the frozen contract, including the full sub-block order. A resume with changed inputs is refused.
- `network_baseline.json`: the unshaped baseline.
- `subblocks/<cell>-r<rep>/`, holding:
  - `plan.json`;
  - `network_pre.json` and `network_post.json`;
  - `qdisc_before.json` and `qdisc_after.json`, with the drop counters;
  - `run/` (`interleave.json` plus one folder per placement);
  - `restarts.jsonl`;
  - `validation_report.json`.
- `invalid/`: sub-blocks that failed a gate and were rerun.
- `status.json` and `campaign.log`.

**Derived tables.** After the campaign, build the Predictor V2 partitions and the training measured-oracle table:

```bash
.venv-model/bin/python scripts/campaign/derive_g2v2_tables.py all --campaign artifacts/g2v2/physical-unattended --contract configs/predictor/g2v2_dataset_contract.json --output artifacts/g2v2/derived
```

Validation targets are written under `/tmp` with no read permission. Evaluation cells are opened only with `--splits evaluation --unseal-evaluation`, after the selector is frozen.

