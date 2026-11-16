# G2 v2 on-site checklist

Print this page. The step numbers match the [runbook's G2 v2 section](physical-campaign.md#g2-v2-campaign), which has the exact commands. **(commit)** marks a step that changes tracked files: commit it on the edge, push, and pull on the cloud before the final preflight, because preflight refuses mismatched or dirty trees.

## Before travelling (edge laptop)

- [ ] All code changes committed and pushed. The reference binds the source, so no `src/` changes after this point. *(Runbook 1)*
- [ ] Ruff, mypy and pytest pass. GitHub Actions is billing-locked, so this local run stands in for CI.
- [ ] `rehearse_g2v2_faults.py --fresh` passes every check. *(1)*
- [ ] Optional, saves about 1.5 h on site: edge `same-device` and `runtime` reference steps for both tasks on this commit. *(4)*
- [ ] At least 15 GB free on C:. `%TEMP%\DiagOutputDir` held 20.7 GB.
- [ ] The commit, the model cache and the frozen images are on both hosts.

## Cloud PC *(2)*

- [ ] Same commit pulled; `.venv-model` ready; pinned model cache present.
- [ ] `ssh CLOUD_ALIAS true` works from the edge into the cloud's Linux. Passwordless `sudo -n tc` works on both hosts.
- [ ] Updated `adaptive-vlm-cloud.service` installed; all three services restarted.
- [ ] The journal shows `"reports_resource_state": true` and both supported paths.
- [ ] `sleep infinity` open in a cloud WSL terminal.

## Network *(3)*

- [ ] LAN IPs, portproxy and firewall rules checked for TCP 5050, 5051 and 5052.
- [ ] If the cloud IP changed: both grids and `physical_vqa_v2.yaml` updated **(commit)**.
- [ ] Unshaped iperf3 in both directions, 200 Mb/s ± 20%. If not: `rate_fallback: true` in both settings files.
- [ ] (The unshaped RTT baseline needs nothing from you; preflight records it.)

## Correctness reference *(4)*

- [ ] Edge `same-device` (about 1 h) and `runtime` done.
- [ ] Cloud `same-device` (service stopped; restart it afterwards) and `runtime` done; copied to the edge.
- [ ] Edge `control`, about 15–20 min, cloud service running, no shaping.
- [ ] Repeat all four steps for the long-answer task: `--config configs/experiments/physical_vqa_v2_long.yaml` and `reference-long/` paths (about 30 min edge, 5 min cloud, 15 min control).
- [ ] `bundle` passes and prints the index path.
- [ ] Both index paths (short and long) set in `g2v2_dataset_contract.json` **(commit)** and in both settings files.

## Settings *(5)*

- [ ] `artifacts/g2v2/pilot-settings.json` and `artifacts/g2v2/settings.json` copied from the examples.
- [ ] Filled in: `cloud_ssh`, `cloud_repo`, `cloud_python`, `cloud_interface`, `same_device_reference_evidence` and `rate_fallback`.

## Pilot, then discard it *(6)*, about 1.5 h

- [ ] Pilot `preflight`, then `start`, then `status` until `completed`.
- [ ] `derive_g2v2_tables.py profiles`, then `oracle`, on the pilot.
- [ ] **Go/no-go.** If any of these fails: stop, fix, and record a dated protocol amendment.
  - [ ] All 10 pilot sub-blocks valid: no netem drops, probes within tolerance, no L0 spill.
  - [ ] Each of edge, cloud and distributed wins at least one pilot cell (distributed is expected at `P05-L1-long` or `P01-L1-long`).
  - [ ] Cloud median at P04-L0 (35/20) is between 1.6 and 2.6 s. Otherwise re-choose L1 and L2 (protocol section 6).
  - [ ] Cloud rows carry `server_state_*` (or `server_state_nvml_available: false`).
  - [ ] Recorded: wall time per sub-block, edge model load per sub-block, and probe and shaping time.

## Freeze the protocol *(7)* **(commit)**

- [ ] Run length replaced with the pilot's numbers.
- [ ] Section 9 lowercase decision made.
- [ ] Status set to frozen.
- [ ] `sha256sum` of the protocol put in `protocol_sha256` in `physical_network_grid_v2.json`.
- [ ] Committed, pushed and pulled on the cloud.

## Hosts *(8)*

- [ ] AC power on both hosts; sleep and hibernate off; lid action set to do nothing; Windows Update paused.
- [ ] GPU applications closed; display on the integrated GPU if possible.
- [ ] Remote access from the third device tested.

## Launch *(8–9)*

- [ ] `preflight` passes with `artifacts/g2v2/settings.json`.
- [ ] `start` launched.
- [ ] First two sub-blocks valid, with `"valid": true` in `validation_report.json`.
- [ ] `status` shows the supervisor running and attempts climbing.

| Time | Sub-block | Measured / 17,550 | Notes |
| --- | --- | --- | --- |
| | | | |
| | | | |
| | | | |
