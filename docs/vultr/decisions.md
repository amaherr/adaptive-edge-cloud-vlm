# Vultr decisions and frozen baseline

## P0 — 7 November 2026

Use the existing `vultr-cloud-backup` branch. Its starting commit is
`621e5127d03ae7e0ee89f893e61cc464465e54f4`, identical to local and remote
`cloud-infrastructure`. Main is `22567a67b2b2a1a828a4a7d2d61e0e0c277cc36f`.
Remote references were refreshed with `git fetch --all --prune`.
The only starting working-tree change was the owner's untracked
[roadmap](../Vultr_Cloud_Backup_Agent_Roadmap.md), preserved verbatim.
No branch switch, cherry-pick, AWS state operation or paid launch is necessary.

The existing Vultr branch inherits ten AWS development commits, from `21f47b3`
to `621e512`; no new AWS resources will be imported into a Vultr environment.
All existing AWS Terraform and physical evidence remain in place. Terraform
state for Vultr must be separate from AWS and from disposable compute.

## Reuse inventory

- `container/Dockerfile`: pinned Python 3.11 base, CUDA wheel lock, non-root
  application, build provenance and dependency verification; reuse one image.
- `container/verify_image.py`, `container/verify_cache.py`,
  `scripts/cloud_container_smoke.py`: reusable acceptance checks, subject to
  validating AWS assumptions at their call sites.
- Compose files currently reference ECR, AWS campaign config and `10.42.1.20`;
  they require explicit provider-neutral configuration before Vultr execution.
- `src/adaptive_vlm/benchmark/runner.py` and `artifacts.py`: strict local plan
  resume and fsynced journals. `7f964ef` supplies graceful SIGTERM handling.
  No remote S3 checkpoint implementation exists: the roadmap's suggestion to
  reuse an existing S3 implementation is not supported by this checkout.
- `scripts/apply_cloud_network_block.py`: peer/port classifier shaping is a
  useful shared foundation; actual Vultr interfaces, routes and both peer
  qdiscs still require measurement.
- `scripts/run_physical_profile_block.py`: deterministic fixed policy schedule
  and independent probes; hard-coded physical host pair, WSL provenance and
  qdisc assumptions must be parameterized without changing physical defaults.
- AWS ECR publication, EC2 host provisioning, IAM, budgets and Terraform wrapper
  are provider-specific and must not be used to provision Vultr.

## Frozen inputs (SHA-256)

| Path | SHA-256 |
| --- | --- |
| `configs/model/moondream2.yaml` | `831f7255a11c5be66816e821497e2f54cb1c114ca34bfe8e043c5f95174f2d09` |
| `configs/model/moondream2_artifact.yaml` | `369640ff5a46c7f11c6b3f20a22f780d9ae4501e34db80fe2eaa11263df3a650` |
| `requirements/model-cu128.lock` | `96d07ba6bb6310b29d259c64652de18f4c5f72f2f0629a17844d09ffacedffbd` |
| `data/manifests/physical_vqa_v1.jsonl` | `4e62826280e49359e8cf98d85154000c1583d63474be83a2e9a429da9bde72c9` |
| `configs/evaluation/physical_vqa_split_v1.json` | `608e06c06e3916b79e33eda400f324a6bc9d2db73ccc6549fa1214277945e3cc` |
| `configs/profiles/physical_network_blocks_v1.json` | `a2818dc67744b7e196a25b061f2ef908fc9ad5d5424659800efdf7f83b87b16c` |

Keep BF16, greedy decoding, 32 new tokens, the same 100 requests and annotations,
60/20/20 split, N4/N9 final block holdouts, and three fixed policies. A Vultr
hardware pair starts a new campaign. Gate 2 is 8,100 measured attempts plus
27 excluded warmups; historical physical results cannot establish Vultr gates.

## P1 region preference

The owner selected Atlanta, USA (`atl`). Vultr documents this region code in
its [official guide](https://docs.vultr.com/how-to-establish-private-connection-between-vultr-regions-using-headscale).
This preference does not establish GPU inventory, account entitlement, two-node
concurrency, prices or launch approval. Use `--region atl` for P1 discovery.

## P1 budget and alert destination

The owner targets approximately **$80 total** and selected
**a.maher1506@gmail.com** for alerts. Treat $80 as a planning limit until a
concrete launch quote is approved. The owner is setting up API credentials and
will notify the agent. No email was sent or alert integration activated.

The public 7 November API snapshot contains no Atlanta GPU inventory. Do not
substitute marketing availability for this response. See the updated
[cost analysis](cost_estimate.md) for conditional runtime sensitivity, storage
retention costs, the proposed $5 preflight and proposed alert/teardown thresholds.
None of those proposed allocations or launches is yet approved.

## Corrected provider discovery and approved operations

The owner approved reusing S3, the $5 single-node preflight, alert/checkpoint/
teardown controls and destruction of idle campaign VMs. AWS access was refreshed;
S3 durability acceptance is recorded in [checkpoints](checkpoints.md).

The owner correctly reported Atlanta GPU availability in the portal. The prior
`type=vcg` filter was incomplete: A16/A40 `vcg-` SKUs are classified `vdm` by the
current API. Use all-type discovery with local GPU identification. Public
all-type listing confirms Atlanta A16 candidates at $0.471/$0.942 per hour;
exact per-device layout and stronger cloud execution remain unverified. Earlier
empty filtered evidence is preserved but superseded as a capacity conclusion.
