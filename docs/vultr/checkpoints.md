# S3 checkpoint implementation and acceptance

The owner selected the existing S3 bucket and approved implementing remote
checkpoints, alerts, teardown and a $5 single-node preflight. GPU launch remains
blocked: authenticated Atlanta discovery at 11:51 UTC on 7 November 2026 returned
no available GPU plans. AWS access was refreshed; the artifact bucket is
accessible, AES256-encrypted and versioned. The temporary Vultr key handoff file
was removed after successful authentication; no key was logged or committed.

## Supported path

`adaptive-vlm benchmark --checkpoint-config PATH` adds S3 durability to the
existing benchmark, without changing its schedule, request rows or resume CLI.
The JSON settings file supplies `bucket`, optional AWS CLI `profile`, `prefix`
and `contract`. Use an independent prefix for every campaign/block/policy/run.
A campaign contract includes `campaign_id`, `block_id`, `policy`, `host_pair_id`,
`provider`, immutable `image_digest`, and SHA-256 values for `model_sha256`
(the model contract fingerprint), `dataset_sha256`, `split_sha256`, `grid_sha256`
and `hardware_contract_sha256`. Values must describe actual verified inputs;
placeholders are rejected. A changed contract cannot resume old evidence.

Requires AWS CLI supporting conditional PutObject and DeleteObject operations
(tested with **2.39.4**). The research container does not yet include this CLI;
a pinned CLI installation or a reviewed host/controller integration is mandatory
before enabling this path inside Docker. AWS credentials belong in a scoped
runtime credential source, never config JSON, Terraform state or the image.

The runner checks S3 before scheduling a trial and uploads after inference
and local fsync, outside the measured latency. It uploads every 25 rows or
300 seconds between trials, plus initial/final checkpoints and orderly stop.
No background network transfer is introduced during timed inference. A single
long-running trial may exceed the target time interval; its active inference
is not interrupted to upload. Upload failures stop further scheduling.

Segments, plan/manifest/summary snapshots and chain-linked commit manifests
use SHA-256 object keys and conditional create. Uploads are read back before
committing. Only a conditional `head.json` update advances authoritative state;
orphan uploads remain ignored. A persistent conditional `writer.json` lock
rejects a second writer. It never expires automatically, avoiding split-brain
writers after a suspended host returns.

Recovery validates contract, commit sequence, every referenced SHA-256, full
journal hash/count and unique trial keys before creating a fresh directory:

```bash
.venv/bin/python scripts/vultr/checkpoint_control.py restore \
  --config CHECKPOINT_CONFIG.json --destination artifacts/restored-run
# Then reuse the exact original benchmark arguments and add --resume.
```

After a crash, first prove the old writer has stopped or its VM was destroyed.
Inspect the lock object's current ETag; release only that exact lock with:

```bash
.venv/bin/python scripts/vultr/checkpoint_control.py release-stopped-writer \
  --config CHECKPOINT_CONFIG.json --old-writer-confirmed-stopped \
  --expected-lock-etag 'EXACT_ETAG'
```

A restored older journal never overwrites a newer local journal: restore requires
a new directory. Reconcile unuploaded local tails explicitly. The restore
retains original plan/run identity so existing `--resume` can rebuild a missing
summary without rerunning committed trials. Remote `PREFIX/STOP` prevents new
trials after forcing a checkpoint. Treat STOP as a permanent session stop until
reviewed; the controller does not silently clear it.

## Validation performed

- Offline checkpoint/runner/CLI regression tests: **18 passed**. Tests cover
  failed segment/commit/head uploads, corruption, changed image/model/hardware/
  dataset/grid, concurrent writers, stale lock release, 5-minute cadence,
  remote STOP, uncommitted tails, rollback refusal, final-summary regeneration,
  local journal retention on upload failure, restore and existing resume.
- Live S3 conditional create/CAS/delete, stale-condition rejection and readback:
  **passed**, under a new diagnostic prefix only. The probe object was deleted;
  older versions may remain under bucket versioning. No retained data touched.
- Live CPU **fake-adapter** run, immutable upload, byte-matching restore and
  existing resume: **passed**, 12 diagnostic measured rows plus one warmup.
  The acceptance contract uses explicitly synthetic image/holdout references;
  this is not research, real-model or GPU evidence.
- Reports: `artifacts/vultr/p5/s3-api-acceptance.json` and
  `artifacts/vultr/p5/live-roundtrip-001/report.json`; copied nonsecret summaries
  under `docs/vultr/evidence/` for review.

P5 is **not fully passed**: real container SIGKILL, live GPU VM destruction/
recreation, complete hardware provenance, timing interference pilot and deployed
control-plane failure handling remain unperformed. Do not describe the Vultr
backup as READY based on these storage tests.
