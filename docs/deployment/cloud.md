# All-cloud deployment

## Localhost smoke test

Use `configs/experiments/chunk4_local_cloud.yaml` and the two commands in the
README. The edge client transmits the original encoded image bytes and prompt;
only the server loads and runs Moondream. Stop the server with Ctrl+C after the
request completes.

For a fast transport-only smoke test, use the fake configuration on both
commands:

```bash
.venv/bin/adaptive-vlm cloud-server \
  --config configs/experiments/chunk4_fake_cloud.yaml

.venv/bin/adaptive-vlm cloud-infer \
  --config configs/experiments/chunk4_fake_cloud.yaml \
  --image tests/fixtures/smoke_image.txt \
  --prompt "Describe this fixture."
```

## Two physical hosts

The repository does not yet contain the cloud machine inventory. Before a
physical-host run:

1. Check out the same commit on edge and cloud, create the locked Python 3.11
   model environment on the cloud, and place the verified pinned model in its
   local cache.
2. Replace `configs/hardware/cloud_pending.yaml` with measured hardware and
   software details and set its status to `verified`.
3. Copy `chunk4_local_cloud.yaml` to a new experiment file. Reference the
   verified cloud hardware file and set `transport.host` to the cloud address
   reachable from the edge. Keep the same port, protocol version, model file,
   and generation settings on both hosts.
4. Permit inbound TCP port 5050 only from the edge address, or use an SSH/VPN
   tunnel. The baseline protocol provides integrity and bounds checks, but no
   encryption, authentication, or authorization. Do not expose it directly to
   the public internet.
5. On the cloud, bind the server to the required interface while retaining the
   reachable address in the shared experiment config:

   ```bash
   .venv-model/bin/adaptive-vlm cloud-server \
     --config configs/experiments/physical_cloud.yaml \
     --artifact-manifest configs/model/moondream2_artifact.yaml \
     --cache-dir .model-cache \
     --offline \
     --device cuda:0 \
     --bind-host 0.0.0.0
   ```

6. Confirm that the readiness JSON reports the expected contract fingerprint,
   then run `cloud-infer` from the edge with the same experiment file. Preserve
   the generated manifest and request record.

For CPU cloud execution, replace `--device cuda:0` with `--device cpu`.
Restrict each server process to one experiment contract: changing model,
precision, preprocessing, or generation settings produces a different
fingerprint and is rejected by the peer.

## Operational behavior

- Framing uses a four-byte network-order header length, bounded JSON metadata,
  a bounded binary payload, and a SHA-256 payload checksum.
- The server processes requests sequentially and accepts one framed request per
  TCP connection. This is intentional for the batch-size-one latency baseline.
- Socket timeouts, truncated frames, oversized frames, invalid schemas,
  protocol-version mismatches, and contract mismatches are rejected safely.
- A failed model request returns a structured non-retryable error while the
  server remains available for later requests.
- Client E2E time and exact request/response wire bytes are recorded alongside
  server model-component and memory metrics.
