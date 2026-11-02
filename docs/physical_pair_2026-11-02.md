# Physical edge/cloud verification — 2026-11-02

## Hosts and contract

- Edge: Windows Wi-Fi `192.168.8.71`, WSL project at `/mnt/d/Programming/adaptive-edge-cloud-vlm`.
- Cloud: Windows Wi-Fi `192.168.8.70`, WSL server on TCP 5050 through Windows portproxy.
- Both repositories were at Git commit `ee29a81` for the first physical requests.
- The cloud Windows Public firewall permits inbound TCP 5050 only from `192.168.8.71`.
- Model artifact SHA-256: `70a7d94c0c8349eb58ed2d9e636ef2d0916960f321ecabeac6354b8ba3d7403f`.
- Model contract fingerprint: `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`.

## Completed checks

The cloud server loaded the pinned Moondream2 model offline on `cuda:0` and advertised `raw_image` and `projected_synchronous`. The cloud experiment configuration validated.

| Path | Result | Evidence |
| --- | --- | --- |
| `cloud/raw_image` | PASS | Edge request `adaptive-0001`, run `20261102T085442Z-f40eb341`, 32 output tokens, cloud projected source and target `cuda:0`. |
| `distributed/projected/synchronous` | PASS | Edge request `adaptive-0001`, run `20261102T090644Z-9aa66222`, 32 output tokens. Edge sent a BF16 `[729, 2048]` tensor (2,985,984 payload bytes) from CPU vision; cloud GPU handled text. |

The two completed paths returned identical decoded text and identical 32 token IDs for `tests/fixtures/moondream_smoke.ppm` with prompt `Describe this image.`. The physical split request transferred 2,986,419 bytes to the server and completed successfully.

## Remaining verification

- Run `edge/full` on the same image and compare its output token IDs with the two completed paths.
- Run `scripts/verify_phase1.py` on the edge with `data/manifests/correctness.jsonl` and inspect `artifacts/phase1/physical-v1/phase1_smoke.json` for all 12 request triples.
- Treat the physical gate as open until all three paths and the full manifest have been checked.

These results are from edge CPU vision and cloud GPU inference. A later edge GPU vision run is a separate numerical placement and needs its own equivalence check.
