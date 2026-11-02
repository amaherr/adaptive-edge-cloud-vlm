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
| `edge/full` | PASS for execution; different output | Corrected-prompt edge request `adaptive-0001`, run `20261102T094044Z-29e38feb`, 32 output tokens on CPU. Its tokens match the CPU edge row for `correctness-001` and diverge from the cloud GPU output at token 20 (one-based). |

The two completed paths returned identical decoded text and identical 32 token IDs for `tests/fixtures/moondream_smoke.ppm` with prompt `Describe this image.`. The physical split request transferred 2,986,419 bytes to the server and completed successfully.

## Twelve-request Phase-1 campaign

The edge ran `scripts/verify_phase1.py` on the frozen `data/manifests/correctness.jsonl` manifest at commit `ee29a81`. All 36 inferences returned output tokens. Eleven of 12 request triples have exactly equal token IDs and text. For `correctness-001`, `cloud/raw_image` and `distributed/projected/synchronous` match exactly, while `edge/full` diverges at token 20. The script wrote `complete_and_equal: false` and raised `RuntimeError: Phase 1 outputs differ`. The raw edge report was copied to `artifacts/phase1/physical-v1/phase1_smoke.json` on the cloud host; this path is ignored by Git.

The same divergence occurs on the corrected single-image smoke request. This supports a numerical difference between CPU edge inference and GPU cloud inference, rather than a transport error. The cloud GPU reports about 4,550 MiB peak allocated for both cloud paths. The model contract fingerprint and artifact SHA-256 match across the hosts.

The physical network and split transport checks pass. The strict three-path token-equivalence gate remains open at 11/12. A CPU cloud reference run would isolate the hardware-placement effect; any later edge GPU placement would need its own equivalence check.
