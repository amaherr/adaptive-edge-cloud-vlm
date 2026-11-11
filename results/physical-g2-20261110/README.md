# Completed physical G2 baseline — 10–11 November 2026

All **8,100 measured requests succeeded**, plus 27 warmups. All nine controlled network blocks passed the user-approved `physical-heterogeneous-v2` criterion. The campaign completed at **2026-11-11T05:20:58+00:00** (07:20:58 Cairo), in **15.20 hours** including preflight. Source commit: `3c38be2a68d902dbbf286621a593832aa2149978`.

- [summary.json](summary.json): completion, aggregate placement timings, counts and correctness outcomes.
- [block_summary.csv](block_summary.csv): latency, quality and framed-byte summaries for all 27 block/placement combinations.
- [evidence.tar.gz](evidence.tar.gz): original request journals, plans, manifests, profiles, GPU samples, pre/post network measurements, validation reports, preflight logs and hash-bound same-device correctness references.
- [archive_inventory.json](archive_inventory.json): SHA-256 and size of every archived file.
- [SHA256SUMS](SHA256SUMS): archive checksum.

Verify and restore from the repository root:

```bash
(cd results/physical-g2-20261110 && sha256sum -c SHA256SUMS)
tar -xzf results/physical-g2-20261110/evidence.tar.gz
```

The archive preserves the original repository-relative artifact paths and byte content. Recorded absolute paths identify the original hosts; using another machine requires explicit path mapping. Model weights, raw dataset images, executable caches, PID and lock files are excluded. The original complete working artifacts remain locally under `artifacts/unattended/site-preflight-20261110/`.

The approved criterion requires exact same-device path parity, stable per-placement reference outputs, repeatability, matching per-request task scores, exact transport and valid timing/network evidence. **Legacy cross-GPU exact-token gates remain false** and are preserved in every block report; differing wording is not represented as identical output. The provisional VQA scores are not an official VQA evaluation or a general accuracy claim. Byte accounting covers application frames, excluding TCP/IP/Ethernet overhead, retransmissions, probes and SSH. The edge model used WDDM shared GPU memory; success does not establish dedicated-VRAM feasibility.
