# Archive (roadmap phase 8), 21 November 2026

## What is archived

**1. The whole artifact bucket**, mirrored to `artifacts/aws-s3-mirror/` and verified on 21 November 2026 ([bucket_receipt.json](bucket_receipt.json)). The bucket itself was deleted later that day with every other AWS resource, so this mirror and the tarball below are the only copies:

| Check | Result |
| --- | --- |
| Objects | 25,209 (916.8 MB) |
| Verified by MD5 | 25,199 |
| Multipart uploads, verified by size | 10 (the source tarballs) |
| Missing, wrong size or wrong MD5 | **0** |

The per-object listing is `artifacts/archive-20261121/bucket_objects.csv`, SHA-256 `objects_csv_sha256` in the receipt.

**2. The final copies of both hosts' data volumes** (`cloud-v1/{edge,cloud}/ebs-final/`), synced on 21 November 2026 after the G5 campaign. They hold every campaign folder, including the preflights and final statuses that the per-segment checkpoints leave out:
- **edge:** 10,786 files, 370 MB;
- **cloud:** 42 files.

G5's final status: `completed`, 80 of 80 segments valid, 14,400 of 14,400 measured attempts, 480 warmups ([copy](../g5-aws-20261120/campaign/final_status.json)).

**3. The off-machine backup tarball:** `artifacts/archive-20261121/aws-s3-mirror-20261121.tar.gz`.
- 265 MB, 30,825 entries;
- SHA-256 `00bcdf31…7156` ([hash file](aws-s3-mirror-20261121.tar.gz.sha256)).

## Restore test

The tarball was extracted to a temporary directory and both campaigns were checked against their committed checksum files:

| Campaign | Files checked | Failures |
| --- | ---: | ---: |
| G5 (`cloud-v1/edge/g5-campaign/` against [SHA256SUMS](../g5-aws-20261120/campaign/SHA256SUMS)) | 2,721 | 0 |
| G2 v2+ (`cloud-v1/edge/ebs-final/g2v2plus/campaign/` against [SHA256SUMS](../aws-campaign-20261119/SHA256SUMS)) | 5,921 | 0 |

## Off-machine copy (to do by hand)

**Now essential: the AWS copy no longer exists.** Copy the tarball and the committed hash file next to it ([aws-s3-mirror-20261121.tar.gz.sha256](aws-s3-mirror-20261121.tar.gz.sha256), which names the tarball without a folder) to storage off this laptop, for example an external disk or university storage. Then verify the copy there, in the folder that holds both:

```bash
sha256sum -c aws-s3-mirror-20261121.tar.gz.sha256
```

## Source identities

Annotated git tags pin every result: `g2v2plus-campaign`, `g3-r2-specification`, `g3-r2-candidate`, `g3-passed`, `g4-specification`, `g4-passed`, `g5-campaign`, `g5-freeze`, `g5-passed` and `final-analysis`. [Reproducing the results](../../docs/reproducibility.md) lists the commands per table and figure.
