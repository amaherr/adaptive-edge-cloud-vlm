# Vultr cost analysis — approximately $80 total

Updated 7 November 2026. Region: Atlanta, USA (`atl`). Alert destination:
**a.maher1506@gmail.com**. Owner approved reusing existing AWS S3, the proposed
**$5 single-node preflight**, alerts/checkpoints/teardown and destruction of idle
campaign instances. This does not authorize the full 8,100-attempt campaign.

Two bounded one-node access attempts now have an estimated combined minimum
GPU cost of **$0.942**, before tax/transfer and small AWS controls/probe costs.
The corrected retry also failed before any GPU/model execution; no measured
Vultr throughput is available. See the [retry record](preflight_retry_2026-11-07.md).
The provider invoice remains authoritative.

## Corrected GPU discovery and candidate costs

**Correction:** the initial `type=vcg` API query omitted A16/A40 GPU SKUs because
Vultr classifies those plans as **`vdm`**, despite their `vcg-` IDs. The owner
reported seeing GPUs in the Atlanta portal; querying **all** plan families
confirmed that. Earlier empty filtered responses are retained as historical
API evidence, not proof that Atlanta has no GPU capacity. The preflight now
queries all families and filters GPU fields/SKU family locally, preserving
ordinary and VPC-only availability separately.

Public all-type API evidence: [archived response](evidence/all_types_gpu_catalog_2026-11-07.json).
Sources: [plans](https://api.vultr.com/v2/plans?type=all&per_page=500) and
[Atlanta availability](https://api.vultr.com/v2/regions/atl/availability?type=all).

| Candidate SKU | vCPU / RAM | Advertised aggregate GPU memory | USD/hour | Atlanta API listing |
| --- | --- | --- | ---: | --- |
| `vcg-a16-6c-60g-16vram` | 6 / 60 GiB | A16, 16 GB | **0.471** | Present |
| `vcg-a16-12c-120g-32vram` | 12 / 120 GiB | A16, 32 GB | **0.942** | Present |
| Pair of the above | Separate compute allocations | Per-device layout unverified | **1.413** | Both listed; concurrency unverified |

Prices, CPU, RAM and locations above are explicit API fields. The A16 rows omit
explicit GPU model/VRAM/count fields beyond `gpu_brand`; the GPU-memory labels
are advertising/SKU descriptions, not measured device inventories. Vultr's
[overview](https://docs.vultr.com/product-overviews/cloud-gpu) advertises the
16 GB A16 starting configuration. Confirm exact GPU topology in the portal and
with `nvidia-smi`/PyTorch. A 32 GB aggregate plan may expose multiple 16 GB GPUs;
the application's single `cuda:0` does not pool memory or automatically benefit
from additional GPUs. More CPU/RAM alone must not be represented as a measured
single-GPU speedup. Freeze the hardware pair only after this is resolved.

The API also lists a 48 GB A40 at $1.712/hour in Los Angeles and an 80 GB A100 at
$2.397/hour in New Jersey, not Atlanta in this snapshot. L40S remains cataloged
at $1.671/hour with empty locations. Neither is a verified same-region/VPC cloud
partner for the Atlanta A16. Do not silently switch regions or relax topology.

The earlier $2.142/hour A16/L40S sensitivity model is **superseded** for the
current Atlanta candidate. None of these public listings is a stock reservation,
two-node entitlement or driver/model/network acceptance result.

## Gate 2 sensitivity at $1.413/hour

The frozen workload is **8,100 measured attempts plus 27 warmups**. No Vultr
runtime has been measured. Let `s` be mean elapsed seconds per measured attempt,
including failure/timeout duration, averaged across all blocks/policies:

```text
measured_hours = 8100 * s / 3600 = 2.25 * s
billable_pair_hours_estimate = 1.30 * measured_hours + 4
compute_estimate = billable_pair_hours_estimate * 1.413
estimated_total = compute_estimate + 7 + 12
```

Planning assumptions: 30% runtime allowance, four paired hours for setup,
preflight/pilot/probes/warmups/checkpoint validation/rounding, a **$7 one-month
ancillary allowance**, and **$12 contingency**. These are not measured guarantees.
The single-node preflight is conservatively covered inside the four paired
hours; its actual charge uses only that node's rounded-up lifetime.

| Mean attempt duration | Pair hours with allowances | Estimated total with existing S3 |
| --- | ---: | ---: |
| 5 s | 18.625 | **$45.32** |
| 10 s | 33.250 | **$65.98** |
| 15 s | 47.875 | **$86.65** |
| 20 s | 62.500 | **$107.31** |
| 30 s | 91.750 | **$148.64** |

At this assumed pair rate, $80 leaves **43.17 paired hours**, or about **13.39 s
mean per attempt** after the stated allowances. Later G5/G6 GPU campaigns,
extra reruns or longer retention consume part of that headroom; the table does
not promise full thesis completion for $80. Historical RTX single-request
measurements of 9.3 s/17.5 s do not predict an A16 mean or a GPU speedup.

## Persistent services and operational overhead

Reuse the existing independently retained AWS S3 bucket and state backend. The
bucket has been checked live for access, versioning and AES256 encryption.
New Vultr work uses only `research/adaptive-vlm/vultr-v1/` and separate state
keys `vultr-v1/operations/terraform.tfstate` and `vultr-v1/gpu/terraform.tfstate`.
No existing AWS experiment stack is changed or destroyed by this work.

| Item | Basis / allowance |
| --- | --- |
| S3 artifacts, state, requests and restores | Usage-based, included in $7 allowance; actual growth/retention still to measure |
| Existing ECR or Vultr free registry | Existing app image about 4.03 GB; verify actual new image/registry usage and internet transfer |
| Off-VM controls | Scheduled Lambda, SNS email, CloudWatch alarms/logs, SSM secret, occasional small CodeBuild Terraform teardown; included in $7 allowance, not assumed free |
| Node disks | Use capacity included with the verified plan; no extra volumes or snapshots assumed |
| Tax, FX, transfer and rounding | Draw from contingency; requote once account billing and session lifetimes are known |

Vultr's [registry plans](https://docs.vultr.com/support/products/container-registry/what-are-the-available-plans-for-vultr-container-registry)
include a free 10 GB tier, subject to availability and actual image size.
[Vultr Standard Object Storage](https://docs.vultr.com/support/platform/billing/how-is-object-storage-billed)
is $18/month; reusing existing S3 avoids that extra subscription. S3 requests,
retained versions, AWS-to-Vultr restores and image pulls can still cost money;
see [S3 pricing](https://aws.amazon.com/s3/pricing/). No credits are assumed.

The watchdog polls every five minutes using Lambda; CodeBuild starts only for
teardown. Running CodeBuild itself as the five-minute polling loop would waste
budget and is deliberately avoided. Consult [Lambda](https://aws.amazon.com/lambda/pricing/),
[CodeBuild](https://aws.amazon.com/codebuild/pricing/), [SNS](https://aws.amazon.com/sns/pricing/)
and [CloudWatch](https://aws.amazon.com/cloudwatch/pricing/) pricing. The $7 total
allowance is a planning buffer, not a measured service quote or hard cap.

## Approved controls and remaining launch prerequisites

- Single-node preflight: **at most $5**, within the $80 total. Use the smallest
  verified model-capable candidate, at $0.471/hour if the quote holds. A proposed
  two-hour initial window is $0.942 compute before rounding/extras; installing
  or downloading still accrues billable time. One preflight was launched and
  automatically destroyed after access failed; its compute estimate is $0.471.
  See [the preflight record](preflight_2026-11-07.md). Count that cost toward the
  $5 preflight allowance and the $80 total before any retry.
- Alerts at **$40/$60/$72/$80**; at **$72**, checkpoint, stop scheduling and
  request Terraform teardown. A session's own $5 ceiling/expiry/idle limit also
  triggers shutdown before the campaign threshold.
- A separate AWS control plane enforces expiry/idle checks even when SSH or the
  laptop fails. Confirm SNS email and test the deployed watcher before GPU launch.
- Record prior campaign charges and retain destroyed-node lifetimes in the
  ledger. Counts must include stopped nodes and persistent AWS service charges.
- [Server billing](https://docs.vultr.com/support/platform/billing/how-am-i-billed-for-my-servers)
  has a one-hour minimum unit and continues while stopped. Destroy idle nodes;
  never destroy independently retained checkpoints/state as compute cleanup.
- Complete authenticated all-type discovery, exact OS/image and private-network
  compatibility, operator SSH access, per-device GPU layout and a reviewed
  Terraform launch plan. Prove checkpoint/teardown on the small preflight before
  authorizing the full campaign.

**Assessment:** $80 looks more plausible with the corrected Atlanta candidate
rates, but runtime, GPU topology and durable operational controls must be tested.
The main remaining uncertainty is no longer the filtered API's apparent absence
of Atlanta GPUs; it is the exact purchasable hardware contract and measured fit.
