# Vultr cost analysis — $80 planning target, P1 BLOCKED

Updated 7 November 2026. Owner preferences: Atlanta, USA (`atl`), approximately
**$80 total**, alerts to **a.maher1506@gmail.com**. The amount is a planning target,
not permission to launch a particular paid resource. A separate single-node
preflight ceiling and concrete launch approval remain outstanding. The alert
address is recorded; delivery is **not configured or tested**.

## Current availability is the first blocker

Read-only public API requests at **11:40 UTC on 7 November 2026** returned:

- `/v2/plans?type=vcg&per_page=500`: exactly three plans, all L40S; pagination
  total is 3 and next link is empty.
- `/v2/regions/atl/availability?type=vcg`: both `available_plans` and
  `available_vpc_only_plans` are empty.
- All three catalog plans have empty `locations` and `location_cost` fields.

| Public catalog plan | GPU allocation | Hourly USD | Locations returned |
| --- | --- | ---: | --- |
| `vcg-l40s-16c-180g-48vram` | 1 L40S, 48 GB aggregate VRAM | 1.671 | None |
| `vcg-l40s-32c-375g-96vram` | 2 L40S, 96 GB aggregate VRAM | 3.342 | None |
| `vcg-l40s-64c-750g-192vram` | 4 L40S, 192 GB aggregate VRAM | 6.684 | None |

These are public catalog rates, **not an authenticated quote, capacity reservation
or evidence of purchasable Atlanta inventory**. There is currently no verified
edge/cloud pair to deploy. Archived responses, source URLs and timestamps are in
[evidence/public_gpu_catalog_2026-11-07.json](evidence/public_gpu_catalog_2026-11-07.json).
Do not divide monthly catalog figures to replace explicit hourly rates. The
current model device mapping uses one GPU per process; buying a multi-GPU VM
does not automatically accelerate it or combine its VRAM.

Vultr's [Cloud GPU overview](https://docs.vultr.com/product-overviews/cloud-gpu)
advertises an A16 configuration with 16 GB GPU memory at **$0.471/hour**. It was
absent from the live response above. Treat it only as an indicative candidate,
not an available SKU or approved host. Smaller fractional tiers mentioned by
third-party sites were not verified and are excluded from this estimate.

## Conditional hardware and storage estimate

For sensitivity analysis only, assume a 16 GB A16 virtual edge at $0.471/hour
and a 48 GB L40S virtual cloud at $1.671/hour: **$2.142/hour for both VMs**.
These rates combine an advertised edge rate and a public catalog cloud rate;
they do not establish that the pair can be bought together anywhere. GPU fit,
BF16/driver support, throughput difference, private networking and regional
pricing must pass P1 and the pilot before selecting the pair.

[Vultr billing](https://docs.vultr.com/support/platform/billing/how-am-i-billed-for-my-servers)
uses a minimum one-hour server billing unit and continues while a VM is stopped.
Budget **all time from creation to destruction**, including setup, downloads,
waiting, failures and idle time. Use per-instance rounded-up session hours for
approval, not just benchmark inference duration. One paired day at this assumed
rate is **$51.41** before extras; destroy both nodes outside approved sessions.

| Ancillary item | Cost basis / planning treatment |
| --- | --- |
| Vultr registry | Published Start Up tier: $0/month for 10 GB; confirm account availability and total stored image bytes |
| Existing AWS S3 + existing ECR or free registry | **$7 total allowance for one month** for storage, requests, image pulls/restores, state, control watchdog and alert delivery; a planning allowance, not a quote or guaranteed cap |
| New Vultr Standard Object Storage | **$18 for one retained month**; budget a full month pending subscription billing confirmation; separate watchdog/alerts/other charges still apply |
| Node-local disks | Use storage included with the verified GPU plan; no extra block volumes/snapshots assumed |
| Public transfer / taxes / FX / billing rounding | Not precisely quoted; draw from contingency and re-estimate after account/billing and session details are known |

The registry [plan documentation](https://docs.vultr.com/support/products/container-registry/what-are-the-available-plans-for-vultr-container-registry)
lists the free 10 GB tier and a $5/month 20 GB tier. The repository's existing
published image is approximately 4.03 GB; a new Vultr build may differ. Verify
actual registry usage before assuming the free tier is sufficient. Keep only
necessary immutable releases; never delete an image needed for recovery.

[Vultr Object Storage billing](https://docs.vultr.com/support/platform/billing/how-is-object-storage-billed)
lists Standard at $18/month including 1000 GB storage and 1000 GB outbound
transfer; excess outbound is $0.01/GB. Retaining it for three months is **$54**
before other charges, which is material within an $80 total target. Do not use
an archival tier for live checkpoints without validating immediate recovery.

**Recommended cost-saving option:** reuse the existing independently retained
AWS S3 bucket, subject to verifying credentials, permissions and incremental
billing. The EC2 GPU quota refusal does not by itself prohibit S3 use. The
repository records an already retained bucket and image; their present account
state still needs verification. S3 has usage and request charges and AWS-to-Vultr
restores/image pulls may incur transfer charges. Consult [AWS S3 pricing](https://aws.amazon.com/s3/pricing/).
The $7 allowance does not assert a current AWS per-GB quote or assume credits.
Do not retain AWS EC2 nodes, private endpoints or unrelated paid infrastructure
for the Vultr experiment. Existing AWS resources are not changed by this review.

## Gate 2 runtime sensitivity

The frozen campaign has **8,100 sequential measured attempts plus 27 warmups**.
No representative Vultr mean latency has been measured. Historical physical
single-request timings of 9.3 s (synthetic) and 17.5 s (one COCO image) establish
neither a Vultr average nor an A16/L40S speedup.

Let `s` be the arithmetic mean seconds per measured attempt across all three
policies and nine blocks, including completed failure/timeout time:

```text
measured_hours = 8100 * s / 3600 = 2.25 * s
billable_pair_hours_estimate = 1.30 * measured_hours + 4
compute_estimate = billable_pair_hours_estimate * 2.142
```

The **30% runtime allowance and 4 extra paired hours are planning assumptions**
for warmups, model loads, probes, cache preparation, preflight/pilot, checkpoint
validation, teardown and session rounding. They are not measured guarantees.
Unusual retries, longer setup or many short sessions can exceed them. The
single-edge preflight is conservatively budgeted inside that four-hour allowance;
its actual bill will use single-node time and require separate approval.

The following includes **$12 contingency** and the ancillary allowance shown:

| Assumed mean / attempt | Pair hours incl. allowances | Compute | Total, existing S3 path (+$7+$12) | Total, new Vultr storage (+$18+$12)* |
| --- | ---: | ---: | ---: | ---: |
| 5 s | 18.625 | $39.89 | **$58.89** | **$69.89** |
| 10 s | 33.250 | $71.22 | **$90.22** | **$101.22** |
| 15 s | 47.875 | $102.55 | **$121.55** | **$132.55** |
| 20 s | 62.500 | $133.88 | **$152.88** | **$163.88** |
| 30 s | 91.750 | $196.53 | **$215.53** | **$226.53** |

*The new-storage column replaces the $7 hybrid allowance with the $18 storage
subscription. Separate watchdog/alert/transfer costs must also fit contingency;
it is a lower-bound planning scenario if those exceed contingency. All figures
assume one month of retention and no GPU charges outside these session hours.

Under these assumptions, $80 allows at most:

- **28.48 paired hours / about 8.37 s mean per attempt** with the $7 hybrid
  allowance and $12 contingency.
- **23.34 paired hours / about 6.61 s mean per attempt** with the $18 Vultr
  subscription and $12 contingency, before separately quoted ancillary extras.

These limits cover Gate 2 and the stated allowances, **not completion of the
whole thesis**. G3/G4 local fitting may be cheap, but new G5/G6 GPU campaigns,
additional correctness tests, reruns and later retention consume extra budget.
Do not promise full thesis completion within $80 without measuring and scoping
those phases. If one used two lowest-priced catalog L40S nodes, their identical
hardware would not establish the intended weaker-edge/stronger-cloud contrast;
that pair is also unavailable in Atlanta and is not recommended as a workaround.

## Proposed controls and next decision

- Keep the total planning target at $80; propose a **$5 maximum single-edge
  preflight** within that target, pending owner approval and a verified plan.
- Proposed cumulative alert thresholds: **$40, $60, $72 and $80**, sent to
  `a.maher1506@gmail.com`. Destination is recorded, not configured/tested.
- At $72, checkpoint and stop scheduling, then invoke a verified approved
  destroy path; $8 headroom is a proposed buffer for delayed billing and cleanup.
  Alerts alone are not a hard cap. A watchdog outside both GPU VMs must enforce
  approved session expiry and verify destruction through the provider API.
- Establish already-incurred campaign spend before approval. Count persistent
  subscriptions between sessions and all campaign-related AWS charges in $80.
- Finish account API setup, then repeat authenticated `--region atl` discovery.
  If GPU inventory is still empty, ask provider support about GPU access and
  capacity; select another region only with the owner's agreement.
- If a pair becomes available, run the approved small pilot across representative
  images and fastest/slowest network blocks. Use mean elapsed attempt time,
  load/probe/setup time and measured storage growth to replace this sensitivity
  model, including an explicit reserve for later G5/G6 work. Requote before full
  campaign approval; retain the frozen scientific workload.

**Assessment:** $80 is a plausible limited preflight/pilot budget and may cover
Gate 2 at sufficiently low measured runtimes. It is **not yet a defensible
all-in completion estimate**. The immediate blocker is GPU availability and
account verification, followed by an actual pilot runtime. No paid launch or
email delivery was performed for this analysis.
