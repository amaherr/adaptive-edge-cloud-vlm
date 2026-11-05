# Predeployment cost review (not an AWS quote)

## Live pricing check, 5 November 2026

Read-only AWS Price List API queries in us-east-1 confirmed g6f.2xlarge at
$0.475/hour, g6.2xlarge at $0.9776/hour, gp3 storage at $0.08/GB-month, and
interface endpoints at $0.01/endpoint-hour. Raw pricing products are archived
under ignored artifacts/infra. These confirm the arithmetic below; they still
exclude ancillary services, image preparation and taxes. No GPU resources have
been launched. The applied G/VT quota is currently zero; a Terraform-managed
request for 16 vCPUs was declined; an appeal is being prepared.

## Storage revision before first GPU deployment

Measured local inputs are approximately 3.7 GiB model/cache and 39 MiB image data.
The initial 150 GiB per-node data-volume default was excessive for this frozen
workload. Before creating any data volume, the default was reduced to **50 GiB
per node**, with a minimum input validation of 30 GiB. Both remain gp3 with the
same default IOPS/throughput; this does not change the proposed GPU pair.
Actual free space and logs will be checked during preflight and monitored at
85% utilization. No existing volume was reduced or deleted.

The revised provisioned total is 2 x 50 GiB data + 2 x 40 GiB root = 180 GiB,
or **$14.40/month** while all those volumes exist. Two retained data volumes
alone cost $8/month after disposable root volumes are removed. Seven endpoints
still cost $51.10/month if accidentally retained. A two-hour test window using
the current defaults is approximately **$3.08** for compute, endpoint-hours and
prorated EBS, plus the ancillary services listed below. Earlier 380 GiB figures
are retained as the superseded initial estimate, not the current configuration.

User input: $20/month is an alert threshold, not a hard ceiling. Region candidate
us-east-1. No paid resources have been created by this migration.

Seven interface endpoints in one AZ at $0.01/endpoint-hour imply $0.07/hour,
$1.68/day, or **$51.10 per 730-hour month**, before endpoint data processing.
This alone exceeds the $20 notification threshold. Foundation deployment now
creates zero interface endpoints by default; GPU launch requires explicitly
turning them on. Delete endpoints through Terraform when the test window ends;
stopping instances does not stop these charges.

The current planned storage is 2 x 150 GiB data plus 2 x 40 GiB root = 380 GiB.
At an illustrative gp3 rate of $0.08/GB-month, that is **$30.40/month** while all
volumes exist, before snapshots. Data volumes intentionally survive instance
replacement and are protected from destroy. Retaining 300 GiB alone at this
rate is $24/month. Confirm the actual regional quote before apply. No provisioned
IOPS/throughput upgrades are requested.

The roadmap's indicative compute estimate is $1.4526/hour for the pair; this
has NOT been independently verified as the current account/region price.
Using it only as a planning assumption, a two-hour window would be $2.9052 compute
+ $0.14 endpoint-hours + approximately $0.0833 EBS (380 GiB, prorated over 730 h),
or **about $3.13**, excluding retained storage, AMI construction, S3, ECR,
CloudWatch, data processing/transfer and taxes. This is not authorization to
charge an unreviewed bill and is not an all-in estimate.

Before apply, obtain current On-Demand quotes for both GPU types, gp3, endpoint
hours and storage/log services; record exact resource lifetimes, existing month
to date spend, model image/cache sizes and AMI preparation costs. Send the user
the resulting deployment estimate. The final estimate is blocked on working AWS
access and AMI/build decisions; do not represent this preliminary model as final.

Sources checked:
- https://aws.amazon.com/privatelink/pricing/ (endpoint ENI hourly example and
  billing continues until endpoint deletion)
- https://aws.amazon.com/ebs/pricing/ (gp3 example; price varies by region)
- docs/Gate2_Cloud_Migration_Roadmap_v2.md (unverified compute planning assumption)


Image size measurement: Docker reports 4,032,456,950 bytes of image content,
while the unpacked Python site-packages occupy 6,972,311,078 bytes. These are
different measurements. Registry-reported image bytes will be used for ECR
billing estimates after publication; do not label Docker's figure as unpacked
filesystem size. ECR's published storage example is $0.10/GB-month, before any
account-specific allowance (https://aws.amazon.com/ecr/pricing/).


The published registry image is 4,032,453,520 bytes, giving approximately
$0.40/month at the published $0.10/GB-month storage example rate before
allowances. No free credits are assumed by the cost model. GPU execution,
data EBS and hourly endpoints remain disabled after the quota refusal.
