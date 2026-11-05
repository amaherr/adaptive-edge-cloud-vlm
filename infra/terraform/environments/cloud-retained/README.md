# Retained campaign infrastructure

This independent encrypted/versioned S3 state owns the private artifact bucket,
ECR repository, monthly budget and optional persistent data volumes. The
`cloud-v1` disposable stack reads its outputs; destroying compute/network/SSM
roles cannot delete these retained resources or remove the budget notifications.
The backend bucket remains in its own protected bootstrap state.

Fresh deployment order:

1. Bootstrap the state bucket; migrate bootstrap and account quota states.
2. Initialize cloud-retained with the bucket/region backend arguments. Supply
   reviewed variables and keep provision_data_volumes=false. Plan/review/apply
   storage/ECR and the budget before any GPU launch or billable data volumes.
3. Initialize cloud-v1 with the same bucket and its separate backend key. Set
   state_bucket_name and matching budget/bucket expectations in local tfvars.
   Keep compute/endpoints disabled until the documented gates pass.
4. After quota, AMI and notification checks, explicitly provision the two data
   volumes here, review that plan, then review and enable endpoints/compute in
   cloud-v1. The volume IDs and AZ must match; the compute module refuses a
   different AZ or incomplete pair. Defaults are two encrypted 50 GiB gp3
   volumes, not the superseded 150 GiB allocation.

Existing-deployment migration completed on 5 November 2026:

- Seven storage/budget resources were imported into this state using one-time
  ignored import blocks. The reviewed plan had seven imports and zero AWS
  creates, updates or deletes. Import blocks were then archived and removed.
- cloud-v1 uses explicit removed blocks with destroy=false for the old module
  addresses. Its reviewed apply forgot only the already-imported resources.
- No EBS data volumes existed during migration, so none required transfer.
- Both real stacks must show clean plans; retain remote version history.

The current disposable destroy plan lists 29 foundation resources and excludes
S3/ECR/budget/EBS/backend deletion. It was reviewed, **not applied**. Regenerate
and review it after real nodes are provisioned. Archive all logs/results and
verify independent S3 restore before teardown. Retain the content hashes and
review receipt in ignored artifacts/infra; raw plan files may contain private
configuration and must not be committed.

This stack has prevent_destroy on its thesis bucket, ECR and data volumes.
A full retained-stack destroy should refuse those deletions. Review any final
retention change after thesis acceptance; do not simply remove the protection
to get a destroy command to succeed. Data EBS continues billing while instances
are stopped, so the complete project cost plan must include its lifetime.
