# Account quota prerequisite

This independent state requests 16 regional On-Demand G/VT vCPUs in us-east-1,
enough for one g6f.2xlarge and one g6.2xlarge. It creates no compute or networking.
Use the same pinned Terraform/provider and `scripts/terraform_cloud.sh` bridge.
Review a saved plan before apply. Protect/back up its ignored local state with
the bootstrap state until a remote backend migration is performed. Once the
bootstrap bucket exists, copy backend.tf.example to ignored backend.local.tf
and use init -migrate-state with the bucket/region arguments. This migration
was completed for the current deployment on 5 November 2026.

Terraform accepting the request does **not** mean AWS approved it. Check the
Service Quotas request history and the actual applied quota before launch.
Quota increases do not reserve capacity and do not start instance billing.
Destroying this tracking resource does not undo an approved account quota.
