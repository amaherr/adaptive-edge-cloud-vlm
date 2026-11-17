# Protected state bootstrap

See `../../../docs/cloud_environment_handoff_v1.md` for the deployment sequence.
Use pinned Terraform 1.10.5, authenticate with SSO/OIDC, and supply region and
state_bucket_name. Start with local state in this directory (ignored by Git),
restrict permissions, encrypt/back it up, then migrate to a separate protected
backend key. Run init, fmt -check, validate, plan -out and review before apply.
This bucket is separate from experiment artifacts and protected from destroy.

After creating the bucket, copy backend.tf.example to ignored backend.local.tf
and run init -migrate-state with the bucket and region backend arguments. The
bootstrap, account quota and experiment stacks use separate locked S3 keys.
Keep local migration backups protected; never commit them.
