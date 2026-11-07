# Disposable Vultr GPU compute only

Pinned Terraform 1.10.5 and official Vultr provider 2.33.2; commit the lockfile.
`deploy_gpu_instances=false` by default creates zero GPUs. Only `edge` and
`cloud` roles are allowed (at most two). A paid launch requires recorded owner
approval, a future expiry, verified plan/OS and existing VPC/firewall/SSH refs.
The operator selected Atlanta (`atl`) and Ubuntu 24.04 LTS x64 (API OS ID 2284)
for the single-node preflight; this is not a verified driver/model acceptance.

The API key comes from environment via encrypted SSM, using
`scripts/vultr/terraform.sh`. Never pass it as a Terraform variable. Keep all
`.tfvars`, plans, state and private keys out of Git. `cloud_init` contains only
nonsecret firewall/host setup; it must not reinstall vendor GPU tooling blindly.
Pass plain cloud-init text: provider 2.33.2 performs the API Base64 encoding.
Encoding it in Terraform as well prevents delivery of the intended cloud-config.

Backend: existing protected state bucket, key `vultr-v1/gpu/terraform.tfstate`,
region `us-east-1`, S3 lockfile enabled. Networking and retained data/state are
managed separately. Teardown sets the deploy flag false, reviews a saved plan,
deletes only these GPU resources and independently verifies the campaign API
inventory. Power-off does not stop billing.

Before launch, the AWS-hosted watchdog must pass live acceptance, and the
approved session/quote must be durably registered **before** Terraform apply.
See [operations](../../operations/README.md). An immutable image digest,
model/data verification and live CUDA/BF16 tests remain mandatory. Do not run
the full campaign from this low-level resource module alone.
