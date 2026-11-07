#!/usr/bin/env bash
# Keep API credentials out of argv, files, Terraform variables and logs.
set +x
set -euo pipefail
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -z ${VULTR_API_KEY:-} ]]; then
  VULTR_API_KEY=$(aws --profile "${CLOUD_AWS_SOURCE_PROFILE:-cloud-infrastructure}" \
    --region us-east-1 ssm get-parameter --name /adaptive-vlm/vultr/api-key \
    --with-decryption --query Parameter.Value --output text --no-cli-pager)
  export VULTR_API_KEY
fi
exec bash "$repository/scripts/terraform_cloud.sh" "$@"
