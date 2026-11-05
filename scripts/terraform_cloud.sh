#!/bin/bash
# Bridge native AWS CLI login/SSO into the pinned Terraform provider.
set -euo pipefail
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$repository/infra/local"
export CLOUD_AWS_SOURCE_PROFILE=${CLOUD_AWS_SOURCE_PROFILE:-cloud-infrastructure}
export AWS_CONFIG_FILE="$repository/infra/local/aws-config"
export AWS_PROFILE=cloud-terraform
export AWS_SDK_LOAD_CONFIG=1
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
# The helper clears this generated config before resolving the source profile.
cat > "$AWS_CONFIG_FILE" <<CONFIG
[profile cloud-terraform]
region = us-east-1
credential_process = bash "$repository/scripts/aws_credential_process.sh"
CONFIG
exec "${TERRAFORM_BIN:-terraform}" "$@"
