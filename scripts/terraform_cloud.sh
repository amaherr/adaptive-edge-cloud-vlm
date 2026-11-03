#!/bin/bash
# Linux Terraform obtains short-lived credentials from the Windows login profile.
set -euo pipefail
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export AWS_CONFIG_FILE="$repository/infra/local/aws-config"
export AWS_PROFILE=cloud-terraform
export AWS_SDK_LOAD_CONFIG=1
if [ ! -f "$AWS_CONFIG_FILE" ]; then
  echo 'Create the local credential bridge per docs/cloud_acceptance_v1.md' >&2
  exit 1
fi
exec "${TERRAFORM_BIN:-terraform}" "$@"
