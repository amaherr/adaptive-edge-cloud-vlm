#!/bin/bash
# Terraform credential_process: stdout is reserved for temporary credential JSON.
set -euo pipefail
unset AWS_CONFIG_FILE AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
exec "${CLOUD_AWS_CLI:-aws}" configure export-credentials --profile "${CLOUD_AWS_SOURCE_PROFILE:-cloud-infrastructure}" --format process
