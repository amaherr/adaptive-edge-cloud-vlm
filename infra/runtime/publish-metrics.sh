#!/bin/bash
# Host-only AWS CLI uses the instance role. No model dependencies on the host.
set -euo pipefail
region=${1:?region required}
role=${2:?role required}
uptime_seconds=$(cut -d. -f1 /proc/uptime)
heartbeat=/srv/adaptive-vlm/artifacts/job.heartbeat
idle_seconds=$uptime_seconds
if [ -f "$heartbeat" ]; then
  idle_seconds=$(( $(date +%s) - $(stat -c %Y "$heartbeat") ))
fi
used_percent=$(df --output=pcent /srv/adaptive-vlm | tail -1 | tr -dc '0-9')
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
[ -n "$gpu_util" ] || { echo 'GPU utilization unavailable' >&2; exit 1; }
aws --region "$region" cloudwatch put-metric-data --namespace AdaptiveVLM/CloudV1 \
  --metric-data "MetricName=IdleSeconds,Dimensions=[{Name=Role,Value=$role}],Value=$idle_seconds,Unit=Seconds" \
  "MetricName=StorageUsedPercent,Dimensions=[{Name=Role,Value=$role}],Value=$used_percent,Unit=Percent" \
  "MetricName=GPUUtilization,Dimensions=[{Name=Role,Value=$role}],Value=$gpu_util,Unit=Percent"
