#!/bin/bash
# Mount only the Terraform-owned Nitro EBS volume identified by its exact ID.
set -euo pipefail
volume_id=${1:?Usage: mount-data.sh vol-ID}
[[ "$volume_id" =~ ^vol-[0-9a-f]+$ ]] || exit 2
serial=${volume_id//-/}
device="/dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_$serial"
for ((attempt=0; attempt<120; attempt++)); do
  [ -b "$device" ] && break
  sleep 2
done
[ -b "$device" ] || { echo 'Expected EBS device did not appear' >&2; exit 1; }
if mountpoint -q /srv/adaptive-vlm; then
  [ "$(readlink -f "$(findmnt -n -o SOURCE /srv/adaptive-vlm)")" = "$(readlink -f "$device")" ] || exit 1
  exit 0
fi
fs_type=$(blkid -o value -s TYPE "$device" || true)
if [ -z "$fs_type" ]; then
  # Refuse any partition table/signature rather than interpreting it as empty.
  [ -z "$(wipefs --no-act --noheadings --output TYPE "$device")" ] || exit 1
  mkfs.ext4 "$device"
elif [ "$fs_type" != ext4 ]; then
  echo "Refusing unexpected filesystem: $fs_type" >&2
  exit 1
fi
uuid=$(blkid -o value -s UUID "$device")
mkdir -p /srv/adaptive-vlm
entry="UUID=$uuid /srv/adaptive-vlm ext4 defaults,nofail 0 2"
grep -qF "UUID=$uuid " /etc/fstab || echo "$entry" >> /etc/fstab
mount /srv/adaptive-vlm
for directory in models data artifacts; do
  install -d -o 10001 -g 10001 "/srv/adaptive-vlm/$directory"
done
