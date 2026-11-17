#!/bin/bash
# EC2 Image Builder only. Model packages and weights are never installed here.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# Transient timer is in /run and is not baked into the final AMI.
uptime_seconds=$(cut -d. -f1 /proc/uptime)
remaining_seconds=$((7200 - uptime_seconds))
if [ "$remaining_seconds" -le 0 ]; then
  shutdown -h now
  exit 1
fi
systemd-run --unit=cloud-vlm-image-build-deadline --on-active="${remaining_seconds}s" /sbin/shutdown -h now
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service || true
apt-get update
apt-get install -y awscli build-essential dkms "linux-headers-$(uname -r)" \
  curl ca-certificates gnupg unzip docker.io docker-compose-v2 iproute2 iperf3 git
# Use the specific AWS GRID artifact and reject any content change.
aws s3 cp s3://ec2-linux-nvidia-drivers/grid-19.5/NVIDIA-Linux-x86_64-580.159.03-grid-aws.run /tmp/grid.run --no-progress
echo '8ea16dc3c9034813a11bf52d8024f5466b48ab36cae6d91a7c95f8e77912e5dd  /tmp/grid.run' | sha256sum --check --strict
if lsmod | awk '{print $1}' | grep -qx nouveau; then
  echo 'Base image loaded nouveau; a new explicitly reviewed image recipe is required' >&2
  exit 1
fi
printf 'blacklist nouveau\noptions nouveau modeset=0\n' > /etc/modprobe.d/disable-nouveau.conf
update-initramfs -u
sh /tmp/grid.run --silent --dkms
modprobe nvidia
# Official NVIDIA toolkit repository; package versions are archived in the AMI.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker
curl -fsSL --retry 3 https://amazoncloudwatch-agent.s3.amazonaws.com/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb -o /tmp/cloudwatch-agent.deb
dpkg -i /tmp/cloudwatch-agent.deb
mkdir -p /usr/local/share/cloud-vlm-host
dpkg-query -W > /usr/local/share/cloud-vlm-host/os-packages.tsv
sha256sum /tmp/grid.run /tmp/cloudwatch-agent.deb > /usr/local/share/cloud-vlm-host/install-artifacts.sha256
uname -a > /usr/local/share/cloud-vlm-host/kernel.txt
nvidia-smi -q > /usr/local/share/cloud-vlm-host/gpu-build-inventory.txt
nvidia-ctk --version > /usr/local/share/cloud-vlm-host/toolkit.txt
docker version > /usr/local/share/cloud-vlm-host/docker.txt
docker compose version >> /usr/local/share/cloud-vlm-host/docker.txt
# Preserve the image provenance, not downloaded installers or temporary caches.
rm -f /tmp/grid.run /tmp/cloudwatch-agent.deb
apt-get clean
rm -rf /var/lib/apt/lists/*
