# Render the production templates without deferred resource IDs or deployment.
locals {
  infra = "${path.module}/../../../../.."
  cloud_init = { for role in ["edge", "cloud"] : role => templatefile("${local.infra}/cloud-init/${role}.yaml", {
    max_seconds    = 7200, idle_seconds = 900, data_volume_id = "vol-0123456789abcdef0",
    mount_script   = file("${local.infra}/runtime/mount-data.sh"),
    metrics_script = file("${local.infra}/runtime/publish-metrics.sh"),
    region         = "us-east-1", role = role
  }) }
}
output "cloud_init" { value = local.cloud_init }
