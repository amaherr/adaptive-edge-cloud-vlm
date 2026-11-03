provider "aws" {
  region = var.region
  default_tags { tags = { Project = "adaptive-edge-cloud-vlm", Campaign = "cloud-v1", ManagedBy = "Terraform" } }
}
module "network" {
  source = "../../modules/network"
  region = var.region
  az     = var.availability_zone
}
module "storage" {
  source      = "../../modules/storage"
  bucket_name = var.artifact_bucket_name
}
module "security" {
  source     = "../../modules/security"
  vpc_id     = module.network.vpc_id
  bucket_arn = module.storage.bucket_arn
  ecr_arn    = module.storage.ecr_arn
}
module "budgets" {
  source = "../../modules/budgets"
  amount = var.monthly_budget_usd
  email  = var.notification_email
}
module "compute" {
  source          = "../../modules/compute"
  enabled         = var.enable_compute
  approved        = var.launch_approved
  subnet_id       = module.network.subnet_id
  az              = var.availability_zone
  amis            = { edge = var.edge_ami_id, cloud = var.cloud_ami_id }
  groups          = module.security.groups
  profiles        = module.security.profiles
  data_volume_gib = var.data_volume_gib
  max_hours       = var.experiment_max_runtime_hours
  idle_minutes    = var.idle_timeout_minutes
  depends_on      = [module.budgets]
}
module "observability" {
  source    = "../../modules/observability"
  instances = module.compute.instance_ids
  email     = var.notification_email
}
output "deployment" {
  value = { instances = module.compute.instance_ids, data_volumes = module.compute.volume_ids, availability_zone = var.availability_zone, vpc_id = module.network.vpc_id, subnet_id = module.network.subnet_id, bucket = module.storage.bucket_arn, ecr = module.storage.ecr_url }
}
