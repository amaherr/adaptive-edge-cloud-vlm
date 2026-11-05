# Retained thesis data, images, budget and data volumes live outside disposable compute.
variable "region" { type = string }
variable "availability_zone" { type = string }
variable "artifact_bucket_name" { type = string }
variable "monthly_budget_usd" { type = number }
variable "notification_email" { type = string }
variable "provision_data_volumes" {
  type    = bool
  default = false
}
variable "data_volume_gib" {
  type    = number
  default = 50
  validation {
    condition     = var.data_volume_gib >= 30 && var.data_volume_gib == floor(var.data_volume_gib)
    error_message = "Choose an integer data volume capacity of at least 30 GiB."
  }
}
provider "aws" {
  region = var.region
  default_tags { tags = { Project = "adaptive-edge-cloud-vlm", Campaign = "cloud-v1", ManagedBy = "Terraform" } }
}
module "storage" {
  source      = "../../modules/storage"
  bucket_name = var.artifact_bucket_name
}
module "budgets" {
  source = "../../modules/budgets"
  amount = var.monthly_budget_usd
  email  = var.notification_email
}
resource "aws_ebs_volume" "data" {
  for_each          = var.provision_data_volumes ? toset(["edge", "cloud"]) : toset([])
  availability_zone = var.availability_zone
  size              = var.data_volume_gib
  type              = "gp3"
  encrypted         = true
  tags              = { Role = each.key }
  lifecycle { prevent_destroy = true }
  depends_on = [module.budgets]
}
output "bucket_name" { value = var.artifact_bucket_name }
output "bucket_arn" { value = module.storage.bucket_arn }
output "ecr_arn" { value = module.storage.ecr_arn }
output "ecr_url" { value = module.storage.ecr_url }
output "monthly_budget_usd" { value = var.monthly_budget_usd }
output "availability_zone" { value = var.availability_zone }
output "volume_ids" { value = { for role, volume in aws_ebs_volume.data : role => volume.id } }
