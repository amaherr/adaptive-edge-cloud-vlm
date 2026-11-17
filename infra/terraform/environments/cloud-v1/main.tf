provider "aws" {
  region = var.region
  default_tags { tags = { Project = "adaptive-edge-cloud-vlm", Campaign = "cloud-v1", ManagedBy = "Terraform" } }
}
module "network" {
  source                   = "../../modules/network"
  region                   = var.region
  az                       = var.availability_zone
  enable_private_endpoints = var.enable_private_endpoints
}
# Retained resources are imported first; forgetting them here never deletes AWS data.
removed {
  from = module.storage
  lifecycle { destroy = false }
}
removed {
  from = module.budgets
  lifecycle { destroy = false }
}
data "terraform_remote_state" "retained" {
  backend = "s3"
  config  = { bucket = var.state_bucket_name, region = var.region, key = "cloud-retained/terraform.tfstate" }
}
module "security" {
  source     = "../../modules/security"
  vpc_id     = module.network.vpc_id
  bucket_arn = data.terraform_remote_state.retained.outputs.bucket_arn
  ecr_arn    = data.terraform_remote_state.retained.outputs.ecr_arn
}
module "compute" {
  source              = "../../modules/compute"
  region              = var.region
  enabled             = var.enable_compute
  approved            = var.launch_approved && data.terraform_remote_state.retained.outputs.monthly_budget_usd == var.monthly_budget_usd && data.terraform_remote_state.retained.outputs.bucket_name == var.artifact_bucket_name
  subnet_id           = module.network.subnet_id
  az                  = var.availability_zone
  amis                = { edge = var.edge_ami_id, cloud = var.cloud_ami_id }
  groups              = module.security.groups
  profiles            = module.security.profiles
  volume_ids          = data.terraform_remote_state.retained.outputs.volume_ids
  data_volume_az      = data.terraform_remote_state.retained.outputs.availability_zone
  max_hours           = var.experiment_max_runtime_hours
  idle_minutes        = var.idle_timeout_minutes
  edge_instance_type  = var.edge_instance_type
  cloud_instance_type = var.cloud_instance_type
  depends_on          = [data.terraform_remote_state.retained]
}
# Decision D7: above budget_stop_usd of actual spend, AWS Budgets stops both
# instances (it does not cap billing; EBS and S3 keep accruing).
resource "aws_iam_role" "budget_action" {
  count              = var.enable_compute ? 1 : 0
  name_prefix        = "cloud-v1-budget-stop-"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "budgets.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}
resource "aws_iam_role_policy_attachment" "budget_action" {
  count      = var.enable_compute ? 1 : 0
  role       = aws_iam_role.budget_action[0].name
  policy_arn = "arn:aws:iam::aws:policy/AWSBudgetsActions_RolePolicyForResourceAdministrationWithSSM"
}
resource "aws_budgets_budget_action" "stop" {
  count              = var.enable_compute ? 1 : 0
  budget_name        = data.terraform_remote_state.retained.outputs.budget_name
  action_type        = "RUN_SSM_DOCUMENTS"
  approval_model     = "AUTOMATIC"
  notification_type  = "ACTUAL"
  execution_role_arn = aws_iam_role.budget_action[0].arn
  action_threshold {
    action_threshold_type  = "ABSOLUTE_VALUE"
    action_threshold_value = var.budget_stop_usd
  }
  definition {
    ssm_action_definition {
      action_sub_type = "STOP_EC2_INSTANCES"
      region          = var.region
      instance_ids    = values(module.compute.instance_ids)
    }
  }
  subscriber {
    address           = var.notification_email
    subscription_type = "EMAIL"
  }
  depends_on = [aws_iam_role_policy_attachment.budget_action]
}
module "observability" {
  source       = "../../modules/observability"
  instances    = module.compute.instance_ids
  idle_minutes = var.idle_timeout_minutes
  email        = var.notification_email
}
output "deployment" {
  value = { instances = module.compute.instance_ids, data_volumes = module.compute.volume_ids, availability_zone = var.availability_zone, vpc_id = module.network.vpc_id, subnet_id = module.network.subnet_id, bucket = data.terraform_remote_state.retained.outputs.bucket_arn, ecr = data.terraform_remote_state.retained.outputs.ecr_url }
}
