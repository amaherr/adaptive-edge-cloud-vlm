mock_provider "aws" {}
override_data {
  target = data.terraform_remote_state.retained
  values = { outputs = {
    bucket_name        = "test-cloud-vlm-artifacts", bucket_arn = "arn:aws:s3:::test-cloud-vlm-artifacts",
    ecr_arn            = "arn:aws:ecr:us-east-1:123456789012:repository/test", ecr_url = "123456789012.dkr.ecr.us-east-1.amazonaws.com/test",
    monthly_budget_usd = 50, availability_zone = "us-east-1a",
    volume_ids         = { edge = "vol-0123456789abcdef0", cloud = "vol-0123456789abcdef1" }
  } }
}
variables {
  state_bucket_name            = "test-cloud-vlm-state"
  region                       = "us-east-1"
  availability_zone            = "us-east-1a"
  artifact_bucket_name         = "test-cloud-vlm-artifacts"
  monthly_budget_usd           = 50
  notification_email           = "test@example.com"
  edge_ami_id                  = "ami-0123456789abcdef0"
  cloud_ami_id                 = "ami-0123456789abcdef0"
  experiment_max_runtime_hours = 2
  idle_timeout_minutes         = 15
}
run "foundation_has_no_gpu" {
  command = plan
  assert {
    condition     = length(module.compute.instance_ids) == 0
    error_message = "Foundation must not launch GPU instances."
  }
}
run "launch_requires_approval" {
  command = plan
  variables { enable_compute = true }
  expect_failures = [var.launch_approved, var.enable_private_endpoints]
}

run "foundation_has_no_hourly_endpoints" {
  command = plan
  assert {
    condition     = module.network.interface_endpoint_count == 0
    error_message = "Foundation must not create hourly endpoints by default."
  }
}

run "approved_pair_plan" {
  command = plan
  variables {
    enable_compute           = true
    enable_private_endpoints = true
    launch_approved          = true
  }
  assert {
    condition     = length(module.compute.instance_ids) == 2 && module.network.interface_endpoint_count == 7
    error_message = "Approved deployment requires exactly two nodes and seven private endpoints."
  }
}
run "runtime_templates_render" {
  command = plan
  module { source = "./tests/runtime_template" }
  assert {
    condition     = alltrue([for role, value in output.cloud_init : length(yamldecode(value).write_files) >= 8])
    error_message = "Both runtime templates must render as valid cloud-init YAML."
  }
}
