mock_provider "aws" {}
variables {
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
  expect_failures = [var.launch_approved]
}
