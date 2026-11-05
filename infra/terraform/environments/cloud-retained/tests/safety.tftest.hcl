mock_provider "aws" {}
variables {
  region               = "us-east-1"
  availability_zone    = "us-east-1a"
  artifact_bucket_name = "test-retained-cloud-vlm"
  monthly_budget_usd   = 20
  notification_email   = "test@example.com"
}
run "no_data_volumes_by_default" {
  command = plan
  assert {
    condition     = length(aws_ebs_volume.data) == 0
    error_message = "Data-volume billing must require explicit provisioning."
  }
}
run "paired_encrypted_data_volumes" {
  command = plan
  variables { provision_data_volumes = true }
  assert {
    condition     = length(aws_ebs_volume.data) == 2 && alltrue([for role, v in aws_ebs_volume.data : v.encrypted && v.size == 50 && v.type == "gp3" && v.availability_zone == "us-east-1a"])
    error_message = "Provision exactly two encrypted 50 GiB gp3 volumes in the selected AZ."
  }
}
