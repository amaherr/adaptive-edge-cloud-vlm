# Account administration only: no billable compute or network resources.
provider "aws" { region = "us-east-1" }
resource "aws_servicequotas_service_quota" "gpu_pair" {
  service_code = "ec2"
  quota_code   = "L-DB2E81BA"
  value        = 16
}
output "gpu_quota" {
  value = {
    service         = "ec2"
    quota_code      = aws_servicequotas_service_quota.gpu_pair.quota_code
    requested_vcpus = aws_servicequotas_service_quota.gpu_pair.value
  }
}
