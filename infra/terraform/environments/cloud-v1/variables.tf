variable "region" { type = string }
variable "availability_zone" { type = string }
variable "artifact_bucket_name" { type = string }
variable "monthly_budget_usd" { type = number }
variable "notification_email" { type = string }
variable "edge_ami_id" { type = string }
variable "cloud_ami_id" { type = string }
variable "experiment_max_runtime_hours" { type = number }
variable "edge_instance_type" {
  type    = string
  default = "g6f.2xlarge"
}
variable "cloud_instance_type" {
  type    = string
  default = "g6.xlarge"
}
variable "idle_timeout_minutes" { type = number }
variable "enable_compute" {
  type    = bool
  default = false
}
variable "launch_approved" {
  validation {
    condition     = !var.enable_compute || var.launch_approved
    error_message = "Launch requires verified budget, prices, quotas and drivers."
  }
  description = "True only after budget destination, prices, quota, AMIs, GRID and runtime are verified."
  type        = bool
  default     = false
}
variable "state_bucket_name" { type = string }

variable "enable_private_endpoints" {
  description = "Billable hourly endpoints; enable only for a reviewed deployment window."
  type        = bool
  default     = false
  validation {
    # Decision D9: nodes reach SSM/ECR/S3 through their public IPv4 instead.
    condition     = !var.enable_private_endpoints
    error_message = "Interface endpoints are replaced by outbound public IPv4 (D9)."
  }
}
variable "build_http_egress" {
  description = "Phase C4 only: let the edge reach Debian's HTTP mirrors while it builds the image; turn off after."
  type        = bool
  default     = false
}
variable "budget_stop_usd" {
  description = "Decision D7: AWS Budgets stops both instances above this actual spend."
  type        = number
  default     = 165
}
