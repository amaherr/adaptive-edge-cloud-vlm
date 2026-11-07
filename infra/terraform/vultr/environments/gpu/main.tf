variable "deploy_gpu_instances" {
  type    = bool
  default = false
}
variable "region" {
  type    = string
  default = "atl"
}
variable "campaign_tag" {
  type    = string
  default = "adaptive-vlm-vultr-v1"
  validation {
    condition     = startswith(var.campaign_tag, "adaptive-vlm-vultr-")
    error_message = "Use a dedicated Vultr campaign tag."
  }
}
variable "launch_approval_id" {
  type    = string
  default = ""
}
variable "expires_at" {
  type    = string
  default = ""
}
variable "cloud_init" {
  type    = string
  default = ""
}
variable "nodes" {
  type = map(object({
    plan_id           = string
    os_id             = number
    ssh_key_ids       = list(string)
    vpc_ids           = list(string)
    firewall_group_id = string
  }))
  default = {}
  validation {
    condition     = length(var.nodes) <= 2 && alltrue([for key in keys(var.nodes) : contains(["edge", "cloud"], key)])
    error_message = "At most the edge and cloud roles may be provisioned."
  }
}
resource "vultr_instance" "gpu" {
  for_each          = var.deploy_gpu_instances ? var.nodes : {}
  region            = var.region
  plan              = each.value.plan_id
  os_id             = each.value.os_id
  ssh_key_ids       = each.value.ssh_key_ids
  vpc_ids           = each.value.vpc_ids
  firewall_group_id = each.value.firewall_group_id
  label             = "${var.campaign_tag}-${each.key}"
  hostname          = "vlm-${each.key}"
  tags              = [var.campaign_tag, "role-${each.key}", "expires-${replace(var.expires_at, "/[^0-9]/", "")}"]
  # Provider v2.33.2 encodes user_data when constructing the API request.
  user_data        = var.cloud_init == "" ? null : var.cloud_init
  backups          = "disabled"
  enable_ipv6      = false
  ddos_protection  = false
  activation_email = false
  lifecycle {
    precondition {
      condition     = length(var.launch_approval_id) > 0
      error_message = "GPU launch requires a recorded owner approval."
    }
    precondition {
      condition     = can(formatdate("YYYY", var.expires_at))
      error_message = "GPU launch requires a valid expiry timestamp."
    }
    precondition {
      condition     = try(timecmp(var.expires_at, timestamp()) > 0, false)
      error_message = "GPU launch expiry must be in the future."
    }
    precondition {
      condition     = length(each.value.ssh_key_ids) > 0 && length(each.value.vpc_ids) > 0 && each.value.firewall_group_id != ""
      error_message = "Verified SSH, private networking and firewall references are mandatory."
    }
  }
}
output "instances" {
  value = { for role, node in vultr_instance.gpu : role => {
    id = node.id, plan = node.plan, region = node.region, main_ip = node.main_ip
  } }
}
