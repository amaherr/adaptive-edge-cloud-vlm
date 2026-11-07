variable "admin_cidr" {
  type = string
  validation {
    condition     = can(cidrnetmask(var.admin_cidr)) && endswith(var.admin_cidr, "/32")
    error_message = "Use the operator's exact IPv4 /32, never a broad public SSH range."
  }
}
variable "ssh_public_key" {
  type = string
  validation {
    condition     = startswith(var.ssh_public_key, "ssh-ed25519 ")
    error_message = "Provide only the dedicated SSH public key, never a private key."
  }
}
resource "vultr_vpc" "experiment" {
  region         = "atl"
  description    = "adaptive-vlm-vultr-v1-private-experiment"
  v4_subnet      = "10.46.0.0"
  v4_subnet_mask = 24
}
resource "vultr_ssh_key" "operator" {
  name    = "adaptive-vlm-vultr-preflight"
  ssh_key = var.ssh_public_key
}
resource "vultr_firewall_group" "control" {
  description = "adaptive-vlm-vultr-v1-control-only"
}
resource "vultr_firewall_rule" "ssh" {
  firewall_group_id = vultr_firewall_group.control.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = split("/", var.admin_cidr)[0]
  subnet_size       = 32
  port              = "22"
  notes             = "Operator only; inference ports have no public rule"
}
output "vpc_id" { value = vultr_vpc.experiment.id }
output "ssh_key_id" { value = vultr_ssh_key.operator.id }
output "firewall_group_id" { value = vultr_firewall_group.control.id }
