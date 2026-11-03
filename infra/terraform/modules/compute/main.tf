variable "enabled" { type = bool }
variable "approved" { type = bool }
variable "subnet_id" { type = string }
variable "az" { type = string }
variable "amis" { type = map(string) }
variable "groups" { type = map(string) }
variable "profiles" { type = map(string) }
variable "data_volume_gib" { type = number }
variable "max_hours" {
  type = number
  validation {
    condition     = var.max_hours > 0 && var.max_hours <= 24
    error_message = "Runtime must be within (0,24] hours."
  }
}
variable "idle_minutes" {
  type = number
  validation {
    condition     = var.idle_minutes >= 5
    error_message = "Idle timeout must be at least five minutes."
  }
}
locals { nodes = var.enabled ? { edge = "g6f.2xlarge", cloud = "g6.2xlarge" } : {} }
resource "aws_ebs_volume" "data" {
  for_each          = local.nodes
  availability_zone = var.az
  size              = var.data_volume_gib
  type              = "gp3"
  encrypted         = true
  tags              = { Role = each.key }
  lifecycle { prevent_destroy = true }
}
resource "aws_instance" "node" {
  for_each                    = local.nodes
  ami                         = var.amis[each.key]
  instance_type               = each.value
  subnet_id                   = var.subnet_id
  private_ip                  = each.key == "edge" ? "10.42.1.10" : "10.42.1.20"
  associate_public_ip_address = false
  vpc_security_group_ids      = [var.groups[each.key]]
  iam_instance_profile        = var.profiles[each.key]
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  root_block_device {
    encrypted   = true
    volume_type = "gp3"
    volume_size = 40
  }
  user_data = templatefile("${path.module}/../../../cloud-init/${each.key}.yaml", { max_seconds = ceil(var.max_hours * 3600), idle_seconds = ceil(var.idle_minutes * 60) })
  tags      = { Role = each.key }
  lifecycle {
    precondition {
      condition     = var.approved
      error_message = "Budget/AMI/GRID/price/quota verification required before GPU launch."
    }
  }
}
resource "aws_volume_attachment" "data" {
  for_each    = local.nodes
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.data[each.key].id
  instance_id = aws_instance.node[each.key].id
}
output "instance_ids" { value = { for k, v in aws_instance.node : k => v.id } }
output "volume_ids" { value = { for k, v in aws_ebs_volume.data : k => v.id } }
