# Builds one private AMI for BOTH experiment roles. No build starts by default.
provider "aws" {
  region = "us-east-1"
  default_tags { tags = { Project = "adaptive-edge-cloud-vlm", ManagedBy = "Terraform", Purpose = "host-image" } }
}
variable "enable_build" {
  type    = bool
  default = false
}
variable "recipe_version" {
  type    = string
  default = "1.0.0"
}
variable "availability_zone" {
  type    = string
  default = "us-east-1a"
}
variable "create_service_linked_role" {
  type    = bool
  default = true
}
resource "aws_iam_service_linked_role" "imagebuilder" {
  count            = var.create_service_linked_role ? 1 : 0
  aws_service_name = "imagebuilder.amazonaws.com"
}
resource "aws_vpc" "builder" {
  cidr_block           = "10.43.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
}
resource "aws_subnet" "builder" {
  vpc_id                  = aws_vpc.builder.id
  cidr_block              = "10.43.1.0/24"
  availability_zone       = var.availability_zone
  map_public_ip_on_launch = true
}
resource "aws_internet_gateway" "builder" { vpc_id = aws_vpc.builder.id }
resource "aws_route_table" "builder" {
  vpc_id = aws_vpc.builder.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.builder.id
  }
}
resource "aws_route_table_association" "builder" {
  subnet_id      = aws_subnet.builder.id
  route_table_id = aws_route_table.builder.id
}
resource "aws_security_group" "builder" {
  vpc_id  = aws_vpc.builder.id
  ingress = []
  # No inbound rules, including no SSH or inference ports.
  egress {
    protocol    = "tcp"
    from_port   = 80
    to_port     = 80
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
  }
}
resource "aws_iam_role" "builder" {
  name_prefix        = "cloud-vlm-host-builder-"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}
resource "aws_iam_role_policy_attachment" "builder" {
  for_each   = toset(["EC2InstanceProfileForImageBuilder", "AmazonSSMManagedInstanceCore"])
  role       = aws_iam_role.builder.name
  policy_arn = "arn:aws:iam::aws:policy/${each.value}"
}
resource "aws_iam_role_policy" "grid" {
  role   = aws_iam_role.builder.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = "s3:GetObject", Resource = "arn:aws:s3:::ec2-linux-nvidia-drivers/grid-19.5/NVIDIA-Linux-x86_64-580.159.03-grid-aws.run" }] })
}
resource "aws_iam_instance_profile" "builder" { role = aws_iam_role.builder.name }
resource "aws_imagebuilder_component" "runtime" {
  name     = "adaptive-vlm-host-runtime"
  platform = "Linux"
  version  = var.recipe_version
  data = yamlencode({ name = "InstallCloudVLMHost", schemaVersion = "1.0", phases = [
    { name = "build", steps = [{ name = "InstallPinnedGridAndHostTools", action = "ExecuteBash", timeoutSeconds = 3600, onFailure = "Abort", inputs = { commands = [file("${path.module}/../../runtime/build-host.sh")] } }] },
    { name = "test", steps = [{ name = "VerifyGpuAndHostTools", action = "ExecuteBash", timeoutSeconds = 600, onFailure = "Abort", inputs = { commands = ["set -eu; nvidia-smi; test $(nvidia-smi --query-gpu=driver_version --format=csv,noheader) = 580.159.03; docker compose version; nvidia-ctk --version; test -x /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl"] } }] }
  ] })
}
resource "aws_imagebuilder_image_recipe" "runtime" {
  name    = "adaptive-vlm-host"
  version = var.recipe_version
  # Canonical owner 099720109477, Ubuntu 22.04 amd64 server, 2026-10-21.
  parent_image = "ami-033ea4ff4a9c726dc"
  component { component_arn = aws_imagebuilder_component.runtime.arn }
  block_device_mapping {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = 40
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }
}
resource "aws_imagebuilder_infrastructure_configuration" "runtime" {
  name                          = "adaptive-vlm-host-build"
  instance_profile_name         = aws_iam_instance_profile.builder.name
  instance_types                = ["g6f.2xlarge"]
  subnet_id                     = aws_subnet.builder.id
  security_group_ids            = [aws_security_group.builder.id]
  terminate_instance_on_failure = true
  depends_on                    = [aws_route_table_association.builder, aws_iam_role_policy_attachment.builder, aws_iam_role_policy.grid]
}
resource "aws_imagebuilder_distribution_configuration" "runtime" {
  name = "adaptive-vlm-host-private"
  distribution {
    region = "us-east-1"
    ami_distribution_configuration {
      name     = "adaptive-vlm-host-${var.recipe_version}-{{ imagebuilder:buildDate }}"
      ami_tags = { Project = "adaptive-edge-cloud-vlm", ManagedBy = "Terraform", HostRecipe = var.recipe_version }
    }
  }
}
resource "aws_imagebuilder_image" "runtime" {
  count                            = var.enable_build ? 1 : 0
  image_recipe_arn                 = aws_imagebuilder_image_recipe.runtime.arn
  infrastructure_configuration_arn = aws_imagebuilder_infrastructure_configuration.runtime.arn
  distribution_configuration_arn   = aws_imagebuilder_distribution_configuration.runtime.arn
  image_tests_configuration {
    image_tests_enabled = true
    timeout_minutes     = 60
  }
  timeouts { create = "120m" }
  lifecycle { prevent_destroy = true }
  depends_on = [aws_iam_service_linked_role.imagebuilder]
}
output "images" { value = aws_imagebuilder_image.runtime[*].output_resources }
