variable "region" { type = string }
variable "az" { type = string }
variable "enable_private_endpoints" { type = bool }
resource "aws_vpc" "this" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
}
resource "aws_subnet" "this" {
  vpc_id                  = aws_vpc.this.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = var.az
  map_public_ip_on_launch = false
}
resource "aws_route_table" "this" { vpc_id = aws_vpc.this.id }
resource "aws_route_table_association" "this" {
  subnet_id      = aws_subnet.this.id
  route_table_id = aws_route_table.this.id
}
resource "aws_vpc_endpoint" "s3" {
  vpc_id          = aws_vpc.this.id
  service_name    = "com.amazonaws.${var.region}.s3"
  route_table_ids = [aws_route_table.this.id]
}
resource "aws_security_group" "endpoints" {
  name_prefix = "cloud-v1-endpoints-"
  vpc_id      = aws_vpc.this.id
  ingress {
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = [aws_subnet.this.cidr_block]
  }
}
resource "aws_vpc_endpoint" "services" {
  for_each            = var.enable_private_endpoints ? toset(["ssm", "ssmmessages", "ec2messages", "ecr.api", "ecr.dkr", "logs", "monitoring"]) : toset([])
  vpc_id              = aws_vpc.this.id
  service_name        = "com.amazonaws.${var.region}.${each.key}"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true
  subnet_ids          = [aws_subnet.this.id]
  security_group_ids  = [aws_security_group.endpoints.id]
}
output "vpc_id" { value = aws_vpc.this.id }
output "subnet_id" { value = aws_subnet.this.id }

output "interface_endpoint_count" { value = length(aws_vpc_endpoint.services) }
