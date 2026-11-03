variable "vpc_id" { type = string }
variable "bucket_arn" { type = string }
variable "ecr_arn" { type = string }
resource "aws_security_group" "node" {
  for_each    = toset(["edge", "cloud"])
  name_prefix = "cloud-v1-${each.key}-"
  vpc_id      = var.vpc_id
  egress {
    protocol        = "tcp"
    from_port       = 443
    to_port         = 443
    cidr_blocks     = ["10.42.0.0/16"]
    prefix_list_ids = [data.aws_prefix_list.s3.id]
  }
}
data "aws_prefix_list" "s3" { name = "com.amazonaws.${data.aws_region.current.name}.s3" }
data "aws_region" "current" {}
resource "aws_security_group_rule" "inference" {
  for_each                 = toset(["5050", "5051", "5052"])
  type                     = "ingress"
  protocol                 = "tcp"
  from_port                = tonumber(each.key)
  to_port                  = tonumber(each.key)
  security_group_id        = aws_security_group.node["cloud"].id
  source_security_group_id = aws_security_group.node["edge"].id
}
resource "aws_security_group_rule" "outbound" {
  for_each                 = toset(["5050", "5051", "5052"])
  type                     = "egress"
  protocol                 = "tcp"
  from_port                = tonumber(each.key)
  to_port                  = tonumber(each.key)
  security_group_id        = aws_security_group.node["edge"].id
  source_security_group_id = aws_security_group.node["cloud"].id
}
resource "aws_iam_role" "node" {
  for_each           = toset(["edge", "cloud"])
  name_prefix        = "cloud-v1-${each.key}-"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}
resource "aws_iam_role_policy_attachment" "ssm" {
  for_each   = aws_iam_role.node
  role       = each.value.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
resource "aws_iam_role_policy" "app" {
  for_each = aws_iam_role.node
  role     = each.value.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
    { Effect = "Allow", Action = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"], Resource = var.ecr_arn },
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = var.bucket_arn, Condition = { StringLike = { "s3:prefix" = ["cloud-v1/*"] } } },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = "${var.bucket_arn}/cloud-v1/${each.key}/*" }
  ] })
}
resource "aws_iam_instance_profile" "node" {
  for_each    = aws_iam_role.node
  name_prefix = "cloud-v1-${each.key}-"
  role        = each.value.name
}
output "groups" { value = { for k, v in aws_security_group.node : k => v.id } }
output "profiles" { value = { for k, v in aws_iam_instance_profile.node : k => v.name } }
