variable "vpc_id" { type = string }
variable "bucket_arn" { type = string }
variable "ecr_arn" { type = string }
resource "aws_security_group" "node" {
  for_each    = toset(["edge", "cloud"])
  name_prefix = "cloud-v1-${each.key}-"
  vpc_id      = var.vpc_id
}
# Decision D9: HTTPS out to AWS public endpoints (SSM, ECR, S3, CloudWatch).
# There is no inbound rule from the internet.
resource "aws_vpc_security_group_egress_rule" "https" {
  for_each          = aws_security_group.node
  security_group_id = each.value.id
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = "0.0.0.0/0"
}
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
    { Effect = "Allow", Action = ["cloudwatch:PutMetricData"], Resource = "*", Condition = { StringEquals = { "cloudwatch:namespace" = "AdaptiveVLM/CloudV1" } } },
    { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"], Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/adaptive-vlm/cloud-v1:*" },
    { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
    { Effect = "Allow", Action = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"], Resource = var.ecr_arn },
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = var.bucket_arn, Condition = { StringLike = { "s3:prefix" = ["cloud-v1/*"] } } },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = "${var.bucket_arn}/cloud-v1/${each.key}/*" },
    # Phase C5: both hosts seed their data volume from the shared prefix.
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${var.bucket_arn}/cloud-v1/data/*" }
  ] })
}
# Decision D3: the edge supervisor controls the cloud host through SSM
# (tc, docker exec, cloud load, heartbeat), and nothing else.
resource "aws_iam_role_policy" "edge_ssm" {
  role = aws_iam_role.node["edge"].id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["ssm:SendCommand"], Resource = "arn:aws:ssm:${data.aws_region.current.name}::document/AWS-RunShellScript" },
    { Effect = "Allow", Action = ["ssm:SendCommand"], Resource = "arn:aws:ec2:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:instance/*", Condition = { StringEquals = { "ssm:resourceTag/Role" = "cloud", "ssm:resourceTag/Project" = "adaptive-edge-cloud-vlm" } } },
    { Effect = "Allow", Action = ["ssm:GetCommandInvocation", "ssm:ListCommandInvocations"], Resource = "*" }
  ] })
}
resource "aws_iam_instance_profile" "node" {
  for_each    = aws_iam_role.node
  name_prefix = "cloud-v1-${each.key}-"
  role        = each.value.name
}
output "groups" { value = { for k, v in aws_security_group.node : k => v.id } }
output "profiles" { value = { for k, v in aws_iam_instance_profile.node : k => v.name } }

data "aws_caller_identity" "current" {}
