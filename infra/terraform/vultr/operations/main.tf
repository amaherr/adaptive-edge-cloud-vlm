variable "artifact_bucket" { type = string }
variable "state_bucket" { type = string }
variable "package_zip" { type = string }
variable "notification_email" {
  type    = string
  default = "a.maher1506@gmail.com"
}
variable "prefix" {
  type    = string
  default = "research/adaptive-vlm/vultr-v1"
  validation {
    condition     = var.prefix == "research/adaptive-vlm/vultr-v1"
    error_message = "The watchdog may access only its dedicated Vultr prefix."
  }
}
variable "vultr_key_parameter" {
  type    = string
  default = "/adaptive-vlm/vultr/api-key"
}
variable "enable_schedule" {
  type    = bool
  default = false # Enable only after SNS confirmation, secret setup and watchdog test.
}
data "aws_caller_identity" "current" {}
locals {
  name        = "adaptive-vlm-vultr"
  artifact    = "arn:aws:s3:::${var.artifact_bucket}"
  state       = "arn:aws:s3:::${var.state_bucket}"
  package_key = "${var.prefix}/control/packages/${filesha256(var.package_zip)}.zip"
  secret_arn  = "arn:aws:ssm:us-east-1:${data.aws_caller_identity.current.account_id}:parameter${var.vultr_key_parameter}"
}
resource "aws_sns_topic" "alerts" { name = "${local.name}-alerts" }
resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.notification_email
}
resource "aws_s3_object" "package" {
  bucket                 = var.artifact_bucket
  key                    = local.package_key
  source                 = var.package_zip
  source_hash            = filesha256(var.package_zip)
  server_side_encryption = "AES256"
}
resource "aws_cloudwatch_log_group" "watchdog" {
  name              = "/aws/lambda/${local.name}-watchdog"
  retention_in_days = 7
}
resource "aws_cloudwatch_log_group" "teardown" {
  name              = "/aws/codebuild/${local.name}-teardown"
  retention_in_days = 7
}
resource "aws_iam_role" "watchdog" {
  name               = "${local.name}-watchdog"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}
resource "aws_iam_role" "teardown" {
  name               = "${local.name}-teardown"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "codebuild.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}
resource "aws_iam_role_policy" "watchdog" {
  role = aws_iam_role.watchdog.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = local.artifact },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = "${local.artifact}/${var.prefix}/*" },
    { Effect = "Allow", Action = ["ssm:GetParameter"], Resource = local.secret_arn },
    { Effect = "Allow", Action = ["sns:Publish"], Resource = aws_sns_topic.alerts.arn },
    { Effect = "Allow", Action = ["codebuild:StartBuild", "codebuild:BatchGetBuilds"], Resource = [aws_codebuild_project.teardown.arn, "arn:aws:codebuild:us-east-1:${data.aws_caller_identity.current.account_id}:build/${local.name}-teardown:*"] },
    { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.watchdog.arn}:*" }
  ] })
}
resource "aws_iam_role_policy" "teardown" {
  role = aws_iam_role.teardown.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = "${local.artifact}/${var.prefix}/*" },
    { Effect = "Allow", Action = ["s3:ListBucket", "s3:GetBucketLocation"], Resource = local.state },
    { Effect = "Allow", Action = ["s3:GetBucketLocation", "s3:GetBucketAcl"], Resource = local.artifact },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], Resource = "${local.state}/vultr-v1/gpu/*" },
    { Effect = "Allow", Action = ["ssm:GetParameters"], Resource = local.secret_arn },
    { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.teardown.arn}:*" }
  ] })
}
resource "aws_codebuild_project" "teardown" {
  name                   = "${local.name}-teardown"
  service_role           = aws_iam_role.teardown.arn
  build_timeout          = 15
  queued_timeout         = 5
  concurrent_build_limit = 1
  artifacts { type = "NO_ARTIFACTS" }
  environment {
    type            = "LINUX_CONTAINER"
    compute_type    = "BUILD_GENERAL1_SMALL"
    image           = "aws/codebuild/standard:7.0" # Control plane only; never the research image.
    privileged_mode = false
    environment_variable {
      name  = "BUCKET"
      value = var.artifact_bucket
    }
    environment_variable {
      name  = "PREFIX"
      value = var.prefix
    }
    environment_variable {
      name  = "STATE_BUCKET"
      value = var.state_bucket
    }
    environment_variable {
      name  = "VULTR_API_KEY"
      value = var.vultr_key_parameter
      type  = "PARAMETER_STORE"
    }
  }
  source {
    type     = "S3"
    location = "${var.artifact_bucket}/${aws_s3_object.package.key}"
    buildspec = yamlencode({ version = "0.2", phases = {
      install = { commands = [
        "curl --fail --silent --show-error -o /tmp/terraform.zip https://releases.hashicorp.com/terraform/1.10.5/terraform_1.10.5_linux_amd64.zip",
        "echo '0566a24f5332098b15716ebc394be503f4094acba5ba529bf5eb0698ed5e2a90  /tmp/terraform.zip' | sha256sum -c -",
        "unzip -o /tmp/terraform.zip -d /usr/local/bin terraform",
        "python -m pip install boto3==1.40.76"
      ] },
      build = { commands = ["python teardown.py"] }
    } })
  }
  logs_config {
    cloudwatch_logs {
      group_name = aws_cloudwatch_log_group.teardown.name
    }
  }
}
resource "aws_lambda_function" "watchdog" {
  function_name    = "${local.name}-watchdog"
  role             = aws_iam_role.watchdog.arn
  handler          = "watchdog.handler"
  runtime          = "python3.12"
  filename         = var.package_zip
  source_code_hash = filebase64sha256(var.package_zip)
  timeout          = 90
  memory_size      = 128
  environment {
    variables = {
      BUCKET              = var.artifact_bucket, PREFIX = var.prefix, TOPIC_ARN = aws_sns_topic.alerts.arn,
      VULTR_KEY_PARAMETER = var.vultr_key_parameter, TEARDOWN_PROJECT = aws_codebuild_project.teardown.name
    }
  }
  depends_on = [aws_iam_role_policy.watchdog]
}
resource "aws_cloudwatch_event_rule" "watchdog" {
  name                = "${local.name}-watchdog"
  schedule_expression = "rate(5 minutes)"
  state               = var.enable_schedule ? "ENABLED" : "DISABLED"
}
resource "aws_cloudwatch_event_target" "watchdog" {
  rule = aws_cloudwatch_event_rule.watchdog.name
  arn  = aws_lambda_function.watchdog.arn
}
resource "aws_lambda_permission" "schedule" {
  statement_id  = "AllowScheduledWatchdog"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.watchdog.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.watchdog.arn
}
resource "aws_cloudwatch_metric_alarm" "watchdog_errors" {
  alarm_name          = "${local.name}-watchdog-errors"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions          = { FunctionName = aws_lambda_function.watchdog.function_name }
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
resource "aws_sns_topic_policy" "cloudwatch" {
  arn = aws_sns_topic.alerts.arn
  policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect    = "Allow", Principal = { Service = "cloudwatch.amazonaws.com" },
    Action    = "sns:Publish", Resource = aws_sns_topic.alerts.arn,
    Condition = { StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id } }
  }] })
}
output "topic_arn" { value = aws_sns_topic.alerts.arn }
output "watchdog_name" { value = aws_lambda_function.watchdog.function_name }
output "schedule_enabled" { value = var.enable_schedule }
output "vultr_key_parameter" { value = var.vultr_key_parameter }

resource "aws_cloudwatch_metric_alarm" "teardown_errors" {
  alarm_name          = "${local.name}-teardown-errors"
  namespace           = "AWS/CodeBuild"
  metric_name         = "FailedBuilds"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions          = { ProjectName = aws_codebuild_project.teardown.name }
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# A short STS session may fetch only the existing immutable application image.
# No long-lived AWS credentials or ECR write authority belong on a GPU VM.
resource "aws_iam_role" "image_pull" {
  name = "${local.name}-image-pull"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Principal = { AWS = data.aws_caller_identity.current.arn }, Action = "sts:AssumeRole"
  }] })
}
resource "aws_iam_role_policy" "image_pull" {
  role = aws_iam_role.image_pull.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
    { Effect = "Allow", Action = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"], Resource = "arn:aws:ecr:us-east-1:${data.aws_caller_identity.current.account_id}:repository/adaptive-vlm-cloud-v1" }
  ] })
}
output "image_pull_role_arn" { value = aws_iam_role.image_pull.arn }
