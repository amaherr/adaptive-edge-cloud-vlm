variable "instances" { type = map(string) }
variable "email" { type = string }
resource "aws_cloudwatch_log_group" "runtime" {
  name              = "/adaptive-vlm/cloud-v1"
  retention_in_days = 365
}
resource "aws_sns_topic" "alerts" { name = "adaptive-vlm-cloud-v1-alerts" }
resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.email
}
resource "aws_cloudwatch_metric_alarm" "health" {
  for_each            = var.instances
  alarm_name          = "cloud-v1-${each.key}-status"
  namespace           = "AWS/EC2"
  metric_name         = "StatusCheckFailed"
  dimensions          = { InstanceId = each.value }
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 2
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

variable "idle_minutes" { type = number }
locals {
  metrics = { idle = { name = "IdleSeconds", threshold = var.idle_minutes * 60, periods = 1 },
  storage = { name = "StorageUsedPercent", threshold = 85, periods = 2 } }
  alarms = { for pair in setproduct(keys(var.instances), keys(local.metrics)) : "${pair[0]}-${pair[1]}" => { role = pair[0], metric = local.metrics[pair[1]] } }
}
resource "aws_cloudwatch_metric_alarm" "runtime" {
  for_each            = local.alarms
  alarm_name          = "cloud-v1-${each.key}"
  namespace           = "AdaptiveVLM/CloudV1"
  metric_name         = each.value.metric.name
  dimensions          = { Role = each.value.role }
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = each.value.metric.periods
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = each.value.metric.threshold
  treat_missing_data  = "missing"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
resource "aws_cloudwatch_metric_alarm" "telemetry" {
  for_each            = var.instances
  alarm_name          = "cloud-v1-${each.key}-missing-telemetry"
  namespace           = "AdaptiveVLM/CloudV1"
  metric_name         = "IdleSeconds"
  dimensions          = { Role = each.key }
  statistic           = "SampleCount"
  period              = 60
  evaluation_periods  = 5
  comparison_operator = "LessThanThreshold"
  threshold           = 1
  treat_missing_data  = "breaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
