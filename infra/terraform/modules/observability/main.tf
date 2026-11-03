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
