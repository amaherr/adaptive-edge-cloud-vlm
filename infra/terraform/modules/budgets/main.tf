# Decision D7: one cumulative ceiling for everything on AWS (G2 v2+ and G5).
# An annual budget starting 1 November 2026 accumulates instead of resetting
# monthly. Alerts are absolute amounts; the stop action at a higher amount
# lives in the disposable stack, which knows the instance IDs.
variable "amount" {
  type = number
  validation {
    condition     = var.amount > 0
    error_message = "Choose a positive budget."
  }
}
variable "email" { type = string }
variable "period_start" {
  type    = string
  default = "2026-11-01_00:00"
}
variable "alert_usd" {
  type    = list(number)
  default = [60, 110, 150]
}
resource "aws_budgets_budget" "monthly" {
  name              = "adaptive-vlm-cloud-v1"
  budget_type       = "COST"
  limit_amount      = tostring(var.amount)
  limit_unit        = "USD"
  time_unit         = "ANNUALLY"
  time_period_start = var.period_start
  dynamic "notification" {
    for_each = toset(var.alert_usd)
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "ABSOLUTE_VALUE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.email]
    }
  }
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.email]
  }
}
output "name" { value = aws_budgets_budget.monthly.name }
