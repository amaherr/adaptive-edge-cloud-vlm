variable "amount" {
  type = number
  validation {
    condition     = var.amount > 0
    error_message = "Choose a positive monthly budget."
  }
}
variable "email" { type = string }
resource "aws_budgets_budget" "monthly" {
  name         = "adaptive-vlm-cloud-v1"
  budget_type  = "COST"
  limit_amount = tostring(var.amount)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  dynamic "notification" {
    for_each = { actual50 = [50, "ACTUAL"], actual80 = [80, "ACTUAL"], actual100 = [100, "ACTUAL"], forecast100 = [100, "FORECASTED"] }
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value[0]
      threshold_type             = "PERCENTAGE"
      notification_type          = notification.value[1]
      subscriber_email_addresses = [var.email]
    }
  }
}
