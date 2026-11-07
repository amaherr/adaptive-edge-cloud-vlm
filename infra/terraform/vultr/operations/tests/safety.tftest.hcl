mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
}
variables {
  artifact_bucket = "existing-artifact-bucket"
  state_bucket    = "existing-state-bucket"
  package_zip     = "versions.tf" # Local hash fixture; no deployment in a mock plan.
}
run "guard_defaults_to_disabled_and_correct_recipient" {
  command = plan
  assert {
    condition     = aws_cloudwatch_event_rule.watchdog.state == "DISABLED"
    error_message = "An unconfigured watchdog must not silently enable itself."
  }
  assert {
    condition     = aws_sns_topic_subscription.email.endpoint == "a.maher1506@gmail.com"
    error_message = "Alerts must use the owner-selected destination."
  }
}
run "foreign_prefix_is_refused" {
  command = plan
  variables { prefix = "aws-existing-research" }
  expect_failures = [var.prefix]
}
