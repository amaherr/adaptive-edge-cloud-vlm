mock_provider "vultr" {}
run "default_has_no_gpu" {
  command = plan
  assert {
    condition     = length(vultr_instance.gpu) == 0
    error_message = "The default must create no paid GPU resources."
  }
}
run "launch_without_approval_is_refused" {
  command = plan
  variables {
    deploy_gpu_instances = true
    nodes = {
      edge = {
        plan_id     = "unverified", os_id = 1,
        ssh_key_ids = ["test"], vpc_ids = ["test"], firewall_group_id = "test"
      }
    }
  }
  expect_failures = [vultr_instance.gpu]
}
run "too_many_nodes_is_refused" {
  command = plan
  variables {
    nodes = {
      extra = { plan_id = "test", os_id = 1, ssh_key_ids = [], vpc_ids = [], firewall_group_id = "" }
    }
  }
  expect_failures = [var.nodes]
}
