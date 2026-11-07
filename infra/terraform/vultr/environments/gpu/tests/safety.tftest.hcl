mock_provider "vultr" {}
run "cloud_init_is_plaintext_for_provider_encoding" {
  command = plan
  variables {
    deploy_gpu_instances = true
    launch_approval_id   = "test-only-no-real-launch"
    expires_at          = "2099-01-01T00:00:00Z"
    cloud_init          = "#cloud-config\nssh_pwauth: false\n"
    nodes = {
      edge = {
        plan_id = "test", os_id = 2284,
        ssh_key_ids = ["test"], vpc_ids = ["test"], firewall_group_id = "test"
      }
    }
  }
  assert {
    condition     = vultr_instance.gpu["edge"].user_data == "#cloud-config\nssh_pwauth: false\n"
    error_message = "Pass raw cloud-init: Vultr provider 2.33.2 performs the API Base64 encoding."
  }
}
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
