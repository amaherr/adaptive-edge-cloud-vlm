mock_provider "vultr" {}
variables {
  admin_cidr     = "192.0.2.1/32"
  ssh_public_key = "ssh-ed25519 TEST-PUBLIC-KEY-FIXTURE"
}
run "public_access_is_ssh_from_one_address_only" {
  command = plan
  assert {
    condition     = vultr_firewall_rule.ssh.port == "22" && vultr_firewall_rule.ssh.subnet_size == 32
    error_message = "Public access must be SSH from one administrative address."
  }
}
run "broad_ssh_is_refused" {
  command = plan
  variables { admin_cidr = "0.0.0.0/0" }
  expect_failures = [var.admin_cidr]
}
