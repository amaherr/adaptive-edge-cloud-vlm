# Vultr private-network and SSH foundation

Private VPC `10.46.0.0/24` in Atlanta, a dedicated public SSH key and firewall
allowing only TCP 22 from the operator's exact IPv4 /32. No public 5050–5052
rule, NAT gateway, GPU, registry or storage subscription is provisioned here.
Host firewall defense is supplied separately through nonsecret cloud-init.

Backend: existing protected S3 state bucket, key
`vultr-v1/foundation/terraform.tfstate`; it is independent of disposable compute.
Private SSH keys are generated outside Terraform (the current preflight key is
in a mode-600 `/tmp` file). Only the public key is a variable/state value.
Operator CIDR/public key settings belong in ignored local tfvars. Use
`scripts/vultr/terraform.sh`; plan and review before apply.
