terraform {
  backend "s3" {
    key          = "host-image/terraform.tfstate"
    encrypt      = true
    use_lockfile = true
  }
}
