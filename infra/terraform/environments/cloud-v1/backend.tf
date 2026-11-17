terraform {
  backend "s3" {
    key          = "cloud-v1/terraform.tfstate"
    encrypt      = true
    use_lockfile = true
  }
}
