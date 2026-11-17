terraform {
  backend "s3" {
    key          = "cloud-retained/terraform.tfstate"
    encrypt      = true
    use_lockfile = true
  }
}
