terraform {
  required_version = "= 1.10.5"
  required_providers {
    vultr = {
      source  = "vultr/vultr"
      version = "= 2.33.2"
    }
  }
  backend "s3" {}
}
provider "vultr" {} # VULTR_API_KEY only; never a Terraform variable.
