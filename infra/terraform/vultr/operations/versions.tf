terraform {
  required_version = "= 1.10.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "= 5.100.0"
    }
  }
  backend "s3" {}
}
provider "aws" {
  region = "us-east-1"
  default_tags { tags = { Project = "adaptive-edge-cloud-vlm", Campaign = "vultr-v1", ManagedBy = "Terraform" } }
}
