mock_provider "aws" {}
run "no_image_build_by_default" {
  command = plan
  assert {
    condition     = length(aws_imagebuilder_image.runtime) == 0
    error_message = "Host image creation must require enable_build=true."
  }
  assert {
    condition     = length(aws_security_group.builder.ingress) == 0
    error_message = "The temporary builder must not expose inbound services."
  }
}
