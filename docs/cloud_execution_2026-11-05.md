# Live cloud migration execution — 5 November 2026

Phases 1A/T0 and 1B/D0 are **not yet complete**. The foundation is deployed;
GPU-dependent acceptance is blocked by AWS's applied regional quota of zero.

## Verified live evidence

- Native WSL AWS CLI 2.39.4 authenticates; Docker daemon 29.1.2 is accessible.
- Both proposed instance types are offered in the account's us-east-1a/b/c/d.
- Current AWS Price List products confirm $0.475/hour edge, $0.9776/hour cloud,
  $0.08/GB-month gp3, and $0.01/interface-endpoint-hour.
- Terraform submitted a request for 16 On-Demand G/VT vCPUs. A successful quota
  request apply is not an approved quota. AWS reports the request pending.
- Terraform created the protected state bucket and 32 foundation resources:
  private VPC/subnet/routes/S3 gateway, instance roles/security groups, private
  versioned artifact bucket, immutable ECR, logs, SNS and the $20 monthly budget.
  Four budget notifications cover actual 50/80/100% and forecast 100%.
- The user confirmed the SNS email subscription; AWS lists a real subscription
  ARN and accepted a delivery-test publication. The user confirmed receiving\n  the test email, completing the delivery check.
- S3 backend lock contention was tested with two Terraform operations: the
  second was rejected while the first held the lock, then the lock was released.
- The initial live drift check caught mixed inline/standalone egress ownership.
  Existing HTTPS rules were imported into standalone resources; their tags were
  reconciled through a reviewed plan. The next live plan reports **no changes**.
- Bootstrap and account-quota states were migrated into separate encrypted,
  versioned S3 keys using the committed backend templates. The experiment uses
  its own separate S3 key. No state or credentials are committed.
- The single application image built from app commit `a659c59` with the original
  model lock. Installed-version checks and pip check passed. An offline CPU
  fake-inference smoke passed as UID 10001 with no network, dropped capabilities
  and a read-only root filesystem. This is not evidence of GPU equivalence.

Raw outputs, pricing products, image metadata and logs are under ignored
`artifacts/infra/`. Deployment evidence is archived privately, not mixed into
physical profiling results. The image's application Git revision is recorded
separately from later infrastructure commits; both nodes must use one digest.

## Work prepared while quota approval is pending

`infra/terraform/host-image` defines a disabled-by-default EC2 Image Builder
recipe. Its Canonical Ubuntu base and AWS GRID 19.5 installer were verified via
AWS read APIs; the installer was downloaded and SHA256-pinned. The recipe installs
host tooling and tests the driver. It has not run on AWS and must not be described
as a verified AMI. No builder GPU, experiment GPU, data EBS volume, NAT gateway
or hourly interface endpoint has been launched by this work.

The Docker desktop credential helper failed during ECR login. The publisher now
uses an isolated temporary Docker configuration for the short-lived ECR token,
without changing global credentials. Publication/digest evidence must be checked
in image-deployment.json before declaring the image available in ECR.

## Remaining acceptance

Retention was separated after the initial foundation apply. Seven protected
storage/budget resources were transferred through import and destroy=false
state removal, with no AWS changes. Future data volumes belong to the retained
stack, so disposable compute teardown cannot delete them. A full disposable
destroy plan was reviewed and not applied: 29 current foundation deletions,
zero retained S3/ECR/budget/EBS/backend deletions. See the retained-stack README.
Both stacks require clean final plans. The frozen inputs were hash-verified
again, including every remote-code file, tokenizer and 3.85 GB model weights.

Before any data volumes were created, their default was reduced to 50 GiB per
node based on the verified 3.7 GiB model cache and 39 MiB image dataset. Revised
root+data EBS storage is $14.40/month while all four volumes exist; initial
380 GiB estimates are superseded. Future profiling must still monitor free space.

Wait for the **applied** quota to reach at least 16, then review the host-image
build window and plan, build/test the private AMI, and freeze its ID and inventory.
Review/apply the paired-node plan with budget/notification destinations confirmed.
Test drivers/BF16, exact model correctness on all three actual paths, persistence,
host/container comparison, measured network shaping, telemetry and the stop guard.
Review retention-preserving teardown and archive actual evidence before passing
T0/D0. The S3 campaign checkpoint manager and Gate R0 remain later work.

After T0/D0 pass, deliver the user-requested detailed cost plan for the entire
bachelor project: remaining gates, 8,100-attempt profiling, calibration, final
evaluation, dynamic traces, ablations, reruns, idle time, storage/AMI/ECR retention,
monitoring and contingency. Use measured throughput where available and label
all unmeasured assumptions. The $20/month input is an alert threshold, not a cap.


Additional cache checks passed inside the built shared image through read-only
bind mounts: all 20 model artifacts and all 100 original images. New unit tests
reject changed weights, tokenizer and executable model code. Mutable runtime
caches are separated from read-only source inputs. These checks do not stand in
for the remaining AWS GPU tests.
