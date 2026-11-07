# Connection audit after the blocked preflight

The root cause is unknown. Escalating to support before isolating our own
configuration was premature. No further paid VM was launched for this audit.

## Confirmed configuration defect

The GPU module passed `base64encode(var.cloud_init)` as `user_data`. The
[pinned provider source](https://github.com/vultr/terraform-provider-vultr/blob/v2.33.2/vultr/resource_vultr_instance.go#L348)
also encodes this value when constructing the API request. The resulting
decoded user data is Base64 text rather than `#cloud-config`. The module now
passes plain text, with a mock plan regression test. This explains why the
intended cloud-init cannot be assumed to have executed; it does not prove the
cause of the SSH timeout. Provider SSH-key injection was a separate setting.

Validation: Terraform formatting passed; all four GPU module mock tests passed
with the pinned provider. The sandbox initially prevented the provider plugin
from opening its socket; the same mock tests passed outside that restriction.
No Terraform apply or hosted control deployment was performed for this audit.

## Gaps in the first diagnosis

- Linux and Windows shared the operator network. Two HTTPS public-IP checks
  agreed with the firewall, but neither proves TCP traffic to this Vultr IP
  followed the same path. GitHub SSH reachability only excludes a general block.
- API running/ok metadata did not prove guest SSH readiness. No guest logs,
  packet capture, or independent-network SSH probe were obtained.
- Console connection failure could also involve browser/WebSocket access.
  Another browser and network were not tested.
- The watcher observed power stopped at 13:16:10 UTC after the accepted reboot,
  before manual expiry at 13:18:30. Later probes cannot establish failure of a
  running guest. Automatic stopped-instance cleanup intentionally remained
  enabled; there was no reboot maintenance coordination.
- Guest firewall, private VPC, image and provider firewall were introduced
  together. We did not isolate the intended host configuration from a baseline.

## Controlled follow-up

Prepare one fresh A16 instance with a fresh SSH key and exact operator /32
provider firewall, retaining the independent $5 phase/$80 total controls and
carrying forward the first attempt's cost. Use minimal plain-text cloud-init
without guest firewall changes initially. Do not expose inference ports.
Verify public IPv4 settings, assigned firewall/key IDs and the API user-data
round trip; record provisioning/power transitions and verbose SSH TCP results.
Once reachable, inspect cloud-init, routes, SSH and UFW before introducing the
intended guest firewall rules. Keep VPC changes separate from public SSH tests.

If needed, test from another trusted network with its own exact /32 allowlist
and try the console in another browser/network. A connection timeout before
authentication will not be repaired by changing the SSH username or key alone.
Do not reboot blindly: first preserve diagnostics and coordinate a short bounded
maintenance window with teardown controls. Stop/expiry/budget limits must not
be disabled to diagnose access. If the VM remains stopped, cleanup is correct.

Only after connectivity works should GPU drivers, BF16 and model fit be tested.
Changing the Ubuntu image or asking support for host/console logs remains an
option if the corrected, isolated configuration still fails. No provider fault,
GPU feasibility or automatic image-driver compatibility is established yet.
