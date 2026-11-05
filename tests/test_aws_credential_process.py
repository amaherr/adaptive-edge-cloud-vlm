"""The provider bridge must not recursively load its own generated profile."""

import json
import os
import subprocess
from pathlib import Path


def test_bridge_clears_override_config_and_static_credentials(tmp_path):
    fake = tmp_path / "aws"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "names = ['AWS_CONFIG_FILE', 'AWS_PROFILE', 'AWS_ACCESS_KEY_ID', "
        "'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_SECURITY_TOKEN']\n"
        "assert not any(name in os.environ for name in names)\n"
        "assert sys.argv[1:] == ['configure', 'export-credentials', "
        "'--profile', 'source-profile', '--format', 'process']\n"
        "print(json.dumps({'Version': 1, 'fixture': True}))\n"
    )
    fake.chmod(0o755)
    environment = {
        **os.environ,
        "AWS_CONFIG_FILE": "recursive-config",
        "AWS_PROFILE": "bridge-profile",
        "AWS_ACCESS_KEY_ID": "fixture",
        "AWS_SECRET_ACCESS_KEY": "fixture",
        "AWS_SESSION_TOKEN": "fixture",
        "AWS_SECURITY_TOKEN": "fixture",
        "CLOUD_AWS_CLI": str(fake),
        "CLOUD_AWS_SOURCE_PROFILE": "source-profile",
    }
    script = Path(__file__).parents[1] / "scripts/aws_credential_process.sh"
    result = subprocess.run(
        ["bash", str(script)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"Version": 1, "fixture": True}
    assert result.stderr == ""
