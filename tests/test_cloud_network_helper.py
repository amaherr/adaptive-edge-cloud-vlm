"""Verify host shaping cannot select HTTPS or unrelated peer traffic."""

import runpy
from pathlib import Path

import pytest

HELPER = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/apply_cloud_network_block.py")
)
BLOCKS, PEERS, commands = (HELPER[key] for key in ("BLOCKS", "PEERS", "commands"))


@pytest.mark.parametrize("role", ["edge", "cloud"])
@pytest.mark.parametrize("block", list(BLOCKS))
def test_only_experimental_peer_ports_enter_netem(role, block):
    plan = commands(role, block, "ens5")
    assert plan[0][-16:] == ["0"] * 16  # Default band bypasses shaping.
    assert "7b00:3" in plan[1]
    for port, command in zip((5050, 5051, 5052), plan[2:], strict=True):
        assert command[command.index("dst_ip") + 1] == PEERS[role]
        selector = "dst_port" if role == "edge" else "src_port"
        assert command[command.index(selector) + 1] == str(port)
        assert command[-2:] == ["classid", "7b00:3"]
        assert "443" not in command


def test_invalid_block_rejected():
    with pytest.raises(ValueError):
        commands("edge", "N10", "ens5")
