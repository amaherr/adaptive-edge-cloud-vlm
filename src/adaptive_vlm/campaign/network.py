"""netem shaping with an explicit queue limit, drop accounting and live probes.

Measurement tolerances are G2's (protocol section 5): iperf3 receiver rate
within 20% of the target in both directions, echo RTT within max(5 ms, 20% of
the added RTT) of the unshaped baseline plus the added RTT.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Runner = Callable[[Sequence[str]], str]

_RATE_UNITS = {"bit": 1e-6, "Kbit": 1e-3, "Mbit": 1.0, "Gbit": 1e3}
_TIME_UNITS = {"us": 1e-3, "ms": 1.0, "s": 1e3}


@dataclass(frozen=True)
class QdiscState:
    kind: str
    rate_mbps: float | None
    delay_ms: float | None
    limit_packets: int | None
    sent_packets: int | None
    dropped: int | None
    overlimits: int | None
    raw: str


def netem_command(
    interface: str, *, rate_mbps: float, delay_ms: float, limit_packets: int
) -> list[str]:
    return [
        "sudo",
        "-n",
        "tc",
        "qdisc",
        "replace",
        "dev",
        interface,
        "root",
        "netem",
        "delay",
        f"{delay_ms:g}ms",
        "rate",
        f"{rate_mbps:g}mbit",
        "limit",
        str(limit_packets),
    ]


def delete_command(interface: str) -> list[str]:
    return ["sudo", "-n", "tc", "qdisc", "del", "dev", interface, "root"]


def show_command(interface: str) -> list[str]:
    return ["tc", "-s", "qdisc", "show", "dev", interface]


def parse_qdisc(output: str) -> QdiscState:
    """Parse the root qdisc from ``tc -s qdisc show`` output."""

    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty tc output")
    first = lines[0]
    kind_match = re.match(r"qdisc\s+(\S+)", first)
    if kind_match is None:
        raise ValueError(f"unrecognised tc output: {first}")
    rate = re.search(r"\brate\s+([\d.]+)(bit|Kbit|Mbit|Gbit)", first)
    delay = re.search(r"\bdelay\s+([\d.]+)(us|ms|s)\b", first)
    limit = re.search(r"\blimit\s+(\d+)", first)
    stats = re.search(
        r"Sent\s+\d+\s+bytes\s+(\d+)\s+pkt\s+\(dropped\s+(\d+),\s+overlimits\s+(\d+)",
        output,
    )
    return QdiscState(
        kind=kind_match[1],
        rate_mbps=float(rate[1]) * _RATE_UNITS[rate[2]] if rate else None,
        delay_ms=float(delay[1]) * _TIME_UNITS[delay[2]] if delay else None,
        limit_packets=int(limit[1]) if limit else None,
        sent_packets=int(stats[1]) if stats else None,
        dropped=int(stats[2]) if stats else None,
        overlimits=int(stats[3]) if stats else None,
        raw=first,
    )


def check_netem(
    state: QdiscState,
    *,
    interface: str,
    rate_mbps: float,
    delay_ms: float,
    limit_packets: int,
) -> None:
    if state.kind != "netem":
        raise RuntimeError(f"expected netem on {interface}: {state.raw}")
    if state.rate_mbps is None or abs(state.rate_mbps - rate_mbps) > 0.01:
        raise RuntimeError(f"wrong rate on {interface}: {state.raw}")
    if state.delay_ms is None or abs(state.delay_ms - delay_ms) > 0.01:
        raise RuntimeError(f"wrong delay on {interface}: {state.raw}")
    if state.limit_packets != limit_packets:
        raise RuntimeError(f"wrong queue limit on {interface}: {state.raw}")
    if state.dropped is None:
        raise RuntimeError(f"no drop statistics on {interface}: {state.raw}")


def is_default_qdisc(state: QdiscState) -> bool:
    return state.kind in {"mq", "noqueue", "fq_codel", "pfifo_fast"}


def measure_network(
    host: str,
    iperf_port: int,
    echo_port: int,
    *,
    echo_script: Path,
    run: Runner | None = None,
) -> dict[str, Any]:
    """Two 8 s iperf3 runs (edge to cloud, then reverse) and 10 TCP echoes."""

    execute = run or _run
    iperf = []
    for reverse in (False, True):
        command = ["iperf3", "-c", host, "-p", str(iperf_port), "-t", "8", "-O", "1"]
        iperf.append(
            json.loads(execute([*command, "-J", *(["-R"] if reverse else [])]))
        )
    echo = json.loads(
        execute(
            [
                sys.executable,
                str(echo_script),
                "client",
                "--host",
                host,
                "--port",
                str(echo_port),
                "--count",
                "10",
                "--warmup",
                "2",
            ]
        )
    )
    forward, backward = iperf
    return {
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "forward_iperf": forward,
        "reverse_iperf": backward,
        "echo": echo,
        "edge_to_cloud_receiver_bps": forward["end"]["sum_received"]["bits_per_second"],
        "cloud_to_edge_receiver_bps": backward["end"]["sum_received"][
            "bits_per_second"
        ],
        "edge_sender_retransmissions": forward["end"]["sum_sent"].get("retransmits"),
        "cloud_sender_retransmissions": backward["end"]["sum_sent"].get("retransmits"),
        "echo_median_rtt_ms": echo["rtt_ms"]["median"],
    }


def measurement_failures(
    measurement: dict[str, Any],
    *,
    rate_mbps: float,
    added_rtt_ms: float,
    baseline_rtt_ms: float,
) -> list[str]:
    """Reasons a live measurement misses the target condition (empty = pass)."""

    failures = []
    target = rate_mbps * 1_000_000
    for name in ("edge_to_cloud_receiver_bps", "cloud_to_edge_receiver_bps"):
        actual = float(measurement[name])
        if not 0.8 * target <= actual <= 1.2 * target:
            failures.append(f"{name}={actual:.0f} outside {rate_mbps} Mb/s ±20%")
    expected = baseline_rtt_ms + added_rtt_ms
    actual_rtt = float(measurement["echo_median_rtt_ms"])
    if abs(actual_rtt - expected) > max(5.0, 0.2 * added_rtt_ms):
        failures.append(f"echo RTT {actual_rtt:.2f} ms, expected {expected:.2f} ms")
    return failures


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), check=True, capture_output=True, text=True, timeout=90
    )
    return result.stdout


def ssh_runner(target: str, *, timeout: float = 90) -> Runner:
    """Run a command on the cloud Linux host over key-based SSH."""

    import shlex

    def run(command: Sequence[str]) -> str:
        result = subprocess.run(
            [
                "ssh",
                "-oBatchMode=yes",
                "-oConnectTimeout=10",
                "-oServerAliveInterval=15",
                "-oServerAliveCountMax=2",
                target,
                shlex.join(command),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout

    return run


class PhysicalNetwork:
    """Shape both hosts' egress with netem and probe the live path."""

    controlled = True

    def __init__(
        self,
        *,
        edge_interface: str,
        cloud_interface: str,
        cloud_host: str,
        iperf_port: int,
        echo_port: int,
        limit_packets: int,
        echo_script: Path,
        local: Runner | None = None,
        remote: Runner,
    ) -> None:
        self.edge_interface = edge_interface
        self.cloud_interface = cloud_interface
        self.cloud_host = cloud_host
        self.iperf_port = iperf_port
        self.echo_port = echo_port
        self.limit_packets = limit_packets
        self.echo_script = echo_script
        self.local = local or _run
        self.remote = remote

    def _peers(self) -> tuple[tuple[str, Runner, str], ...]:
        return (
            ("edge", self.local, self.edge_interface),
            ("cloud", self.remote, self.cloud_interface),
        )

    def qdiscs(self) -> dict[str, QdiscState]:
        return {
            name: parse_qdisc(run(show_command(interface)))
            for name, run, interface in self._peers()
        }

    def apply(self, *, rate_mbps: float, delay_ms: float, owned: bool) -> None:
        """Replace both root qdiscs; refuse to overwrite a foreign qdisc."""

        for name, state in self.qdiscs().items():
            if not is_default_qdisc(state) and not (owned and state.kind == "netem"):
                raise RuntimeError(
                    f"unexpected {name} qdisc; inspect first: {state.raw}"
                )
        for _, run, interface in self._peers():
            run(
                netem_command(
                    interface,
                    rate_mbps=rate_mbps,
                    delay_ms=delay_ms,
                    limit_packets=self.limit_packets,
                )
            )

    def verify(self, *, rate_mbps: float, delay_ms: float) -> dict[str, QdiscState]:
        states = self.qdiscs()
        for (_, _, interface), state in zip(
            self._peers(), states.values(), strict=True
        ):
            check_netem(
                state,
                interface=interface,
                rate_mbps=rate_mbps,
                delay_ms=delay_ms,
                limit_packets=self.limit_packets,
            )
        return states

    def clear(self) -> list[str]:
        """Remove netem from both hosts; return per-host errors."""

        errors = []
        for name, run, interface in self._peers():
            try:
                if parse_qdisc(run(show_command(interface))).kind == "netem":
                    run(delete_command(interface))
                if parse_qdisc(run(show_command(interface))).kind == "netem":
                    raise RuntimeError("netem remains after cleanup")
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        return errors

    def measure(self) -> dict[str, Any]:
        return measure_network(
            self.cloud_host,
            self.iperf_port,
            self.echo_port,
            echo_script=self.echo_script,
            run=self.local,
        )


class LoopbackNetwork:
    """Rehearsal stand-in: no shaping, no probes; records an uncontrolled path."""

    controlled = False

    def qdiscs(self) -> dict[str, QdiscState]:
        return {}

    def apply(self, *, rate_mbps: float, delay_ms: float, owned: bool) -> None:
        return None

    def verify(self, *, rate_mbps: float, delay_ms: float) -> dict[str, QdiscState]:
        return {}

    def clear(self) -> list[str]:
        return []

    def measure(self) -> dict[str, Any]:
        return {"collected_at_utc": datetime.now(UTC).isoformat(), "controlled": False}
