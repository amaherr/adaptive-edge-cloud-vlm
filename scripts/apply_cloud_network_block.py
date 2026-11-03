"""Host-only peer/port shaping; management and checkpoint traffic bypass netem."""

from __future__ import annotations

import argparse
import json
import subprocess

BLOCKS = {
    "N1": (2, 10),
    "N2": (5, 10),
    "N3": (10, 10),
    "N4": (25, 10),
    "N5": (50, 10),
    "N6": (100, 10),
    "N7": (10, 2.5),
    "N8": (10, 25),
    "N9": (10, 50),
}
PEERS = {"edge": "10.42.1.20", "cloud": "10.42.1.10"}
HANDLE = "7b00:"


def commands(role: str, block: str, interface: str) -> list[list[str]]:
    """Return argv arrays; only the frozen peer and experiment ports are shaped."""
    if role not in PEERS or block not in BLOCKS:
        raise ValueError("Unknown role or block")
    rate, delay = BLOCKS[block]
    result = [
        [
            "tc",
            "qdisc",
            "replace",
            "dev",
            interface,
            "root",
            "handle",
            HANDLE,
            "prio",
            "bands",
            "3",
            "priomap",
            *(["0"] * 16),
        ],
        [
            "tc",
            "qdisc",
            "replace",
            "dev",
            interface,
            "parent",
            "7b00:3",
            "handle",
            "7b01:",
            "netem",
            "delay",
            f"{delay}ms",
            "rate",
            f"{rate}mbit",
        ],
    ]
    for priority, port in enumerate((5050, 5051, 5052), 10):
        result.append(
            [
                "tc",
                "filter",
                "replace",
                "dev",
                interface,
                "parent",
                HANDLE,
                "protocol",
                "ip",
                "pref",
                str(priority),
                "flower",
                "ip_proto",
                "tcp",
                "dst_ip",
                PEERS[role],
                "dst_port" if role == "edge" else "src_port",
                str(port),
                "classid",
                "7b00:3",
            ]
        )
    return result


def run(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=PEERS, required=True)
    parser.add_argument("--block", choices=BLOCKS)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.reset == bool(args.block):
        parser.error("Specify exactly one of --block or --reset")
    route = json.loads(run(["ip", "-j", "route", "get", PEERS[args.role]]))
    interface = route[0]["dev"]
    before = json.loads(run(["tc", "-j", "qdisc", "show", "dev", interface]))
    roots = [item for item in before if item.get("root")]
    if any(
        item["handle"] != HANDLE
        and item["kind"] not in {"mq", "noqueue", "fq_codel", "pfifo_fast"}
        for item in roots
    ):
        raise RuntimeError("Refusing to replace an unrelated custom root qdisc")
    if args.reset:
        planned = (
            [["tc", "qdisc", "del", "dev", interface, "root"]]
            if any(item["handle"] == HANDLE for item in roots)
            else []
        )
    else:
        planned = commands(args.role, args.block, interface)
    print(json.dumps({"route": route, "before": before, "commands": planned}, indent=2))
    if not args.dry_run:
        for command in planned:
            subprocess.run(command, check=True)
        print(run(["tc", "-s", "qdisc", "show", "dev", interface]))
        print(run(["tc", "-s", "filter", "show", "dev", interface, "parent", HANDLE]))


if __name__ == "__main__":
    main()
