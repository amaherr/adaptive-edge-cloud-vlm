"""Measure independent WSL-to-WSL application RTT with sequential byte echoes."""

from __future__ import annotations

import argparse
import json
import socket
import statistics
from contextlib import suppress
from datetime import UTC, datetime

from adaptive_vlm.timing import perf_counter_ns


def serve(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen()
        print(f"TCP echo ready on {host}:{port}", flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while data := connection.recv(1):
                    connection.sendall(data)


def probe(host: str, port: int, count: int, warmup: int) -> dict[str, object]:
    samples: list[float] = []
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(10)
        for index in range(count + warmup):
            started = perf_counter_ns()
            connection.sendall(b"x")
            if connection.recv(1) != b"x":
                raise RuntimeError("echo response mismatch")
            elapsed_ms = (perf_counter_ns() - started) / 1_000_000
            if index >= warmup:
                samples.append(elapsed_ms)
    ordered = sorted(samples)
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "host": host,
        "port": port,
        "count": count,
        "warmup": warmup,
        "rtt_ms": {
            "min": ordered[0],
            "median": statistics.median(ordered),
            "mean": statistics.mean(ordered),
            "p95": ordered[max(0, int(0.95 * (count - 1)))],
            "max": ordered[-1],
            "samples": samples,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    server = subparsers.add_parser("server")
    server.add_argument("--bind-host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=5051)
    client = subparsers.add_parser("client")
    client.add_argument("--host", required=True)
    client.add_argument("--port", type=int, default=5051)
    client.add_argument("--count", type=int, default=30)
    client.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.command == "server":
        with suppress(KeyboardInterrupt):
            serve(args.bind_host, args.port)
    else:
        if args.count < 1 or args.warmup < 0:
            parser.error("count must be positive and warmup nonnegative")
        print(
            json.dumps(
                probe(args.host, args.port, args.count, args.warmup), indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
