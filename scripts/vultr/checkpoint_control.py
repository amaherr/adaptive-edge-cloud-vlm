"""Restore S3 checkpoints or release a stopped writer's persistent lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_vlm.ops.checkpoints import restore
from adaptive_vlm.ops.store import S3Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["restore", "release-stopped-writer"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--expected-lock-etag")
    parser.add_argument("--old-writer-confirmed-stopped", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    store = S3Store(settings["bucket"], profile=settings.get("profile"))
    if args.command == "restore":
        if args.destination is None:
            parser.error("restore requires --destination (must not already exist)")
        restore(store, settings["prefix"], settings["contract"], args.destination)
        print(
            f"Verified run restored: {args.destination}; use benchmark --resume"
        )
    else:
        if not args.old_writer_confirmed_stopped or not args.expected_lock_etag:
            parser.error(
                "confirm the old writer is stopped and supply its exact lock ETag"
            )
        store.delete(f"{settings['prefix']}/writer.json", etag=args.expected_lock_etag)
        print(
            "Stopped-writer lock conditionally released; retained data was not deleted"
        )


if __name__ == "__main__":
    main()
