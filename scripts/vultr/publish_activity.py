"""Publish only recent actual job activity; a timer alone must not keep GPUs alive."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from adaptive_vlm.ops.checkpoints import encode
from adaptive_vlm.ops.store import MissingObject, S3Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--prefix", default="research/adaptive-vlm/vultr-v1")
    parser.add_argument("--activity-id", default="job")
    parser.add_argument("--local-marker", required=True, type=Path)
    args = parser.parse_args()
    if args.prefix != "research/adaptive-vlm/vultr-v1":
        parser.error("only the dedicated Vultr campaign prefix is allowed")
    if "/" in args.activity_id or args.activity_id in {"", ".", ".."}:
        parser.error("invalid activity identity")
    age = time.time() - args.local_marker.stat().st_mtime
    if not 0 <= age <= 90:
        raise SystemExit("Job marker is stale/future; no keepalive was uploaded")
    store = S3Store(args.bucket, profile=args.profile)
    key = f"{args.prefix}/control/heartbeats/{args.activity_id}"
    try:
        etag = store.get(key).etag
    except MissingObject:
        etag = None
    store.put(
        key,
        encode({"local_activity_unix": args.local_marker.stat().st_mtime}),
        etag=etag,
    )
    print(json.dumps({"published": key, "local_activity_age_seconds": age}))


if __name__ == "__main__":
    main()
