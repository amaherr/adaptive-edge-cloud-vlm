"""Conditional S3 object operations through the operator's AWS CLI credentials."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class MissingObject(RuntimeError):
    """An object is absent."""


class ObjectConflict(RuntimeError):
    """A conditional write or delete lost its precondition."""


@dataclass(frozen=True)
class Object:
    data: bytes
    etag: str


class Store(Protocol):
    def get(self, key: str) -> Object: ...

    def put(self, key: str, data: bytes, *, etag: str | None = None) -> str: ...

    def delete(self, key: str, *, etag: str) -> None: ...


class S3Store:
    """No unconditional writes/deletes; IAM should restrict the campaign prefix."""

    def __init__(self, bucket: str, *, profile: str | None = None) -> None:
        if not bucket or "/" in bucket:
            raise ValueError("provide a bucket name, not an S3 URL")
        self.bucket = bucket
        self.profile = profile

    def _call(self, *args: str) -> dict[str, str]:
        command = ["aws"]
        if self.profile:
            command += ["--profile", self.profile]
        command += ["--no-cli-pager", "--output", "json", "s3api", *args]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
        if result.returncode:
            # Never expose CLI stderr (credentials/endpoint details may appear).
            if any(code in result.stderr for code in ("NoSuchKey", "(404)")):
                raise MissingObject("S3 object is absent")
            if any(
                code in result.stderr
                for code in (
                    "PreconditionFailed",
                    "ConditionalRequestConflict",
                    "(412)",
                    "(409)",
                )
            ):
                raise ObjectConflict("S3 conditional operation conflicted")
            raise RuntimeError("S3 operation failed; check credentials and permissions")
        value: dict[str, str] = json.loads(result.stdout or "{}")
        return value

    def get(self, key: str) -> Object:
        with tempfile.TemporaryDirectory(prefix="vlm-s3-") as directory:
            path = Path(directory) / "object"
            metadata = self._call(
                "get-object", "--bucket", self.bucket, "--key", key, str(path)
            )
            return Object(path.read_bytes(), metadata["ETag"])

    def put(self, key: str, data: bytes, *, etag: str | None = None) -> str:
        with tempfile.TemporaryDirectory(prefix="vlm-s3-") as directory:
            path = Path(directory) / "object"
            path.write_bytes(data)
            condition = (
                ["--if-none-match", "*"] if etag is None else ["--if-match", etag]
            )
            metadata = self._call(
                "put-object",
                "--bucket",
                self.bucket,
                "--key",
                key,
                "--body",
                str(path),
                "--server-side-encryption",
                "AES256",
                *condition,
            )
            return metadata["ETag"]

    def delete(self, key: str, *, etag: str) -> None:
        self._call(
            "delete-object", "--bucket", self.bucket, "--key", key, "--if-match", etag
        )
