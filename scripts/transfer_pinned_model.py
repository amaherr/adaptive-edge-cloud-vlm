"""Resume a verified model-cache copy over the protected edge/cloud LAN port.

Run ``serve`` on the cloud and ``send`` on the edge. Each request carries at
most 16 MiB, so a dropped connection can resume at the last complete chunk.
The cloud publishes the blob only after its size and SHA-256 are verified.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

MODEL_SHA256 = "70a7d94c0c8349eb58ed2d9e636ef2d0916960f321ecabeac6354b8ba3d7403f"
MODEL_SIZE = 3_854_538_968
CHUNK_SIZE = 16 * 1024 * 1024
DEFAULT_BLOB_DIR = (
    Path(__file__).resolve().parents[1]
    / ".model-cache/hub/models--vikhyatk--moondream2/blobs"
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _serve(part: Path, expected_size: int, expected_sha256: str, port: int) -> None:
    blob = part.with_name(expected_sha256)
    part.parent.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/status":
                self._reply(404, {"error": "unknown path"})
                return
            offset = (
                expected_size
                if blob.exists()
                else part.stat().st_size if part.exists() else 0
            )
            self._reply(
                200,
                {
                    "offset": offset,
                    "verified": blob.exists(),
                },
            )

        def do_PUT(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path != "/chunk":
                self._reply(404, {"error": "unknown path"})
                return
            try:
                offset = int(parse_qs(parsed.query)["offset"][0])
                length = int(self.headers["Content-Length"])
            except (KeyError, IndexError, ValueError):
                self._reply(400, {"error": "missing offset or length"})
                return
            current = part.stat().st_size if part.exists() else 0
            if blob.exists() or offset != current:
                self._reply(409, {"offset": current, "error": "offset changed"})
                return
            if not 0 < length <= CHUNK_SIZE or current + length > expected_size:
                self._reply(400, {"error": "invalid chunk length"})
                return
            data = self.rfile.read(length)
            if len(data) != length:
                self._reply(400, {"error": "chunk ended early"})
                return
            if hashlib.sha256(data).hexdigest() != self.headers.get("X-Chunk-SHA256"):
                self._reply(422, {"error": "chunk checksum mismatch"})
                return
            with part.open("ab") as output:
                output.write(data)
                output.flush()
            self._reply(200, {"offset": current + length})
            print(f"received {current + length:,}/{expected_size:,} bytes", flush=True)

        def do_POST(self) -> None:
            if self.path != "/finalize":
                self._reply(404, {"error": "unknown path"})
                return
            if blob.exists():
                self._reply(200, {"sha256": expected_sha256, "verified": True})
                self.server.verified = True  # type: ignore[attr-defined]
                return
            if not part.exists() or part.stat().st_size != expected_size:
                self._reply(409, {"error": "file is incomplete"})
                return
            actual = _digest(part)
            if actual != expected_sha256:
                self._reply(422, {"error": "final SHA-256 mismatch", "actual": actual})
                return
            part.replace(blob)
            self._reply(200, {"sha256": actual, "verified": True})
            self.server.verified = True  # type: ignore[attr-defined]
            print(f"verified {actual}", flush=True)

    server = HTTPServer(("0.0.0.0", port), Handler)
    server.timeout = 1
    server.verified = False  # type: ignore[attr-defined]
    print(f"resumable receiver ready on TCP {port}", flush=True)
    starting_offset = part.stat().st_size if part.exists() else 0
    print(f"starting offset: {starting_offset:,}", flush=True)
    try:
        while not server.verified:  # type: ignore[attr-defined]
            server.handle_request()
    finally:
        server.server_close()


def _request(
    host: str, port: int, method: str, path: str, body: bytes = b""
) -> dict[str, object]:
    connection = http.client.HTTPConnection(host, port, timeout=120)
    headers = {"Content-Length": str(len(body))}
    if body:
        headers["X-Chunk-SHA256"] = hashlib.sha256(body).hexdigest()
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"cloud returned HTTP {response.status}: {result}")
        if not isinstance(result, dict):
            raise RuntimeError("cloud returned an invalid response")
        return result
    finally:
        connection.close()


def _send(source: Path, host: str, port: int, expected_size: int) -> None:
    if source.stat().st_size != expected_size:
        raise ValueError(f"model file must be {expected_size:,} bytes: {source}")
    failures = 0
    while True:
        try:
            status = _request(host, port, "GET", "/status")
            if status.get("verified") is True:
                print("cloud model already verified", flush=True)
                return
            offset = int(status["offset"])
            if offset == expected_size:
                result = _request(host, port, "POST", "/finalize")
                print(f"cloud verified {result['sha256']}", flush=True)
                return
            if not 0 <= offset < expected_size:
                raise RuntimeError(f"invalid cloud offset: {offset}")
            with source.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read(min(CHUNK_SIZE, expected_size - offset))
            result = _request(host, port, "PUT", f"/chunk?offset={offset}", chunk)
            print(f"sent {result['offset']:,}/{expected_size:,} bytes", flush=True)
            failures = 0
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            failures += 1
            if failures >= 10:
                raise RuntimeError("upload failed after 10 retries") from error
            print(f"retrying after {error}", flush=True)
            time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve")
    server.add_argument(
        "--part", type=Path, default=DEFAULT_BLOB_DIR / f"{MODEL_SHA256}.incomplete"
    )
    server.add_argument("--port", type=int, default=5050)
    server.add_argument("--size", type=int, default=MODEL_SIZE)
    server.add_argument("--sha256", default=MODEL_SHA256)
    client = commands.add_parser("send")
    client.add_argument("--file", type=Path, default=DEFAULT_BLOB_DIR / MODEL_SHA256)
    client.add_argument("--host", required=True)
    client.add_argument("--port", type=int, default=5050)
    client.add_argument("--size", type=int, default=MODEL_SIZE)
    args = parser.parse_args()
    if args.command == "serve":
        _serve(args.part, args.size, args.sha256, args.port)
    else:
        _send(args.file, args.host, args.port, args.size)


if __name__ == "__main__":
    main()
