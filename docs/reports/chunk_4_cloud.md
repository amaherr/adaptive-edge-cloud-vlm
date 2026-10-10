# Chunk 4 transport and all-cloud report

## Result

The localhost all-cloud completion gate passed with the pinned Moondream2
artifact, CPU BF16, greedy decoding, and the configured 32-token limit. The
client sent the original 167-byte PPM fixture and prompt through the real TCP
protocol. The server ran `run_monolithic()` and returned exactly the same 32
native token IDs and decoded text as a local monolithic reference.

The accepted contract fingerprint was
`e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`.
It covers the pinned artifact, revision, precision, preprocessing backend, and
generation settings rather than identifying weights alone.

## Real-model localhost measurement

These are single-fixture feasibility measurements, not benchmark statistics.

| Measurement | Value |
| --- | ---: |
| Client-observed E2E | 38,958.34 ms |
| Server total | 38,956.65 ms |
| Server model E2E | 38,956.28 ms |
| Image encode and prefill | 25,397.24 ms |
| Language generation | 13,530.23 ms |
| Raw image payload | 167 bytes |
| Complete request frame | 504 bytes |
| Complete response frame | 1,344 bytes |
| Server RSS before request | 4,570.89 MiB |
| Server RSS after request | 4,775.29 MiB |

The approximately two-millisecond client/server residual is consistent with a
loopback transfer and framing overhead. It must not be interpreted as a
physical-network result.

## Stability and rejection evidence

The fake adapter completed 100 sequential requests through real localhost TCP
with identical local/remote output, zero failed requests, zero rejected valid
requests, and no deadlock. Process RSS increased from 21.125 to 21.5 MiB, with
0.375 MiB peak growth against the explicit 16 MiB test limit.

Unit and lifecycle tests additionally cover one-byte fragmented reads,
checksums, header/payload bounds, timeout propagation, mid-frame disconnects,
protocol-version mismatch, unsafe image suffixes, unexpected schemas, contract
fingerprint mismatch, structured model errors, readiness, and graceful
shutdown.

## Limitations

- Physical edge-to-cloud execution is not yet possible because the cloud host
  inventory and address remain pending.
- The server is deliberately sequential for the baseline and does not measure
  concurrent throughput.
- The protocol has no TLS, authentication, or authorization and is limited to
  trusted networks or a protected tunnel.
- The retained JSON artifacts under `artifacts/chunk4/` are ignored working
  data; this report records the completion evidence in version control.
