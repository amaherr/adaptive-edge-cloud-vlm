# Chunk 5 distributed inference report

## Result

The fixed distributed policy now supports synchronous tensor transfer and
prompt-first preparation. In the localhost real-model gate, both variants
generated the same 32 native token IDs and decoded text as the local decomposed
reference. The transmitted projected representation was BF16 `[729, 2048]`
(2,985,984 bytes). Local serialization and reconstruction preserved every bit,
and both repeated edge encodings produced the same bytes.

The vision-only adapter retains the encoder and projector; the text-only
adapter retains the language model, tokenizer, attention mask, and caches.
Role guards prevent the edge adapter from generating and the cloud adapter from
encoding images. The server accepts only projected tensors in this mode and
checks the complete model contract fingerprint before inference.

## Same-host measurements

One pinned Moondream2 process held the edge and cloud stages for this
correctness run, with CUDA vision and CPU language inference. These are
single-fixture observations, not benchmark estimates.

| Measurement | Synchronous | Prompt-first |
| --- | ---: | ---: |
| Client end to end | 57,868.1 ms | 41,805.5 ms |
| Edge vision end to end | 1,034.2 ms | 366.7 ms |
| Server image prefill | 41,123.9 ms | 26,794.3 ms |
| Server language generation | 15,667.0 ms | 14,619.3 ms |
| Tensor payload | 2,985,984 bytes | 2,985,984 bytes |
| Tensor request frame | 2,986,424 bytes | 2,986,425 bytes |
| Additional prompt request/response | 0 bytes | 310 / 295 bytes |
| Peak CUDA vision allocation | 914.8 MiB | 914.8 MiB |

Prompt-first finished 16.1 seconds sooner in this sequential observation. Its
prompt round trip was only 3.5 ms, while image prefill alone differed by 14.3
seconds; therefore the run does **not** establish a 16-second benefit from
overlap. The true overlap saving, if any, needs repeated paired measurements
under controlled resource load. Synchronous is the **primary distributed
baseline** for later comparisons because its behavior is verified and it sends
fewer bytes. Prompt-first remains an explicit experimental setting.

The model contract fingerprint was
`e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`.
The server completed both real requests with no rejection or failure.

## Transport stability and lifecycle

Each variant completed 100 sequential full-size BF16 tensor requests plus one
warm-up through real localhost TCP, with zero failed or rejected requests and
zero pending prompts left afterward. Peak process RSS growth after warm-up was
0.500 MiB for synchronous and 0.375 MiB for prompt-first, below the configured
64 MiB limit. Prompt-first tests cover duplicate IDs, prompt mismatch, expiry,
capacity rejection, and cleanup. A disconnected client leaves state only until
its configured expiry, after which the server removes it.

Artifacts are written under `artifacts/chunk5/` (ignored working data). The
real-model comparison is `distributed_variants_equivalence.json`; the current
stability runs are `distributed_synchronous_stability.json` and
`distributed_overlap_stability.json`.

## Limits and deployment notes

- The pinned loader verifies and constructs the complete checkpoint before
  pruning unused modules. Steady-state execution is split, but transient
  load-time RAM still needs room for the full model on each process.
- The comparison used localhost, not a physical edge-to-cloud link. The cloud
  host inventory is still pending, so latency and network conclusions cannot
  be generalized to a separate machine. RSS values in the equivalence harness
  are process-wide and cannot be interpreted as independent host measurements.
- CUDA vision and CPU text matched the *same mixed-device* local reference.
  Chunk 3 observed divergence from the CPU-only reference after token 19 for
  this fixture. Cross-device task quality remains to be evaluated on the full
  correctness set.
- The baseline server processes one connection at a time. Prompt-first moves
  tokenization ahead of vision but does not add concurrent language inference.
- The transport has no TLS or authentication; deploy only on trusted networks
  or behind a protected tunnel.
