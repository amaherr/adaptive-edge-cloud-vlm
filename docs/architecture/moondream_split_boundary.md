# Moondream2 split boundary

The baseline splits after the vision projector and before the language-model
image prefill. The adapter calls the pinned model's `_run_vision_encoder()` and
validates its output before any transfer.

```text
PIL RGB image
    │
    ▼
overlapping crop preparation
    │
    ▼
vision encoder (`_vis_enc`)
    │ global and local features
    ▼
crop reconstruction + projector (`_vis_proj`)
    │
    ├── split tensor: [729, 2048], BF16
    │                  729 visual tokens
    │                  2,985,984 bytes (2.85 MiB)
    ▼
optional device/network transfer
    │
    ▼
prepend BOS embedding → [1, 730, 2048]
    │
    ▼
language-model image prefill → prompt prefill → greedy decoding
```

The public `encode_image()` API is deliberately not used as the boundary
because it continues through language-model prefill and returns KV caches.

The `ProjectedVisionFeatures` contract rejects an unexpected shape and records
the payload's dtype, source device, target device, and byte count. No
quantization, compression, or tensor transformation is applied.
