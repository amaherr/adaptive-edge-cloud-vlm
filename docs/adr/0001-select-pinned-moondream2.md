# ADR 0001: Select pinned Moondream2 for the first baseline

- Status: accepted
- Date: 2026-10-10

## Context

The first baseline needs one pretrained VLM that can run monolithically and
exposes a boundary after the vision encoder and projector. The edge runtime is
WSL 2 on an RTX 3050 Laptop GPU with 4,096 MiB VRAM and 7.637 GiB assigned RAM.

## Decision

Use `vikhyatk/moondream2` release `2025-06-21`, resolved to immutable commit:

```text
9a7d4024050840e001defacec2b00727e89149e6
```

The BF16 safetensors artifact is pinned by SHA-256:

```text
70a7d94c0c8349eb58ed2d9e636ef2d0916960f321ecabeac6354b8ba3d7403f
```

Use the Pillow preprocessing fallback and set the model's optional `variant`
to `null`. This prevents variant/LoRA downloads and keeps preprocessing
identical when the environment is reproduced. Dependency versions are frozen
in `requirements/model-cu128.lock`.

Chunk 3 proved exact CPU monolithic-versus-decomposed equivalence, so the model
configuration is now marked `approved`.

## Evidence

- License: Apache-2.0.
- Parameters: 1,927,237,104, all BF16.
- Vision encoder: 412,987,248 parameters.
- Projector: 35,661,824 parameters.
- Language model: 1,418,270,720 parameters.
- Region model: 60,317,312 parameters.
- CPU monolithic inference completed twice with identical tokens.
- Source exposes `_vis_enc`, `_vis_proj`, and `_run_vision_encoder`; the last
  returns the projected 729-by-2048 visual representation.
- Vision encoder plus projector weights occupy approximately 855.7 MiB in
  BF16, making the intended edge portion plausible on the 4 GiB GPU.

The public `encode_image()` method is not the split boundary: after producing
the projected visual embedding it also runs text-model prefill and returns KV
caches. The adapter therefore captures the lower-level tensor returned by
`_run_vision_encoder()` and resumes text prefill from that tensor.

Chunk 3 verified that this tensor has shape `729 × 2048`, BF16 dtype, and an
uncompressed size of 2,985,984 bytes. CPU monolithic and decomposed executions
produced a bit-identical tensor and the same 32 native output token IDs.

## Edge feasibility outcome

The complete model does not load reliably on the RTX 3050. Two clean-process
attempts failed at the same state-dict loading operation with:

```text
RuntimeError: CUDA driver error: device not ready
```

CUDA remained healthy after each process ended. The BF16 parameters alone are
about 3,675.9 MiB, and a full 2,048-token KV cache is about 384 MiB. Together
they consume about 4,059.9 MiB before CUDA context, model buffers, activations,
and vision intermediates, leaving no workable margin in 4,095.5 MiB usable
VRAM. The repeated load-time device reset is therefore treated as GPU memory
infeasibility, not as a general CUDA compatibility failure.

The same unmodified BF16 model runs on the edge CPU. This keeps the all-edge
policy feasible without quantization, although it is expected to be slow. GPU
and CPU placement must be recorded explicitly in every experiment result.

## Remote-code review notes

The pinned Python sources were hashed and inspected before execution. The only
unexpected network-capable path is optional LoRA variant retrieval in
`lora.py`; it is unreachable when `variant` is `null`. The model constructor
also loads `moondream/starmie-v1` without specifying a revision. Its resolved
commit and tokenizer hash are therefore pinned separately in
`configs/model/moondream2_artifact.yaml`. The adapter verifies that file and
replaces the implicitly loaded tokenizer with the exact local pinned artifact.

## Consequences

- All-edge defaults to CPU BF16, preserving the exact monolithic token sequence.
- Vision and projection fit on the RTX 3050 with a measured 913.8 MiB peak CUDA
  allocation, but the mixed GPU-vision/CPU-text output diverged after a
  19-token common prefix. That placement remains experimental rather than the
  correctness reference.
- The cloud must have enough memory for the language model and KV cache; its
  hardware remains a required input.
- No quantization, feature compression, or alternate checkpoint is introduced.
