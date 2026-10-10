# Chunk 3 decomposition and all-edge report

## Correctness result

The pinned BF16 Moondream2 model was run through two CPU paths on the same edge
host:

1. the original `encode_image()` image-encoding and image-prefill path;
2. the decomposed path that captures projected visual tokens and explicitly
   resumes language-model prefill.

For the configured 32-token greedy response:

- projected tensors were bit-identical (`max_abs_difference = 0.0`);
- all 32 native output token IDs matched;
- decoded text matched exactly; and
- the split tensor was BF16 `[729, 2048]`, or 2,985,984 bytes.

A separate eight-token stability run executed the decomposed path twice. Both
runs reproduced the same tensor, token IDs, and text. Their final RSS values
differed by only 0.035 MiB.

## Component measurements

These are feasibility measurements on one synthetic fixture, not final
benchmark statistics.

| Component | CPU/CPU decomposed | GPU vision + CPU text |
| --- | ---: | ---: |
| Vision encoder | 10,356.7 ms | 551.4 ms |
| Projector | 437.5 ms | 23.1 ms |
| Other vision/preprocessing | 3.4 ms | 102.2 ms |
| Projected-feature transfer | 0.03 ms | 2.49 ms |
| Image prefill | 15,003.4 ms | 19,477.9 ms |
| 32-token generation | 14,272.9 ms | 13,829.3 ms |
| End to end | 40,076.7 ms | 34,031.7 ms |

The mixed-device run used 913.8 MiB peak allocated and 942 MiB peak reserved
GPU memory, confirming that vision plus projection fit comfortably where the
complete model did not.

## Cross-device observation

The GPU-vision/CPU-text response shared its first 19 token IDs with the CPU
correctness reference, then diverged. This is consistent with small numerical
differences between CPU and CUDA BF16 vision kernels reaching a later greedy
argmax decision. It does not invalidate the split boundary: CPU monolithic and
CPU decomposed execution were exactly equivalent.

Accordingly, the all-edge command defaults to CPU for both components. GPU
vision is retained as an explicit deployment experiment, and cross-hardware
quality/equivalence must be evaluated on the correctness dataset before it can
become a default policy.
