# Chunk 2 model and GPU feasibility report

## Runtime inventory

| Item | Observed value |
| --- | --- |
| Edge CPU | Intel Core i7-11800H, 8 cores / 16 logical CPUs |
| WSL memory | 7.637 GiB plus 2 GiB swap |
| Edge GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU memory | 4,096 MiB reported; 4,095.5 MiB visible to PyTorch |
| Compute capability | 8.6 |
| Windows driver | 610.62 |
| WSL | 2.5.10.0, kernel 6.6.87.2 |
| Python | 3.11.17 |
| PyTorch | 2.7.1+cu128 |
| Transformers | 4.52.4 |
| BF16 | Supported; BF16 GPU matrix operation passed |

`/dev/dxg`, Linux `nvidia-smi`, Windows `nvidia-smi.exe`, and PyTorch all see
the same GPU when commands run outside the development sandbox. The sandbox
hides `/dev/dxg`; this is not a host/WSL fault.

## Artifact inventory

| Component | Parameters | BF16 weight MiB |
| --- | ---: | ---: |
| Vision encoder | 412,987,248 | 787.711 |
| Projector | 35,661,824 | 68.020 |
| Language model | 1,418,270,720 | 2,705.137 |
| Region model | 60,317,312 | 115.046 |
| **Total** | **1,927,237,104** | **3,675.913** |

The safetensors file contains 592 BF16 tensors and is 3,854,538,968 bytes.
The complete code, weight, tokenizer, and dependency fingerprints are in
`configs/model/moondream2_artifact.yaml`.

## Controlled inference results

### GPU attempt

Two independent BF16 attempts failed during state-dict placement at the same
Transformers operation. Both returned `CUDA driver error: device not ready`.
CUDA health checks passed before and after, including an actual BF16 matrix
operation. No generation occurred and the code did not fall back to CPU.

Memory explains the result: parameter bytes plus the configured full KV cache
are about 4,059.9 MiB, leaving approximately 35.6 MiB before CUDA context,
buffers, activations, and vision intermediates. The full model is therefore
classified as infeasible on this GPU in unmodified BF16.

### CPU attempt

The identical artifact and precision ran successfully on the edge CPU. The
final offline probe loaded the model once and evaluated three synthetic PPM
fixtures, with one warm-up followed by two measured repetitions per fixture:

| Metric | Result |
| --- | ---: |
| Model load | 81,734.4 ms |
| Measured latency range | 52,584.6–57,671.5 ms |
| Median measured latency | 53,023.7 ms |
| Generated tokens per response | 8 |
| Per-fixture repeated output equality | Exact for all three |
| RSS after inference | 4,633.2 MiB |

The quadrant, bar, and gradient fixtures each produced the same text on both
measured repetitions. Their respective outputs began:

```text
A vibrant abstract image is divided into four quadrants
A gradient strip transitions smoothly from white on the left
A gradient of black to gray colors transitions
```

The token IDs in the probe are obtained by re-encoding the returned text; exact
generation token capture is part of the decomposed adapter in Chunk 3. The
fixtures are synthetic, so this is a repeatability and feasibility smoke test,
not an output-quality claim. These are not final benchmark numbers: there were
only two measured repetitions per fixture and generation was capped at eight
tokens. The substantial difference from the earlier unwarmed single-fixture
probe also shows why the full benchmark campaign needs more repetitions and
controlled host conditions.

## Outcome

- GPU passthrough: **verified**.
- Moondream artifact/revision: **pinned and hashed**.
- Monolithic deterministic inference: **verified on edge CPU**.
- Unmodified all-edge GPU execution: **infeasible at load**.
- Unmodified all-edge device execution: **feasible on CPU, slow**.
- Vision/projector split: **source-level boundary found; equivalence remains a
  Chunk 3 gate**.
- Cloud hardware: **still pending**.
