# Chunk 5 handoff

Chunk 5 is complete for the same-host gate. The real Moondream2 comparison
passed exact projected-tensor and 32-token equivalence for both synchronous
and prompt-first distributed execution. Both pipeline variants passed 100
full-size tensor requests over localhost TCP without failed requests, retained
pending state, or material RSS growth. Synchronous is the primary distributed
baseline; prompt-first is available with `--pipeline prompt_first`.

Use the commands in the [README](../../README.md#distributed-vision-and-language-inference)
to start `distributed-server`, submit a photo and prompt with
`distributed-infer`, and reproduce the two verification harnesses. The
[report](../reports/chunk_5_distributed.md) records the measurements and
limits. The localhost config is
`configs/experiments/chunk5_local_distributed.yaml`; the physical-host
`baseline_distributed.yaml` remains pending until the cloud machine is known.

The next entry point is Chunk 6: a unified benchmark runner and correctness
campaign across edge, cloud, and the fixed distributed policy. Do not treat
the single-run prompt-first latency difference as an overlap speedup.
