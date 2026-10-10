# Chunk 2 handoff

## Status

Chunk 2 — GPU access and monolithic model feasibility — is complete in the
working tree. The changes remain uncommitted pending explicit approval.

The selected Moondream2 release is immutable and hashed. CUDA and BF16 work on
the RTX 3050 through WSL, but the complete unmodified BF16 model does not fit
reliably in its 4 GiB VRAM. The same artifact completed warmed, repeated
inference on the edge CPU for three correctness fixtures.

## Worktree state

- base commit: `26b5fe6` (`docs: document scaffold and chunk one handoff`);
- Chunk 2 source, tests, configurations, reports, and this handoff are
  uncommitted;
- downloaded model data and raw probe results remain in ignored
  `.model-cache/` and `artifacts/chunk2/` paths.

## Commands run

```bash
.venv/bin/uv pip check --python .venv-model/bin/python \
  --cache-dir /tmp/adaptive-vlm-uv-cache
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest --cov=adaptive_vlm --cov-report=term-missing
for config_path in configs/experiments/*.yaml; do
  .venv/bin/adaptive-vlm config validate "$config_path"
done
git diff --check

.venv-model/bin/python scripts/probe_moondream.py \
  --revision 9a7d4024050840e001defacec2b00727e89149e6 \
  --artifact-sha256 70a7d94c0c8349eb58ed2d9e636ef2d0916960f321ecabeac6354b8ba3d7403f \
  --cache-dir .model-cache \
  --offline \
  --device cpu \
  --max-new-tokens 8 \
  --warmups 1 \
  --repetitions 2 \
  --image tests/fixtures/moondream_smoke.ppm \
  --image tests/fixtures/moondream_bars.ppm \
  --image tests/fixtures/moondream_gradient.ppm \
  --output artifacts/chunk2/moondream_bf16_cpu_multifixture_probe.json
```

GPU diagnostics and the two failed full-model GPU probes were run outside the
restricted development sandbox because it hides WSL's `/dev/dxg` device.

## Verification results

- Ruff: passed;
- strict mypy: passed for 15 source files;
- pytest: 38 passed;
- coverage: 81.95%, above the enforced 80% threshold;
- all experiment configurations: valid;
- pinned model environment: 45 installed packages compatible;
- GPU passthrough, CUDA compute, and BF16 matrix operation: passed;
- two full-model BF16 GPU loads: failed consistently during state-dict
  placement because the 4 GiB device has no usable memory margin;
- offline CPU probe: three fixtures, one warm-up and two measured repetitions
  each, with exact repeated outputs for every fixture;
- measured CPU latency: 52,584.6–57,671.5 ms, median 53,023.7 ms;
- final CPU process RSS: 4,633.2 MiB;
- Git whitespace validation: passed before final documentation updates and must
  be rerun at the commit gate.

## Decisions made

- use `vikhyatk/moondream2` release `2025-06-21` at commit
  `9a7d4024050840e001defacec2b00727e89149e6`;
- pin the BF16 safetensors file by SHA-256 and separately fingerprint remote
  model code, tokenizer, and the model dependency lock;
- use the Pillow preprocessing path and no optional LoRA variant;
- keep the model status `candidate` until Chunk 3 proves decomposition
  equivalence;
- classify complete BF16 GPU execution as infeasible on this edge device;
- retain CPU BF16 as the unmodified all-edge feasibility path, while treating
  its roughly 53-second request latency as unsuitable for low-latency use;
- require an explicit repository-local model cache for reproducible offline
  probes.

## Known limitations

- native generated token IDs are not exposed by the public query API; Chunk 2
  records repeatable text and re-encoded token IDs, while Chunk 3 must capture
  native IDs at the decomposed generation boundary;
- the synthetic fixtures test repeatability and plumbing, not VLM quality;
- raw results are local ignored artifacts; the tracked feasibility report
  preserves the measurements needed for the implementation decision;
- cloud hardware is still unknown and marked `pending`;
- vision/projector placement on the RTX 3050 is plausible from weight size but
  has not yet been measured;
- the remote model constructor implicitly requests the tokenizer repository;
  Chunk 3 should replace this with an explicitly verified local path if the
  adapter can do so without modifying model semantics.

## Next entry point

Begin Chunk 3 by implementing a Moondream adapter around the pinned artifact.
Expose the projected visual representation after `_vis_proj`, resume the text
path locally, and compare native generated token IDs against the monolithic
path before adding any network transport. Measure the vision encoder and
projector on the RTX 3050 independently; keep the CPU path available for the
complete all-edge policy.
