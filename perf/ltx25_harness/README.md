# LTX-2.5 measurement harness

The scripts behind `perf/ltx25_metal_campaign.md` (the evidence ledger) and
the LTX-2.5 entries in `perf/optimization_status.md`. They are measurement
tools, not serving code; most need an Apple Silicon machine with MLX, the
official weights and, for the parity checks, a torch environment with the
upstream reference sources. Run model-loading scripts one at a time through
`gpu_run.py` (see `HANDOFF-LTX25.md`, "Ops constraints").

## Machine paths

`paths.py` reads them from the environment; nothing machine-specific is
written in the scripts:

| variable | what | default |
| --- | --- | --- |
| `LTX25_UPSTREAM` | the `packages/` directory of a Lightricks/LTX-2 checkout (`ltx-core`, `ltx-pipelines`) | `~/.local/scratch/ltx25/upstream/packages` |
| `LTX25_SCRATCH` | captured inputs, import stubs (`n11/oiio_stub.py`) and reference kernels (`n8/ltxkf`) | `~/.local/scratch/ltx25` |
| `LTX25_MODEL_ROOT` | the official LTX-2.5 weights as `slimserve ltx25-* --download-only` lays them out | `$SLIMSERVE_CACHE/ltx-2.5/official`, else `~/models/ltx-2.5/official` |
| `LTX25_BF16_PACK` | the dgrauet/ltx-2-mlx bf16 pack the early N2/N7 parity scripts compare against | `$SLIMSERVE_CACHE/ltx-2.5/dgrauet-bf16`, else `~/models/ltx-2.5/dgrauet-bf16` |

## Baseline runner

The baseline bars (ledger section 6) and the precision gates of sections 7
and 9 (`precision_gate.py`, `forward_gate3.py`, `forward_speed.py`,
`run_fp16_cli.py`, `baseline_run.sh`) run against
[dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) v0.15.12, cloned
and installed with uv at `~/.local/scratch/ltx25/ltx-2-mlx`. That project is
third-party code and is not vendored here. The gates used a local
modification of its DiT module (`LTXModel`, `model.py`), reproducible from
this description:

- `LTX_DIT_DTYPE` (`bf16` default, `fp16`, `fp32`) replaces the hard-coded
  `mx.bfloat16` casts of the model inputs (sigma, timestep, video/audio
  latents and text embeddings) at the model boundary;
- `LTX_DIT_STATS=1` appends `(block_idx, max|video_hidden|,
  max|audio_hidden|)` after every transformer block to a module-level
  `STATS` list, which `precision_gate.py --stats` reads.

Licences: LTX-2.5 weights and adapters are under Lightricks' LTX-2 Community
License Agreement (gated on Hugging Face); the prompt enhancer
`google/gemma-4-E2B-it` under Google's Gemma terms.
