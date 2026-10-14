# QuixiCore-Metal vendored kernels

Apple Silicon serving kernels vendored from
[QuixiAI/QuixiCore-Metal](https://github.com/QuixiAI/QuixiCore-Metal)
(ThunderKittens-derived, Metal Shading Language), providing the Apple
counterpart of the gfx942 and sm80 kernel work in the sibling directories.

## Sync contract

This tree is **not** an unmodified copy of upstream. It started as one and has
been developed in place since, following the repository rule: develop serving
kernels here first, then port the finished, measured pieces upstream.

- **Upstream base:** commit `71a08cd4cbcdc622ce31b3fc91e1f505e144b516`
  (2026-07-27, branch `main`). Upstream `main` has had no `kernels/` or
  `include/` commits since that point (last checked 2026-10-14), so every
  difference between this tree and upstream originated in SlimServe.
- **Port-back vehicle:**
  [QuixiCore-Metal PR #3](https://github.com/QuixiAI/QuixiCore-Metal/pull/3),
  branch `dsv4-m1ultra-serving-port` on the `auroter/QuixiCore-Metal` fork.
  Kernel files on that branch are byte-identical twins of this tree at the
  SlimServe commit each port commit names. Upstream additionally carries the
  `.quixicore/kernels.yaml` manifest and the `bindings/pytorch_mps` binding,
  which the port PR keeps in step with `tk_launch.h` signature changes.
- **Layout mirrors upstream** (`kernels/`, `include/metal/`), so a recursive
  diff against a checkout of upstream or of the port branch is the drift
  check. Nothing in the repo automates this yet.
- SlimServe-specific host code stays outside this tree: the Torch/MPS binding
  in `csrc/quixicore/tm_metal/` and the Python surface in `vllm/quixicore/`.
  `kernels/serving_glue/` is the one exception: those kernels are SlimServe's
  serving-step fusions, but they compile into the same metallib and are
  ported upstream with everything else.

### Checking drift

```bash
git clone -q https://github.com/QuixiAI/QuixiCore-Metal.git /tmp/qcm
git -C /tmp/qcm fetch -q origin pull/3/head:port   # or whichever port branch is current
git -C /tmp/qcm checkout -q port
for d in kernels include; do
  diff -rq csrc/quixicore/metal/$d /tmp/qcm/$d | grep -v '^Only in /tmp/qcm'
done
```

Every line printed is unported SlimServe work, or, once upstream starts moving
again, an upstream change to pull in. `Only in /tmp/qcm` lines are the
deliberately un-vendored files listed below and are filtered out.

When a SlimServe change touches this directory, update the tables below and
refresh the port branch in the same campaign. The tables went stale once
already: the 2026-08-06 vendoring described an unmodified copy, and the first
in-place edit landed the next day.

## What this tree changed relative to upstream `71a08cd4`

Modified upstream files:

| Path | Changed by (SlimServe campaign) |
| --- | --- |
| `include/metal/ops/warp/register/tile/dequant.metal` | DSV4 M1 Ultra, Muse-Glimmer, Qwen3.8 |
| `include/metal/ops/warp/register/tile/dequant_tables.metal` | Qwen3.8 |
| `kernels/attention/mla/mla.metal` | DSpark TurboQuant restore, DSV4 M1 Ultra, exact fp8 scale fixes |
| `kernels/attention/paged_attn_v2/paged_attn_v2.metal` | DSV4 M1 Ultra, Muse-Glimmer verify, Qwen3.8 D=256 split-K, `kv_block_stride` |
| `kernels/common/tk_launch.h` | every campaign (host ABI for everything below; most recently Metal NVMe KV tier, Metal CT native quantized prefill GEMM, GLM-5.3-Flash Metal) |
| `kernels/linear_attention/gdn/gdn.metal` | Qwen3.8 GDN fusion |
| `kernels/norms/rms_norm/rms_norm.metal` | Qwen3.8 |
| `kernels/quantization/qgemm/qgemm.metal` | DSV4 M1 Ultra, Qwen3.8 |
| `kernels/quantization/qgemv/qgemv.metal` | DSpark TurboQuant restore through Qwen3.8 (multi-row MoE GEMVs, fp8ch/nvfp4, bf16 axis); GLM-5.3-Flash Metal (q8_0 NR / multi-batch / pair-SwiGLU GEMVs, dual GEMV, iq2_xxs texture-codebook MoE GEMV and slot grouping) |
| `kernels/quantization/turboquant/turboquant.metal` | DSpark TurboQuant restore, Qwen3.8 |
| `kernels/serving/indexer/indexer.metal` | DSV4 M1 Ultra, long-context guards |
| `kernels/serving/kv_cache/kv_cache.metal` | Qwen3.8 stride-aware paged kernel; Metal NVMe KV tier (`kv_cache_scatter` takes a physical `cache_block_stride` instead of `block_mult`) |

Added files inside upstream directories:

| Path | What |
| --- | --- |
| `kernels/quantization/qgemm/qgemm_planar.metal` | Metal CT native quantized prefill GEMM: `qgemm_nvfp4_planar` / `qgemm_fp8ch` reading the checkpoint's planar buffers |
| `kernels/sampling/sampling/flatten_sampled.metal` | Metal NVMe KV tier campaign: v2 sampler, packs per-request sampled tokens (variable counts) into one flat output |
| `kernels/sampling/sampling/logit_bias.metal` | Metal NVMe KV tier campaign: v2 sampler allowed-token / logit-bias / min-tokens stop masking in one dispatch |
| `kernels/serving/kv_cache/kv_cache_fp8_strided.metal` | FP8 KV directive: FP8 E4M3 paged-cache scatter with explicit 64-bit physical strides |
| `kernels/serving/kv_cache/mamba_align.metal` | Metal NVMe KV tier campaign: GPU-resident mamba align-mode plan and state copies for hybrid models |

Added directories (no upstream counterpart before PR #3):

| Path | What |
| --- | --- |
| `kernels/attention/glm5_indexer/` | GLM-5.3-Flash pooled indexer (pool logits, top-k expand) and paged row insert |
| `kernels/attention/mla_sparse_latent/` | GLM-5.3-Flash sparse NoPE-MLA decode over latent pages, per-head and simdgroup-MMA variants |
| `kernels/linear_attention/kda/` | KDA (Kimi Delta Attention) serving chain: prepare, recurrence, spec-verify recurrence |
| `kernels/moe/moe_mm_id/` | tiled MoE prefill GEMM (per-expert work queue, iq2_xxs simdgroup-MMA, q2_K down; GLM-5.3-Flash Metal raised the expert cap to 512) |
| `kernels/moe/moe_router/` | single-group MoE router: scores, bias, top-k, renormalize, scale in one dispatch (id -1, weight 0 for slots with no finite score) |
| `kernels/quantization/qgemm_mma/` | small-M weight-stationary MMA GEMM, q8_0 / q4_K x bf16, for the 9..32-row dense band |
| `kernels/quantization/qgemm_sm/` | simdgroup-matrix quantized GEMM for the tensor-ops verify path |
| `kernels/serving/dflash_conv/` | fused DFlash2 grouped dynamic convolution (drafter block conv) |
| `kernels/serving/dsv4_mhc/` | fused DeepSeek-V4 mHC pre/post (projection, gates, softmax, Sinkhorn); GLM-5.3-Flash Metal added the RMSNorm-fused finalize |
| `kernels/serving/dsv4_router/` | DeepSeek-V4 MoE router top-k (softplus, sqrt, bias, top-k in one dispatch) |
| `kernels/serving/kv_meta/` | per-step KV metadata for the Metal runner (block-table gather and slot mapping in one launch) |
| `kernels/serving/moe_finalize/` | weighted expert-row sum for the GGUF grouped path |
| `kernels/serving/qk_norm_rope_gate/` | fused Qwen3-Next attention prep (gated-q split, QK RMSNorm, partial RoPE) |
| `kernels/serving/rms_norm/` | weighted RMS norm for the decode path (w32 / strided variants; GLM-5.3-Flash Metal added the dual q_a/kv_a norm) |
| `kernels/serving/swiglu/` | fused SwiGLU mirroring the eager torch chains bit for bit |
| `kernels/serving_glue/` | Muse step glue, DFlash prepare, DFlash2 conv, GDN decode step, rejection sampling (the NVMe KV tier campaign added a reduction barrier to `rejection_sample.metal` and removed the superseded `muse_kv_store` from `muse_step.metal`) |

## What was and was not taken from upstream

Taken:

| Path | What |
| --- | --- |
| `kernels/**/*.metal` | the MSL kernel sources (79 upstream at the base commit, 104 here) |
| `include/metal/**` | the tile substrate `tk.metal` pulls in |
| `kernels/common/tk_launch.h` | host-side launch ABI, pure C++, no MLX, no Metal |
| `kernels/common/base_q_descriptor.h` | BaseQN packed-quant descriptor |

Deliberately not taken:

- The per-kernel `.cpp`/`.h` pairs beside each `.metal`. Those are MLX
  primitives (`#include "mlx/ops.h"`); this fork drives the kernels from
  PyTorch MPS through `tk_launch.h`, so that header is the only host-side file
  worth carrying. Kernels added here consequently ship upstream without MLX
  pairs.
- `include/quixicore/` (the BaseRT contract stubs) and `include/MetalSingle.hpp`.
- `bindings/python/`, `bindings/mlx/` (the MLX extension plus a vendored MLX
  source tree) and `bindings/pytorch_mps/`. The last one's `torch_kernels.mm`
  is the structural model for `csrc/quixicore/tm_metal/`, but it JIT-builds
  the metallib and the extension at import (~22 s cold on an M5 Max). A served
  fork compiles both ahead of time, so the binding is written here rather than
  copied.
- `tests/`, `perf/`, `docs/`, and `.quixicore/kernels.yaml`.

## Building the metallib

`cmake/metal.cmake` globs every `.metal` under `kernels/` into one
`quixicore_metal.metallib` at build time and re-runs when anything under
`include/metal/` changes. The equivalent command line:

```bash
cd csrc/quixicore/metal
xcrun metal -std=metal4.0 -O2 -I include/metal -I kernels/common \
  $(find kernels -name '*.metal') -o quixicore_metal.metallib
```

Upstream's own README builds with `-std=metal3.1`. This tree compiles under
both, and the port PR is checked against 3.1 so upstream's toolchain floor is
preserved.

This needs the Xcode Metal toolchain component, which is a separate download
from Xcode itself:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Two `-Wunused-variable` warnings come out of
`include/metal/ops/warp/shared/tile/maps.metal`. They are upstream's and are
left alone.
