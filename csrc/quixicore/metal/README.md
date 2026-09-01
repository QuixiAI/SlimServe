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
  `include/` commits since that point (last checked 2026-09-01), so every
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
| `kernels/common/tk_launch.h` | every campaign (host ABI for everything below) |
| `kernels/linear_attention/gdn/gdn.metal` | Qwen3.8 GDN fusion |
| `kernels/norms/rms_norm/rms_norm.metal` | Qwen3.8 |
| `kernels/quantization/qgemm/qgemm.metal` | DSV4 M1 Ultra, Qwen3.8 |
| `kernels/quantization/qgemv/qgemv.metal` | DSpark TurboQuant restore through Qwen3.8 (multi-row MoE GEMVs, fp8ch/nvfp4, bf16 axis) |
| `kernels/quantization/turboquant/turboquant.metal` | DSpark TurboQuant restore, Qwen3.8 |
| `kernels/serving/indexer/indexer.metal` | DSV4 M1 Ultra, long-context guards |
| `kernels/serving/kv_cache/kv_cache.metal` | Qwen3.8 stride-aware paged kernel |

Added directories (no upstream counterpart before PR #3):

| Path | What |
| --- | --- |
| `kernels/moe/moe_mm_id/` | tiled MoE prefill GEMM (per-expert work queue, iq2_xxs simdgroup-MMA, q2_K down) |
| `kernels/quantization/qgemm_sm/` | simdgroup-matrix quantized GEMM for the tensor-ops verify path |
| `kernels/serving/dflash_conv/` | fused DFlash2 grouped dynamic convolution (drafter block conv) |
| `kernels/serving/dsv4_mhc/` | fused DeepSeek-V4 mHC pre/post (projection, gates, softmax, Sinkhorn) |
| `kernels/serving/dsv4_router/` | DeepSeek-V4 MoE router top-k (softplus, sqrt, bias, top-k in one dispatch) |
| `kernels/serving/moe_finalize/` | weighted expert-row sum for the GGUF grouped path |
| `kernels/serving/qk_norm_rope_gate/` | fused Qwen3-Next attention prep (gated-q split, QK RMSNorm, partial RoPE) |
| `kernels/serving/rms_norm/` | weighted RMS norm for the decode path (w32 / strided variants) |
| `kernels/serving/swiglu/` | fused SwiGLU mirroring the eager torch chains bit for bit |
| `kernels/serving_glue/` | Muse step glue, DFlash prepare, DFlash2 conv, GDN decode step, rejection sampling |

## What was and was not taken from upstream

Taken:

| Path | What |
| --- | --- |
| `kernels/**/*.metal` | the MSL kernel sources (79 upstream at the base commit, 93 here) |
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
