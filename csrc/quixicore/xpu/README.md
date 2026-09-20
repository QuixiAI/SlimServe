# QuixiCore-XPU vendored SYCL tree

Intel GPU (SYCL / oneAPI / Level Zero) kernels, runtime, and dispatch layer
vendored from the local `~/QuixiCore/QuixiCore-XPU` tree, providing the Intel
counterpart of the gfx942 (`rocm/`, `tm_rocm/`), Ampere (`moe/`, `quant/`,
`tm_cuda/`), and Metal (`metal/`, `tm_metal/`) kernel work in this directory.

- Vendored from commit: `47d0055efb4676223c4e576df7c12b063ae1f068` (2026-08-15)
- Layout mirrors upstream so files diff cleanly and resync is a copy of the
  same paths:
  - `include/quixicore/xpu/` public ABI (`ops.hpp`, `runtime.hpp`, `graph.hpp`,
    `variants.hpp`, `ipc.hpp`, `backend.hpp`, contract headers)
  - `src/backend.cpp`, `src/runtime/` (queue/device enumeration, current-queue
    SYCL command-graph capture, Level Zero IPC), `src/dispatch/` (per-family
    `Variant` routing: native SYCL vs. oneDNN vendor path)
  - `kernels/<family>/<op>/variants/xpu_sycl/*.sycl.cpp` native SYCL kernels,
    `.../xpu_onednn/*.onednn.cpp` vendor variants, `kernels/common/` shared
    helpers (`vec_map.hpp` 16-byte vectorised elementwise, `xmx_tile.hpp`
    joint_matrix/DPAS tiles, `quant_codecs.hpp` fp8/fp4/e8m0 decoders)
- The whole tree is compiled by `cmake/xpu.cmake` into ONE shared library
  (`libquixicore_xpu_ops.so`), and the pybind11 binding in `../tm_xpu/`
  (`qc_xpu_ext.sycl`, adapted from upstream `bindings/pytorch/tk_xpu_ext.sycl`)
  is built as `vllm._quixicore_C`. Python access is `vllm/quixicore/ops.py`,
  the same layer the CUDA/ROCm/Metal builds use.

## SlimServe kernels developed here (ported back to QuixiCore-XPU byte-identical)

`kernels/quantization/gguf_gemv/variants/xpu_sycl/gguf_routed.sycl.cpp` — the
DeepSeek-V4 GGUF path: routed/batched GEMV and dequantize for ggml types
Q8_0 (8), Q2_K (10), IQ2_XXS (16) on the on-disk block layout, exposed as
`ops::gguf_routed_gemv` / `ops::gguf_dequantize` (ops.hpp, dispatch/
quantization.cpp) and bound in `../tm_xpu/qc_xpu_slimserve.sycl` as
`ggml_mul_mat_vec_a8`, `ggml_moe_a8_vec`, `ggml_dequantize[_into]`. Verified
bit-exact against gguf-py `dequantize` for all three formats and at bf16
rounding for the GEMV/MoE outputs (2026-08-18). Correctness-first: lanes
stride 32-wide K units of the interleaved layout; the repack + tile GEMM is
the tracked next step (measured 88 GB/s on the 1-token x 6-expert IQ2_XXS
gate/up shape vs the ~450 GB/s class ceiling).

`../tm_xpu/qc_xpu_slimserve.sycl` also provides `get_xpu_view_from_cpu_tensor`
(UVA view over pinned host memory for the V1 runner's input buffers).

## Why the whole tree, not a curated subset

Upstream ships one op library and one binding whose dispatch table references
every family; vendoring a subset would need a forked `ops.hpp` / dispatch tree
that stops diffing against upstream. Like the Metal metallib (79 shaders,
GLOB'd), the SYCL library is one artifact. Serving-path relevance for
DeepSeek-V4-Flash on B70 lives in: `quantization/gguf_gemv` (IQ2_XXS/IQ2_XS,
Q2_K & friends, `gguf_iq_tables.hpp`), `moe/{moe_route,moe_permute,
grouped_qgemm}`, `serving/{mqa_logits,kv_cache_paged}` (DSA indexer,
paged KV), `ssm/dsv4_hc` (mHC post), `quantization/turboquant` (draft KV
codec), `norms/rms_norm`, `activations/glu{,_quant}`, `sampling/`.

## Build requirements (learned upstream, see bindings/pytorch/README.md there)

1. The op library must be a *shared* object built by `icpx -fsycl`; a static
   archive does not register its SYCL device images with the runtime
   ProgramManager and kernel submits segfault.
2. The final extension link must be a `-fsycl` device link; `cmake/xpu.cmake`
   makes icpx the project C++ compiler and puts `-fsycl` on the link line.
3. `-fp-model=precise -foffload-fp32-prec-div -foffload-fp32-prec-sqrt` are
   part of the codec correctness contract (bit-exact TurboQuant); do not drop.
4. ONE SYCL runtime per process. torch+xpu wheels bundle `libsycl.so.9` /
   `libur_*` (pip `intel-sycl-rt`) in `<venv>/lib`; `cmake/xpu.cmake` pins
   both our libraries to that directory with DT_RPATH so they can never pick
   up `/opt/intel/oneapi`'s copy. Corollary: `source setvars.sh` is a
   BUILD-time step only. With it in the serving environment torch.xpu itself
   segfaults on the first allocation (2026.1.0 bundled runtime vs 2026.1.1
   system UR on LD_LIBRARY_PATH; reproduced 2026-08-18 on QuadB70 with plain
   `torch.randn(4, device="xpu")`).

```bash
# build (setvars only here)
( source /opt/intel/oneapi/setvars.sh && \
  VLLM_TARGET_DEVICE=xpu pip install -e . --no-build-isolation )
# run: clean environment, no setvars
python -c "import vllm._quixicore_C as qc; print(qc.device_count())"   # 4
```

Bring-up smoke on QuadB70 (2026-08-18, torch 2.15.0.dev20260815+xpu, icpx
2026.1.1, oneDNN 2026.0.2 vendor variants on): `silu`, `gelu`, `rms_norm`,
`dense_gemm` on bf16/f16 match torch.xpu references at storage-dtype epsilon;
`XPUGraph` constructs; `device_count() == 4`.

Optional AoT for Battlemage: `VLLM_XPU_SYCL_TARGETS=bmg` in the environment at
configure time (default JIT).

Keep local edits out of the vendored files where possible; SlimServe-specific
glue goes in `../tm_xpu/` or `vllm/quixicore/`. Kernels developed here for the
B70 serving path are ported back to `~/QuixiCore/QuixiCore-XPU` byte-identical.
