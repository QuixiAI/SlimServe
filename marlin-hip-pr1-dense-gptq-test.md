# PR1 Dense gptq_marlin HIP Port — Handoff Document

## Branch
`marlin-hip/pr1-dense-gptq` — first of 5 stacked branches porting CUDA Marlin kernels to ROCm HIP for MI300X.

## System
- GPU: AMD Instinct MI300X (gfx942), 304 CUs
- ROCm 7.2, HIP Clang 22.0.0
- Python: `~/.venv/bin/python` (Python 3.12)
- PyTorch 2.9.1 (ROCm build)
- Build: `MAX_JOBS=128 PYTORCH_ROCM_ARCH="gfx942" ~/.venv/bin/python setup.py develop`

## What PR1 Does
Ports the CUDA Marlin quantization kernels to HIP for ROCm. Enables INT4 GPTQ/AWQ, INT8 GPTQ, and FP8 weight-only dense models on MI300X via emulated FP16 MFMA instructions.

## Bugs Fixed This Session (4 total, all committed-worthy)

### Bug 1: `apply_fp8_marlin_linear` called wrong op
- **File**: `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`
- **Fix**: Changed `ops.gptq_marlin_gemm(...)` → `ops.marlin_gemm(...)` (the `marlin_gemm` wrapper in `_custom_ops.py` already handles ROCm redirect to `gptq_marlin_gemm`)

### Bug 2: `b_group_offset` used wrong constant
- **File**: `csrc/quantization/gptq_marlin/gptq_marlin_hip_kernel.hip` line ~398
- **Fix**: Changed `kBTileWords` (global=256) → `Traits::kBTileWords` (128 for 4-bit, 256 for 8-bit)

### Bug 3: Missing workspace zeroing
- **File**: `csrc/quantization/gptq_marlin/gptq_marlin_hip.cu`
- **Fix**: Added `hipMemsetAsync` to zero workspace before kernel launch (prevents stale barrier values)

### Bug 4: Missing ops.def for ROCm
- **File**: `csrc/torch_bindings.cpp`
- **Fix**: Added `ops.def` for `awq_marlin_repack` and `marlin_int4_fp8_preprocess` in `#if defined(USE_ROCM)` block

## Routing Fix Applied This Session

### MarlinLinearKernel enabled on ROCm
Previously, INT4/INT8 dense models on ROCm could only use `ConchLinearKernel` (broken on this system — crashes with `hipErrorIllegalAddress` for INT4, hangs for INT8). Our `gptq_marlin_gemm` op worked at the kernel level but had no model-level routing.

**Files changed:**
- `vllm/model_executor/layers/quantization/kernels/mixed_precision/marlin.py` line 37-38:
  Changed `if not current_platform.is_cuda():` → `if not current_platform.is_cuda() and not current_platform.is_rocm():`
- `vllm/model_executor/layers/quantization/kernels/mixed_precision/__init__.py` lines 48-51:
  Added `MarlinLinearKernel` to `_POSSIBLE_KERNELS[PlatformEnum.ROCM]` (first in priority list, before ConchLinearKernel)

**Status**: MarlinLinearKernel IS selected (confirmed via test_kernel_select.py), model loads successfully, but **hangs during profile_run** (first inference with large batch).

## Current Blocker: Kernel Hang for Large N

### Symptom
`gptq_marlin_gemm` hangs (never returns) for certain large matrix sizes:
- `M=16384, N=4096, K=4096` → **PASS** (returns in ~1s)
- `M=16384, N=12288, K=4096` → **HANG** (never returns)

This matches the profile_run behavior: model loads fine (MarlinLinearKernel selected, weights repacked successfully), then hangs when the first forward pass hits a `gate_up_proj` layer with N=12288.

### Root Cause (suspected)
The kernel uses workspace-based barriers for K-reduction across thread blocks. The workspace has `sms` (304) entries. The grid is launched with `blocks = sms` and each block processes multiple N-tiles in a loop. For large N:
- N=12288 with thread_n_blocks=4 → n_tiles = 12288/64 = 192 N-tiles
- With K-parallel splitting, the barrier logic may deadlock

**Key files to investigate:**
- `csrc/quantization/gptq_marlin/gptq_marlin_hip_kernel.hip` — the main INT4/INT8 kernel. Look at:
  - The grid scheduling loop (how blocks iterate over tiles)
  - The barrier mechanism (`lock`, `unlock` functions using workspace)
  - `thread_n_blocks`, `thread_k_blocks` selection
  - The `par` (parallel K-reduction) logic
- `csrc/quantization/gptq_marlin/gptq_marlin_hip.cu` lines 484-491 — the `gptq_marlin_hip_gemm` dispatch call with `kMaxPar`
- `csrc/quantization/gptq_marlin/marlin_hip_common.h` — barrier implementations (fixed with `__hip_atomic_*` for GPU-scope ordering)

**Known fix pattern from PR2**: The dense FP8 MFMA kernel had the same barrier deadlock. Fix was:
- Changed `blocks = sms` to `blocks = n_tiles * par` (each block owns one N-tile)
- With `thread_k_blocks=1` (no K-splitting), this eliminates barriers entirely
- See memory notes about "W8A16 Dense Barrier Fix" for details

**The same fix likely applies here**: change the grid size in `gptq_marlin_hip_gemm` so each block processes exactly one (N-tile, par-slot) instead of using a grid-scheduling loop with barriers. Or force `thread_k_blocks` such that no K-splitting is needed.

### Test Script
`test_large_gemm.py` — reproduces the hang with direct op calls (no V1 engine needed):
```python
# This hangs:
ops.gptq_marlin_gemm(a, ..., size_m=16384, size_n=12288, size_k=4096, ...)
# This works:
ops.gptq_marlin_gemm(a, ..., size_m=16384, size_n=4096, size_k=4096, ...)
```

## What Works

### Direct kernel tests (test_pr1.py) — ALL PASS
- INT4 `gptq_marlin_gemm`: 3 shapes (M=1,4,16 × small N/K) — PASS
- INT8 `gptq_marlin_gemm`: 3 shapes — PASS
- FP8 `fp8_marlin_gemm` (fn format, fp8_is_fnuz=False): 3 shapes — PASS
- FP8 `marlin_gemm` wrapper (fnuz format on MI300X): 1 shape — PASS
- Config loading for all 3 models — PASS

### FP8 W8A16 model-level benchmark
- **Qwen/Qwen3-8B-FP8**: 11.6-11.7 tok/s (enforce_eager and CUDA graphs)
- Uses `Fp8Config` → `apply_fp8_marlin_linear` → `ops.marlin_gemm` → `gptq_marlin_gemm`
- This path works because FP8 uses `fp8_marlin_hip_gemm` (different kernel function), not `gptq_marlin_hip_gemm`

## What Doesn't Work

### INT4/INT8 model-level inference
- Model loads fine with MarlinLinearKernel
- Hangs during profile_run (dummy forward pass with M=16384)
- Root cause: `gptq_marlin_hip_gemm` kernel hangs for large N (see above)

## File Inventory (PR1 changes)

### New HIP kernel files
- `csrc/quantization/gptq_marlin/gptq_marlin_hip_kernel.hip` — Dense INT4/INT8/FP8 HIP kernel
- `csrc/quantization/gptq_marlin/gptq_marlin_hip.cu` — Entry point for all dense kernels
- `csrc/quantization/gptq_marlin/marlin_hip_common.h` — Shared HIP utilities
- `csrc/quantization/gptq_marlin/awq_marlin_repack_hip.cu` — AWQ repack
- `csrc/quantization/gptq_marlin/marlin_perm_hip.cu` / `.hip` — Perm tables
- `csrc/quantization/gptq_marlin/marlin_int4_fp8_preprocess_hip.cu` — FP8 preprocess
- `csrc/quantization/fp8/fp8_marlin_hip_kernel.hip` — FP8 emulated HIP kernel
- `csrc/quantization/fp8/fp8_marlin_hip.cu` — FP8 entry point (PR1 subset: no MFMA)
- `csrc/quantization/fp8/gptq_marlin_repack_hip.cu` — GPTQ repack for FP8

### Modified files
- `CMakeLists.txt` — Added HIP sources to `VLLM_EXT_SRC`
- `csrc/ops.h` — Added `fp8_marlin_gemm`, `gptq_marlin_gemm` declarations
- `csrc/torch_bindings.cpp` — Op registrations for ROCm
- `vllm/_custom_ops.py` — Python wrappers + fakes
- `vllm/model_executor/layers/quantization/fp8.py` — Enable Marlin on ROCm CDNA
- `vllm/model_executor/layers/quantization/gptq_marlin.py` — `is_marlin_compatible` on ROCm
- `vllm/model_executor/layers/quantization/utils/marlin_utils.py` — ROCm helpers
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py` — FP8 fnuz handling
- `vllm/model_executor/layers/quantization/inc.py` — AutoRound → gptq_marlin on ROCm
- `vllm/model_executor/layers/quantization/kernels/mixed_precision/marlin.py` — ROCm support
- `vllm/model_executor/layers/quantization/kernels/mixed_precision/__init__.py` — ROCm kernel list

## FP8 Format Notes
- MI300X native format: `float8_e4m3fnuz` (0x80 = NaN, single zero at 0x00)
- PyTorch default: `float8_e4m3fn` (0x80 = -0, 0x7F/0xFF = NaN)
- `gptq_marlin_hip.cu` hardcodes `fp8_is_fnuz = arch.rfind("gfx94", 0) == 0` → always True on MI300X
- `fp8_marlin_gemm` takes `fp8_is_fnuz` as explicit parameter
- In production: `process_weights_after_loading` converts fn→fnuz and doubles scales. Marlin kernel undoes this.

## Test Models
- INT4 W4A16: `Intel/Qwen3-8B-int4-AutoRound` (auto-round GPTQ, routed to gptq_marlin via `inc.py`)
- INT8 W8A16: `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8`
- FP8 W8A16: `Qwen/Qwen3-8B-FP8`

## Test Scripts on Disk
- `test_pr1.py` — Direct kernel-level tests (all pass)
- `test_kernel_select.py` — Verifies MarlinLinearKernel is selected
- `test_large_gemm.py` — Reproduces the hang with large N
- `test_int4_load.py` — Attempts model loading (shows hang in profile_run)
- `bench_pr1.py` — LLM-based benchmark (needs `if __name__ == '__main__'` for V1)
- `bench_pr1_serve.sh` — vllm serve + curl benchmark
- `bench_pr1_serve_cudagraph.sh` — Same with CUDA graphs (needs 360s timeout for graph capture)

## Next Steps
1. **Fix the kernel hang** for large N in `gptq_marlin_hip_kernel.hip` / `gptq_marlin_hip.cu`
   - Most likely: barrier deadlock in K-parallel reduction for large grids
   - Fix pattern: change grid sizing or eliminate K-splitting (see "Known fix pattern from PR2" above)
   - Test with `test_large_gemm.py` after fix
2. **Re-run model-level benchmarks** for all 3 models once hang is fixed
3. **Commit all changes** (4 bug fixes + routing fix + kernel fix)
