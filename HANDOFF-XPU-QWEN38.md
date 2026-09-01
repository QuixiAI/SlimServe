# SlimServe × Qwen3.8-27B-NVFP4 on QuadB70 — bring-up state (2026-09-01)

## Status: RESOLVED. SlimServe serves at parity with the fork.

c1 59.0 / c8 260.5 / c32 548.5 tok/s vs the fork control 56.2 / 252.3 / 549.9
(+5.0% / +3.3% / -0.3%), at 65536 / fp8 KV / TP4 to match the comparator.
Method, per-step measurements and rejected hypotheses:
`perf/optimization_status.md` (2026-09-01 entry).

The kernel rebuild described below was the first of four blockers; all are
fixed. Prod on :29734 has NOT been cut over yet -- remaining work is 262k
context, TurboQuant, the 56+58 wedge check, c64/mixed-c16 and a soak.

### Original blocker analysis (kept for the record)

Four SlimServe commits land the code side of the port (e25c1bfcb, c44e266ae,
143c04794, 07794c1be). The engine now gets all the way through model load with
the same kernel selection as prod, then dies at FP8 requantisation.

## The blocker

`vllm-xpu-kernels` (`/home/lazarus/vllm-xpu-kernels`, the `.so`s shipped in
`vllm_xpu_kernels/`) was built against **torch 2.11**. SlimServe's venv runs
**torch 2.15.0.dev20260815+xpu**. The `DispatchKey` enum shifted between those
releases, so the compiled constant for `torch::kXPU` now resolves to **HPU**:

    $ python -c "from torch._C import _dispatch_dump; \
        import vllm_xpu_kernels._C; print(_dispatch_dump('_C::rms_norm'))"
    debug: registered at /home/lazarus/vllm-xpu-kernels/csrc/torch_bindings.cpp:16
    HPU: registered at .../torch_bindings.cpp:16

Every op in the library is affected, checked across both namespaces:

| op | dispatch key under torch 2.15 |
|---|---|
| `_C::rms_norm` | HPU |
| `_C::static_scaled_fp8_quant` | HPU |
| `_xpu_C::nvfp4_gemm` | HPU |
| `_xpu_C::cutlass_grouped_gemm_interface` | HPU |

So the schemas exist, `hasattr` checks pass, the library loads without error —
and every call raises `NotImplementedError: not currently implemented for the
XPU device`. The observed failure is `_C::static_scaled_fp8_quant` during
`ModelOptFp8LinearMethod.process_weights_after_loading`, but that is just the
first call site reached.

This also means **drop-ins 56 and 61 do not transfer to SlimServe as-is**.
Both ride on `_xpu_C.cutlass_grouped_gemm_interface`, which is mis-keyed here.
The `LD_LIBRARY_PATH` trick in `run_qwen38_slimserve.sh` swaps the K32
`libgrouped_gemm_xe_2.so` underneath an op that never dispatches.

The script's comment that the torch-2.11 `.so` "loads clean under 2.15
(verified 2026-08-19)" is true and misleading: loading is not dispatching.

SlimServe's own QuixiCore ops are unaffected — `platforms/xpu_c_ops.py`
registers them from Python with a *string* key (`lib.impl(name, fn, "XPU")`),
resolved at runtime, so they land correctly.

## Ways out

1. **Rebuild `vllm-xpu-kernels` against torch 2.15** with oneAPI icpx
   (`/opt/intel/oneapi/compiler/2026.1`). Correct fix, keeps SlimServe's torch.
   Note `/home/lazarus/vllm-xpu-kernels` HEAD `471b7b6` already carries the K32
   NVFP4 change, so a rebuild picks up drop-in 61 natively and the
   `LD_LIBRARY_PATH` override can be dropped.
2. **Move SlimServe's venv to the fork's torch 2.13.0.dev20260603+xpu.** Avoids
   the SYCL build, but SlimServe's own `_quixicore_C`/`libquixicore_xpu_ops.so`
   were built 2026-08-18 against 2.15 and would need rebuilding instead — same
   problem, other direction.
3. **Port the needed kernels to QuixiCore** and drop the vllm-xpu-kernels
   dependency entirely. Largest job; `perf/xpu_kernel_survey.md` already ranks
   this work.

Option 1 is the smallest correct step.

## What is already done (verified, not assumed)

Most of the fork's XPU work was already in this tree before today. Confirmed
byte-identical or equivalent: NVFP4 DPAS route + `k%32` guard + rowloop
(`kernels/linear/nvfp4/xpu.py`), fused GemmaRMSNorm, graph replay-order
toggles, per-worker `ZE_AFFINITY_MASK`, `modelopt_mixed`, dense
`Qwen3_5ForConditionalGeneration`, hybrid GDN + mamba `align`.

Landed today:

- `e25c1bfcb` — `KVQuantMode.TURBOQUANT` + `turboquant_*` mapping (hard boot
  blocker at the target config); mamba device-pointer int64 fold; UVA
  `.contiguous()` for packed NVFP4; `FULL_DECODE_ONLY` exempted from the
  TP>1→PIECEWISE downgrade (`VLLM_XPU_FORCE_PIECEWISE_TP=1` restores it).
- `c44e266ae` — duplicate custom-op registration between QuixiCore and
  vllm-xpu-kernels, in both the `vllm::xpu_fp8_*mqa_logits` and `torch.ops._C`
  namespaces. Ordering mattered because model-registry inspection runs in a
  subprocess that inherits `PYTHONPATH` but not preloaded libraries.
- `143c04794` — `KernelConfig.{linear,moe}_backend` default to ROCm-only
  `"aiter"`; coerced to `"auto"` on XPU. GGUF profiles never hit that path,
  which is why `dsv4-xxs-b70-4` never needed it.

Verified reaching: correct kernel selection (`XPUNvFp4W4A16LinearKernel` +
`XPUW8A16FP8LinearKernel`, matching prod), `FULL_DECODE_ONLY` surviving at TP4,
weights loading, all 4 workers pinned to one card each.

## Also worth knowing

Two 27B TP4 engines do not fit on 4×32 GiB. Prod holds ~27.9 GiB/card, leaving
~4.5 GiB. Any SlimServe bring-up requires stopping `vllm-qwen38.service` first;
there is no side-by-side option on this box.
