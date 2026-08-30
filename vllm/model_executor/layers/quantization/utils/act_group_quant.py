# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wide-tile fused SiLU-mul + fp8 group quantization for ROCm.

AITER's `act_mul_and_fp8_group_quant` launches one program per (row, 128-wide
group): for the Qwen3.8-27B MLP that is M x 136 programs each moving 640 bytes,
so launch and indexing overhead dominate a purely memory-bound op. Measured on
gfx942 it sustains ~740-800 GB/s, about 14% of the 5.3 TB/s peak.

This kernel does identical arithmetic but gives each program BLOCK_M rows x
GROUPS groups, which reaches ~3.75 TB/s (71% of peak) and is bit-identical to
the AITER kernel's output.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _silu(x):
    # Matches AITER's exp2-based SiLU exactly.
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _wide_act_mul_fp8_group_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    M,
    N,  # N is the output width (half of x's last dim)
    sx_m,
    sx_n,
    sy_m,
    sy_n,
    ss_m,
    ss_n,
    BLOCK_M: tl.constexpr,
    GROUPS: tl.constexpr,
    QB: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    BN: tl.constexpr = GROUPS * QB

    # int64 row offsets: M * 2N overflows int32 past ~2.1e9 elements. AITER
    # casts its strides for the same reason, as do the sibling kernels in
    # fp8_utils.py. Not reachable at today's max_num_batched_tokens, but this
    # helper is generic and the overflow would silently read out of bounds.
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = pid_g * BN + tl.arange(0, BN).to(tl.int64)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    off = rm[:, None] * sx_m + rn[None, :] * sx_n
    a = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + off + N * sx_n, mask=mask, other=0.0).to(tl.float32)
    y = _silu(a) * b

    y3 = tl.reshape(y, (BLOCK_M, GROUPS, QB))
    amax = tl.maximum(tl.max(tl.abs(y3), axis=-1), 1e-10)
    scale = amax.to(tl.float32) / DTYPE_MAX
    q = tl.clamp(y3 * (1.0 / scale.reshape(BLOCK_M, GROUPS, 1)), -DTYPE_MAX, DTYPE_MAX)

    tl.store(
        y_ptr + rm[:, None] * sy_m + rn[None, :] * sy_n,
        tl.reshape(q, (BLOCK_M, BN)).to(y_ptr.dtype.element_ty),
        mask=mask,
    )

    rg = pid_g * GROUPS + tl.arange(0, GROUPS)
    smask = (rm[:, None] < M) & (rg[None, :] < tl.cdiv(N, QB))
    tl.store(s_ptr + rm[:, None] * ss_m + rg[None, :] * ss_n, scale, mask=smask)


# Rows per program, by batch width. Measured on MI300X (gfx942) against the
# real Qwen3.8 MLP width, graph-timed, us per call:
#
#     M       aiter   BM2G8   BM4G8   BM8G8
#     1        2.38    2.23    2.26    3.50
#     8        2.53    2.33    2.34    3.52
#     32       4.75    2.41    2.39    3.71
#     64       7.70    3.07    3.04    4.00
#     128     14.27    4.37    4.14    4.41
#     512     57.71   15.37   13.70   14.45
#     2048   251.07   55.66   53.24   51.77
#
# A fixed BLOCK_M=8 is the WORST config below M=64 -- 3.5us against aiter's
# 2.4us, a ~40% regression on exactly the single-stream decode path -- because
# one program per 8 rows leaves a 304-CU card with a handful of workgroups.
# It only earns its keep from about M=1024. Selection is a pure function of M,
# resolved before launch, so it stays safe under CUDA graph capture; autotune
# would not, because it benchmarks at first call.
_GROUPS = 8


def _block_m(m: int) -> int:
    if m < 64:
        return 2
    if m < 1024:
        return 4
    return 8


def wide_act_mul_fp8_group_quant(
    x: torch.Tensor, group_size: int, dtype_quant: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N2 = x.shape
    # The registered fake impl asserts this; without the same check here the
    # eager path would silently pair the wrong columns and drop the last one,
    # so traced and eager execution would disagree on whether the input is
    # legal instead of both rejecting it.
    if N2 % 2:
        raise ValueError(f"activation width must be even, got {N2}")
    N = N2 // 2
    y = torch.empty((M, N), dtype=dtype_quant, device=x.device)
    s = torch.empty(
        (M, triton.cdiv(N, group_size)), dtype=torch.float32, device=x.device
    )
    block_m = _block_m(M)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, group_size * _GROUPS))
    _wide_act_mul_fp8_group_quant_kernel[grid](
        x,
        y,
        s,
        M,
        N,
        *x.stride(),
        *y.stride(),
        *s.stride(),
        BLOCK_M=block_m,
        GROUPS=_GROUPS,
        QB=group_size,
        # 240.0 for e4m3fnuz, matching aiter's kernel exactly -- this op
        # REPLACES aiter's, so keeping its constant keeps the swap a pure
        # performance change with bit-identical output, which is testable.
        # Note the divergence: _silu_mul_per_token_group_quant_fp8_colmajor
        # in fp8_utils.py deliberately clamps fnuz to 224.0, because 240.0
        # "will cause accuracy issue on dynamic quantization models". That
        # path is not this one, and moving to 224.0 here would be a numerics
        # change that needs an accuracy run behind it, not a drive-by edit.
        DTYPE_MAX=torch.finfo(dtype_quant).max,
    )
    return y, s
