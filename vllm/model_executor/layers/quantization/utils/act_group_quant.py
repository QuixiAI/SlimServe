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


# Timing is flat across (BLOCK_M, GROUPS) once the tile is wide enough, so these
# are fixed rather than autotuned -- autotune benchmarks at first call, which is
# not safe under CUDA graph capture.
_BLOCK_M = 8
_GROUPS = 8


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
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, group_size * _GROUPS))
    _wide_act_mul_fp8_group_quant_kernel[grid](
        x,
        y,
        s,
        M,
        N,
        *x.stride(),
        *y.stride(),
        *s.stride(),
        BLOCK_M=_BLOCK_M,
        GROUPS=_GROUPS,
        QB=group_size,
        DTYPE_MAX=torch.finfo(dtype_quant).max,
    )
    return y, s
