# SPDX-License-Identifier: Apache-2.0
"""Paired, unquantized KDA low-rank projections for Ampere decode.

The independent f_b and g_b matmuls have K=128 and consume strided views
of the merged input projection. Schedule them in one grid, without copying
inputs or repacking/duplicating checkpoint weights. Single-token decode
uses FP32 reductions; batched decode uses BF16 tensor cores.
"""

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _gate_pair_kernel(
    AF,
    AG,
    WF,
    WG,
    OF,
    OG,
    M: tl.constexpr,
    N: tl.constexpr,
    SF: tl.constexpr,
    SG: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    group = tl.program_id(2)
    ap = tl.where(group == 0, AF, AG)
    wp = tl.where(group == 0, WF, WG)
    op = tl.where(group == 0, OF, OG)
    stride = tl.where(group == 0, SF, SG)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, 128)
    if M == 1:
        a = tl.load(ap + k).to(tl.float32)
        w = tl.load(wp + n[:, None] * 128 + k[None, :], n[:, None] < N, 0).to(
            tl.float32
        )
        out = tl.sum(w * a[None, :], axis=1)
        tl.store(op + n, out, n < N)
    else:
        m = tl.program_id(1) * BM + tl.arange(0, BM)
        a = tl.load(ap + m[:, None] * stride + k[None, :], m[:, None] < M, 0)
        w = tl.load(wp + n[None, :] * 128 + k[:, None], n[None, :] < N, 0)
        out = tl.dot(a, w)
        tl.store(
            op + m[:, None] * N + n[None, :], out, (m[:, None] < M) & (n[None, :] < N)
        )


def kda_gate_pair(
    f_a: torch.Tensor,
    g_a: torch.Tensor,
    f_weight: torch.Tensor,
    g_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert f_a.ndim == g_a.ndim == f_weight.ndim == g_weight.ndim == 2
    assert f_a.shape == g_a.shape and f_weight.shape == g_weight.shape
    assert f_a.shape[1] == f_weight.shape[1] == 128
    assert all(
        t.dtype == torch.bfloat16 and t.device == f_a.device
        for t in (f_a, g_a, f_weight, g_weight)
    )
    assert f_a.is_cuda and f_a.stride(1) == g_a.stride(1) == 1
    assert f_weight.is_contiguous() and g_weight.is_contiguous()
    m, n = f_a.shape[0], f_weight.shape[0]
    # Leave larger prefill GEMMs on cuBLAS. This kernel is measured for the
    # registered decode capture sizes, not a general matrix-multiply API.
    if m > 64 or m == 0:
        return F.linear(f_a, f_weight), F.linear(g_a, g_weight)
    out_f = torch.empty((m, n), device=f_a.device, dtype=f_a.dtype)
    out_g = torch.empty_like(out_f)
    bn = 16 if m == 1 else 32
    _gate_pair_kernel[(triton.cdiv(n, bn), triton.cdiv(m, 16), 2)](
        f_a,
        g_a,
        f_weight,
        g_weight,
        out_f,
        out_g,
        m,
        n,
        f_a.stride(0),
        g_a.stride(0),
        16,
        bn,
        num_warps=4,
    )
    return out_f, out_g


def _kda_gate_pair_fake(f_a, g_a, f_weight, g_weight):
    shape = (f_a.shape[0], f_weight.shape[0])
    return f_a.new_empty(shape), g_a.new_empty(shape)


direct_register_custom_op(
    op_name="kda_gate_pair",
    op_func=kda_gate_pair,
    mutates_args=[],
    fake_impl=_kda_gate_pair_fake,
)
