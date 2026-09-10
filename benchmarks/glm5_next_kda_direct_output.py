# SPDX-License-Identifier: Apache-2.0
"""Quarantined direct-output launch of the unchanged packed KDA kernel.

No serving import. This tests whether deleting the caller's output copy is
bit-exact before extending the owned wrapper with an optional output buffer.
Launch constants intentionally match that wrapper; no arithmetic changes.
"""

from vllm.models.kimi_k3.amd.ops.third_party.kda.fused_recurrent import (
    fused_recurrent_kda_packed_decode_kernel,
)
from vllm.utils.math_utils import cdiv, next_power_of_2


def direct_output(
    mixed_qkv, raw_g, raw_beta, a_log, dt_bias, lower_bound, state, indices, out
):
    rows = mixed_qkv.shape[0]
    _, heads, v, k = state.shape
    assert out.shape == (1, rows, heads, v)
    assert out.is_contiguous() and out.dtype == mixed_qkv.dtype
    assert out.device == mixed_qkv.device
    assert mixed_qkv.stride(-1) == 1 and indices.is_contiguous()
    assert state.stride()[1:] == (v * k, k, 1)
    if rows == 0:
        return out
    bv = min(next_power_of_2(v), 32)
    fused_recurrent_kda_packed_decode_kernel[(cdiv(v, bv), rows * heads)](
        mixed_qkv=mixed_qkv,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=a_log,
        dt_bias=dt_bias,
        out=out,
        state=state,
        state_indices=indices,
        lower_bound=lower_bound or 0.0,
        scale=k**-0.5,
        stride_mixed_token=mixed_qkv.stride(0),
        stride_g_token=raw_g.stride(1),
        stride_beta_token=raw_beta.stride(1),
        stride_state_token=state.stride(0),
        H=heads,
        K=k,
        V=v,
        BK=next_power_of_2(k),
        BV=bv,
        SOFTPLUS_THRESHOLD=20.0,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=4,
        num_stages=2,
    )
    return out
