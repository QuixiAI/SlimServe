# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
"""Phase 4.2 full-head conv/state/norm prototype; no serving dispatch.

Adapts the owned GDN whole-head fusion boundary to GLM53's KDA equations.
The arithmetic follows causal_conv1d_update, fused_recurrent_kda_packed_decode
and the FLA gated RMSNorm. Projections remain on their existing kernels.
"""

import torch

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton


@triton.jit
def _conv_head(x, weight, state, features, slot):
    offsets = slot * 18432 + features
    c0 = tl.load(state + offsets)
    c1 = tl.load(state + offsets + 6144)
    c2 = tl.load(state + offsets + 12288)
    current = tl.load(x + features)
    w0 = tl.load(weight + features * 4)
    w1 = tl.load(weight + features * 4 + 1)
    w2 = tl.load(weight + features * 4 + 2)
    w3 = tl.load(weight + features * 4 + 3)
    # Match the original expression's BF16 operands and FP32 accumulator,
    # including the explicit post-convolution BF16 materialization boundary.
    acc = tl.full((128,), 0, tl.float32)
    acc += c0 * w0
    acc += c1 * w1
    acc += c2 * w2
    acc += current * w3
    value = acc / (1 + tl.exp(-acc))
    # Whole-head recurrence broadcasts these vectors across multiple warps.
    # All duplicated readers must finish before the canonical writer shifts
    # the in-place conv history. Shared-memory racecheck cannot detect this
    # global-memory read/write dependency within the CTA.
    tl.debug_barrier()
    tl.store(state + offsets, c1)
    tl.store(state + offsets + 6144, c2)
    tl.store(state + offsets + 12288, current)
    return value.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _core(
    mixed,
    raw_g,
    output_gate,
    raw_beta,
    conv_weight,
    conv_state,
    a_log,
    dt_bias,
    state,
    indices,
    norm_weight,
    output,
    debug_conv,
    MIXED_STRIDE: tl.constexpr,
    G_STRIDE: tl.constexpr,
    BETA_STRIDE: tl.constexpr,
):
    nh = tl.program_id(0)
    token, head = nh // 16, nh % 16
    slot = tl.load(indices + token).to(tl.int64)
    k = tl.arange(0, 128)
    v = tl.arange(0, 128)
    if slot <= 0:
        tl.store(output + nh * 128 + v, 0.0)
        return
    x = mixed + token * MIXED_STRIDE
    features = head * 128 + k
    q = _conv_head(x, conv_weight, conv_state, features, slot)
    key = _conv_head(x, conv_weight, conv_state, 2048 + features, slot)
    value = _conv_head(x, conv_weight, conv_state, 4096 + features, slot)
    if debug_conv is not None:
        tl.store(debug_conv + token * 6144 + features, q)
        tl.store(debug_conv + token * 6144 + 2048 + features, key)
        tl.store(debug_conv + token * 6144 + 4096 + features, value)
    q /= tl.sqrt(tl.sum(q * q) + 1e-6)
    key /= tl.sqrt(tl.sum(key * key) + 1e-6)
    q *= 128**-0.5
    g = tl.load(raw_g + token * G_STRIDE + features).to(tl.float32)
    bias = tl.load(dt_bias + features).to(tl.float32)
    decay = exp(tl.load(a_log + head).to(tl.float32))
    g += bias
    gate = -5.0 * tl.sigmoid(decay * g)
    p_state = state + slot * 16 * 128 * 128 + head * 128 * 128
    p_state += v[:, None] * 128 + k[None, :]
    h = tl.load(p_state).to(tl.float32)
    h *= exp(gate[None, :])
    value -= tl.sum(h * key[None, :], axis=1)
    beta = tl.sigmoid(tl.load(raw_beta + token * BETA_STRIDE + head).to(tl.float32))
    value *= beta
    h += value[:, None] * key[None, :]
    result = tl.sum(h * q[None, :], axis=1)
    tl.store(p_state, h)
    # The existing recurrent op writes BF16 before gated normalization.
    result = result.to(tl.bfloat16).to(tl.float32)
    rstd = 1 / tl.sqrt(tl.sum(result * result) / 128 + 1e-5)
    weight = tl.load(norm_weight + v).to(tl.float32)
    out_gate = tl.load(output_gate + token * G_STRIDE + features).to(tl.float32)
    result = ((result * rstd) * weight) * tl.sigmoid(out_gate)
    tl.store(output + nh * 128 + v, result.to(tl.bfloat16))


def core(mixed, g1, g2, beta, weights, conv_state, state, indices, output, debug=None):
    batch = mixed.shape[0]
    if weights["conv"].shape != (6144, 4) or not weights["conv"].is_contiguous():
        raise ValueError("conv weights must be contiguous [6144, 4]")
    if (
        output.shape != (batch, 2048)
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous BF16 [batch, 2048]")
    if (
        indices.shape != (batch,)
        or indices.dtype not in (torch.int32, torch.int64)
        or not indices.is_contiguous()
    ):
        raise ValueError("indices must be contiguous integer [batch]")
    assert mixed.shape == (batch, 6144) and mixed.dtype == torch.bfloat16
    assert conv_state.stride() == (18432, 1, 6144)
    assert state.dtype == torch.float32 and state.shape[1:] == (16, 128, 128)
    assert state.is_contiguous()
    assert g1.stride() == g2.stride() == (2048, 1)
    assert weights["norm"].numel() == 128
    return _core[(batch * 16,)](
        mixed,
        g1,
        g2,
        beta,
        weights["conv"],
        conv_state,
        weights["a_log"],
        weights["dt_bias"],
        state,
        indices,
        weights["norm"],
        output,
        debug,
        MIXED_STRIDE=mixed.stride(0),
        G_STRIDE=g1.stride(0),
        BETA_STRIDE=beta.stride(0),
        num_warps=8,
        num_stages=2,
    )
