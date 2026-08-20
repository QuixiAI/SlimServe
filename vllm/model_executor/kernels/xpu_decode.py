# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapters for decode operators provided by vllm-xpu-kernels."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


def has_xpu_decode_op(name: str) -> bool:
    return hasattr(torch.ops._xpu_C, name)


if has_xpu_decode_op("nvfp4_gemm"):

    @register_fake("_xpu_C::nvfp4_gemm")
    def _nvfp4_gemm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        block_scales: torch.Tensor,
        global_scale: float,
    ) -> torch.Tensor:
        return x.new_empty((*x.shape[:-1], weight.shape[0]))


if has_xpu_decode_op("nvfp4_moe"):

    @register_fake("_xpu_C::nvfp4_moe")
    def _nvfp4_moe_fake(
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w13_global_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        w2_global_scale: torch.Tensor,
        multiply_router_weight: bool,
    ) -> torch.Tensor:
        return hidden.new_empty(
            (hidden.shape[0], hidden.shape[-1]), dtype=torch.float32
        )


if has_xpu_decode_op("qwen_gdn_decode"):

    @register_fake("_xpu_C::qwen_gdn_decode")
    def _qwen_gdn_decode_fake(
        projected_qkvz: torch.Tensor,
        projected_ba: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (projected_qkvz.shape[0], 32, 128)
        return projected_qkvz.new_empty(shape), projected_qkvz.new_empty(shape)


def nvfp4_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: float,
) -> torch.Tensor:
    return torch.ops._xpu_C.nvfp4_gemm(x, weight, block_scales, global_scale)


def nvfp4_moe(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    multiply_router_weight: bool,
) -> torch.Tensor:
    return torch.ops._xpu_C.nvfp4_moe(
        hidden,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        multiply_router_weight,
    )


if has_xpu_decode_op("nvfp4_moe_relu2"):

    @register_fake("_xpu_C::nvfp4_moe_relu2")
    def _nvfp4_moe_relu2_fake(
        hidden, topk_ids, topk_weights, w1, w1_scale, w1_global_scale, w2,
        w2_scale, w2_global_scale, multiply_router_weight,
    ) -> torch.Tensor:
        return hidden.new_empty(
            (hidden.shape[0], hidden.shape[-1]), dtype=torch.float32
        )


def nvfp4_moe_relu2(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w1_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    multiply_router_weight: bool,
) -> torch.Tensor:
    return torch.ops._xpu_C.nvfp4_moe_relu2(
        hidden,
        topk_ids,
        topk_weights,
        w1,
        w1_scale,
        w1_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        multiply_router_weight,
    )


def qwen_gdn_decode(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._xpu_C.qwen_gdn_decode(
        projected_qkvz,
        projected_ba,
        conv_state,
        ssm_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        state_indices,
    )


if has_xpu_decode_op("mamba2_ssd_decode"):

    @register_fake("_xpu_C::mamba2_ssd_decode")
    def _mamba2_ssd_decode_fake(
        state, x, dt, A, B, C, D, dt_bias, src_indices, dst_indices, out,
        dt_softplus,
    ) -> None:
        return None


def mamba2_ssd_decode(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    out: torch.Tensor,
    dt_softplus: bool,
) -> None:
    torch.ops._xpu_C.mamba2_ssd_decode(
        state, x, dt, A, B, C, D, dt_bias, src_indices, dst_indices, out,
        dt_softplus,
    )


if has_xpu_decode_op("mamba2_conv1d_decode"):

    @register_fake("_xpu_C::mamba2_conv1d_decode")
    def _mamba2_conv1d_decode_fake(
        conv_state, x, weight, bias, indices, out, silu,
    ) -> None:
        return None


def mamba2_conv1d_decode(
    conv_state: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    indices: torch.Tensor,
    out: torch.Tensor,
    silu: bool,
) -> None:
    torch.ops._xpu_C.mamba2_conv1d_decode(
        conv_state, x, weight, bias, indices, out, silu,
    )


if has_xpu_decode_op("mamba2_conv1d_prefill"):

    @register_fake("_xpu_C::mamba2_conv1d_prefill")
    def _mamba2_conv1d_prefill_fake(
        conv_state, x, weight, bias, query_start_loc, cache_indices,
        has_initial_state, out, silu,
    ) -> None:
        return None


def mamba2_conv1d_prefill(
    conv_state: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    out: torch.Tensor,
    silu: bool,
) -> None:
    torch.ops._xpu_C.mamba2_conv1d_prefill(
        conv_state, x, weight, bias, query_start_loc, cache_indices,
        has_initial_state, out, silu,
    )


if has_xpu_decode_op("mamba2_ssd_prefill"):

    @register_fake("_xpu_C::mamba2_ssd_prefill")
    def _mamba2_ssd_prefill_fake(
        x, dt, A, B, C, D, dt_bias, query_start_loc, initial_states, out,
        varlen_states, dt_softplus, dt_limit_lo, dt_limit_hi,
    ) -> None:
        return None


def mamba2_ssd_prefill(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor,
    query_start_loc: torch.Tensor,
    initial_states: torch.Tensor | None,
    out: torch.Tensor,
    varlen_states: torch.Tensor,
    dt_softplus: bool,
    dt_limit_lo: float,
    dt_limit_hi: float,
) -> None:
    torch.ops._xpu_C.mamba2_ssd_prefill(
        x, dt, A, B, C, D, dt_bias, query_start_loc, initial_states, out,
        varlen_states, dt_softplus, dt_limit_lo, dt_limit_hi,
    )
