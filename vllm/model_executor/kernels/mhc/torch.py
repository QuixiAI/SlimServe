# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
    if x.device.type == "xpu":
        # oneDNN's fp32 GEMM for this tall-skinny shape ([T, 16384] x
        # [16384, 12]) splits K with a nondeterministic accumulation order
        # (measured 2026-08-18: K >= 4096 non-reproducible, K <= 2048
        # reproducible at every M). Reduce over fixed 2048-wide K chunks in
        # program order so the residual-stream mix is bit-reproducible;
        # native SYCL mhc_pre replaces this (perf notebook).
        fn32 = fn_flat.to(torch.float32)
        k_total = x.shape[1]
        chunk = 2048
        mixes = None
        for k0 in range(0, k_total, chunk):
            part = torch.matmul(x[:, k0 : k0 + chunk], fn32[:, k0 : k0 + chunk].t())
            mixes = part if mixes is None else mixes + part
    else:
        mixes = torch.matmul(x, fn_flat.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    ).to(torch.bfloat16)
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    # out[..., j, h] = sum_i comb[..., i, j] * residual[..., i, h] + post[..., j] * x[..., h]
    # Written as a broadcast product + fixed-order sum over the (4-wide) stream
    # axis rather than an einsum/bmm: bit-reproducible on every backend and no
    # library kernel selection in the residual-stream path (XPU determinism
    # bisection, perf notebook 2026-08-18).
    comb = comb_res_mix.to(torch.float32)  # [..., i, j]
    res = residual.to(torch.float32)  # [..., i, h]
    mixed_residual = (comb.unsqueeze(-1) * res.unsqueeze(-2)).sum(dim=-3)  # [..., j, h]
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)


def hc_head_fused_torch(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Torch reference for hc_head_fuse_tilelang (used when tilelang is
    unavailable, e.g. sm80 deployments that avoid JIT codegen backends).

    Per token: RMS statistic over all hc_mult*H elements, hc_mult projections
    onto fn rows, sigmoid gate, then a gated sum of the hc copies.
    """
    num_tokens, hc_mult, hidden_size = hs_flat.shape
    x = hs_flat.to(torch.float32).reshape(num_tokens, hc_mult * hidden_size)
    rsqrt_val = torch.rsqrt(
        x.square().sum(dim=-1, keepdim=True) / (hc_mult * hidden_size) + rms_eps
    )
    mixes = x @ fn.to(torch.float32).t()  # (T, hc_mult)
    pre_mix = torch.sigmoid(mixes * rsqrt_val * hc_scale + hc_base) + hc_eps
    out = torch.einsum(
        "tm,tmh->th", pre_mix, hs_flat.to(torch.float32)
    )
    return out.to(torch.bfloat16)
