# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-native MPS fallbacks for the Qwen Gated DeltaNet layer.

The GDN compute path (causal_conv1d, flash-linear-attention chunk/recurrent
ops) is Triton-only; Metal has no Triton. These functions reproduce the exact
numerics of the Triton kernels they replace, in the subset of their contracts
that QwenGatedDeltaNetAttention._forward_core_mps actually exercises
(non-speculative decode + varlen prefill, continuous batching state caches).

Numerics contract (must match ops/causal_conv1d.py and
third_party/flash_linear_attention exactly — the M1 gate is greedy-token
parity against the MLX bf16 oracle):
- L2 norm: x * rsqrt(sum(x^2) + 1e-6), computed in fp32.
- Gating: g = -exp(A_log_fp32) * softplus(a + dt_bias), beta = sigmoid(b),
  fp32; softplus with beta=1, threshold=20 (== F.softplus defaults).
- Delta rule per token, fp32 state S[hv, V, K]: S *= exp(g); kv = S.k
  (post-decay); d = (v - kv) * beta; S += outer(d, k); y = S.(q * scale).
- GQA broadcast is GROUPED: value head i_hv reads key head i_hv // (HV // H)
  == repeat_interleave over the head dim.
- Recurrent state is written back in the cache dtype; slot 0 is the null
  block (skip read/write), mirroring the Triton state_idx <= 0 early-return.

The prefill scan replaces chunk_gated_delta_rule with the sequential
recurrence (the chunked algorithm computes the same function; llama.cpp's
Metal path is recurrent-only as well). Chunk-vs-recurrent fp32 deltas are
within the established parity bars (~3e-3 output / 1e-2 state).
"""

import torch
import torch.nn.functional as F


def l2norm_native(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA l2norm_fwd: normalize the last dim in fp32, keep input dtype."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


def _sigmoid_gating(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """g (log space) and beta, fp32, shapes [..., HV]."""
    g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())
    return g, beta


def causal_conv1d_update_native(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor,
) -> torch.Tensor:
    """Single-token decode conv (causal_conv1d_update, [batch, dim] shape).

    conv_state is [num_lines, dim, state_len] (a transposed view is fine:
    indexed writes go through to the base tensor). Rows whose state index is
    the null block (0) are skipped entirely.
    """
    valid = conv_state_indices > 0
    if not bool(valid.any()):
        return x
    idx = conv_state_indices[valid].long()

    # With speculation configured the pool carries width-1+num_spec columns
    # per channel; the non-spec ring is the FRONT width-1 columns (the tail
    # is spec-verify scratch), so read and write only that ring.
    width = weight.size(1)
    state = conv_state[idx, :, : width - 1].float()  # [n, dim, width-1]
    xt = x[valid].to(conv_state.dtype).float()  # [n, dim]
    window = torch.cat([state, xt.unsqueeze(-1)], dim=-1)  # [n, dim, width]

    out = torch.einsum("ndw,dw->nd", window[..., -width:], weight.float())
    if bias is not None:
        out = out + bias.float()
    if activation in ("silu", "swish"):
        out = F.silu(out)

    conv_state[idx, :, : width - 1] = window[..., 1:].to(conv_state.dtype)
    result = x.clone()
    result[valid] = out.to(x.dtype)
    return result


def causal_conv1d_fn_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_states: torch.Tensor,
    has_initial_state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Varlen prefill conv (causal_conv1d_fn, x is [dim, total_tokens]).

    Depthwise causal conv over each sequence, seeded from the cached state
    when has_initial_state, with the final width-1 inputs written back to
    conv_states[cache_indices[i]]. Returns [dim, total_tokens] in x.dtype.
    """
    dim, _ = x.shape
    width = weight.size(1)
    state_len = width - 1
    out = torch.empty_like(x)
    w = weight.float().unsqueeze(1)  # [dim, 1, width] depthwise
    bf = bias.float() if bias is not None else None

    starts = query_start_loc.tolist()
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        if e <= s:
            continue
        cache_idx = int(cache_indices[i])
        if cache_idx < 0:
            # PAD_SLOT_ID entry: skipped by the Triton kernel as well.
            continue
        xi = x[:, s:e].to(conv_states.dtype).float()  # [dim, T]
        if bool(has_initial_state[i]) and cache_idx > 0:
            # Front width-1 columns only: spec-configured pools are wider.
            prev = conv_states[cache_idx, :, :state_len].float()  # [dim, state_len]
        else:
            prev = torch.zeros(dim, state_len, dtype=torch.float32, device=x.device)
        full = torch.cat([prev, xi], dim=-1)  # [dim, state_len + T]
        yi = F.conv1d(full.unsqueeze(0), w, bias=bf, groups=dim).squeeze(0)
        if activation in ("silu", "swish"):
            yi = F.silu(yi)
        out[:, s:e] = yi.to(x.dtype)
        if cache_idx > 0:
            conv_states[cache_idx, :, :state_len] = full[:, -state_len:].to(
                conv_states.dtype
            )
    return out


def post_conv_prep_native(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fused_post_conv_prep: split conv output, L2-norm q/k, gating.

    Returns q/k [L, H, K], v [L, HV, V] in input dtype; g/beta [L, HV] fp32
    (g in log space, matching output_g_exp=False).
    """
    L = conv_output.shape[0]
    H, K, V = num_k_heads, head_k_dim, head_v_dim
    HV = A_log.shape[0]
    q, k, v = conv_output.split([H * K, H * K, HV * V], dim=-1)
    q = q.reshape(L, H, K)
    k = k.reshape(L, H, K)
    v = v.reshape(L, HV, V).contiguous()
    if apply_l2norm:
        q = l2norm_native(q)
        k = l2norm_native(k)
    g, beta = _sigmoid_gating(a, b, A_log, dt_bias)
    return q.contiguous(), k.contiguous(), v, g, beta


def gated_delta_rule_decode_native(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    tiled_gqa: bool = False,
) -> torch.Tensor:
    """fused_sigmoid_gating_delta_rule_update for the decode case
    (one token per sequence), vectorized over the batch.

    q/k: [1, N, H, K]; v: [1, N, HV, V]; a/b: [N, HV].
    Reads and writes ssm_state (cache layout [num_slots, HV, V, K]) in place
    at state_indices[:N]; slot <= 0 rows are skipped. Returns [1, N, HV, V].
    """
    _, N, H, K = q.shape
    HV = v.shape[2]
    ratio = HV // H
    if scale is None:
        scale = K**-0.5

    out = torch.zeros_like(v)
    idx_all = state_indices[:N].long()
    valid = idx_all > 0
    if not bool(valid.any()):
        return out
    idx = idx_all[valid]

    qf = q[0][valid].float()  # [n, H, K]
    kf = k[0][valid].float()
    vf = v[0][valid].float()  # [n, HV, V]
    if use_qk_l2norm:
        qf = qf * torch.rsqrt(qf.pow(2).sum(-1, keepdim=True) + 1e-6)
        kf = kf * torch.rsqrt(kf.pow(2).sum(-1, keepdim=True) + 1e-6)
    qf = qf * scale
    g, beta = _sigmoid_gating(a[:N][valid], b[:N][valid], A_log, dt_bias)

    if tiled_gqa:
        # ggml tiled layout: value head i_hv uses key head i_hv % H.
        q_hv = qf.repeat(1, ratio, 1)  # [n, HV, K]
        k_hv = kf.repeat(1, ratio, 1)
    else:
        # HF grouped layout: value head i_hv uses key head i_hv // ratio.
        q_hv = qf.repeat_interleave(ratio, dim=1)  # [n, HV, K]
        k_hv = kf.repeat_interleave(ratio, dim=1)

    S = ssm_state[idx].float()  # [n, HV, V, K]
    S = S * torch.exp(g)[..., None, None]
    kv = torch.einsum("nhvk,nhk->nhv", S, k_hv)
    d = (vf - kv) * beta[..., None]
    S = S + torch.einsum("nhv,nhk->nhvk", d, k_hv)
    y = torch.einsum("nhvk,nhk->nhv", S, q_hv)

    ssm_state[idx] = S.to(ssm_state.dtype)
    out[0][valid] = y.to(out.dtype)
    return out


def gated_delta_rule_prefill_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    tiled_gqa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """chunk_gated_delta_rule as a sequential fp32 recurrence.

    q/k: [1, T, H, K] (already L2-normalized); v: [1, T, HV, V];
    g/beta: [1, T, HV] fp32, g in log space; initial_state: [N, HV, V, K]
    (zeroed where no initial state, as prepared by the caller).
    Returns (o [1, T, HV, V] in v.dtype, final_state [N, HV, V, K] fp32).
    """
    _, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    ratio = HV // H
    if scale is None:
        scale = K**-0.5

    if tiled_gqa:
        # ggml tiled layout: value head i_hv uses key head i_hv % H.
        q_hv = (q[0].float() * scale).repeat(1, ratio, 1)  # [T, HV, K]
        k_hv = k[0].float().repeat(1, ratio, 1)
    else:
        # HF grouped layout: value head i_hv uses key head i_hv // ratio.
        q_hv = (q[0].float() * scale).repeat_interleave(ratio, dim=1)  # [T, HV, K]
        k_hv = k[0].float().repeat_interleave(ratio, dim=1)
    vf = v[0].float()
    decay = torch.exp(g[0].float())  # [T, HV]
    betaf = beta[0].float()

    out = torch.empty(T, HV, V, dtype=torch.float32, device=v.device)
    final_state = initial_state.float().clone()

    starts = cu_seqlens.tolist()
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        if e <= s:
            continue
        S = final_state[i]  # [HV, V, K], updated in place
        for t in range(s, e):
            S *= decay[t][:, None, None]
            kv = torch.einsum("hvk,hk->hv", S, k_hv[t])
            d = (vf[t] - kv) * betaf[t][:, None]
            S += torch.einsum("hv,hk->hvk", d, k_hv[t])
            out[t] = torch.einsum("hvk,hk->hv", S, q_hv[t])

    return out.to(v.dtype).unsqueeze(0), final_state


def scatter_num_accepted_native(
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_accepted: torch.Tensor,
) -> None:
    """_scatter_num_accepted_kernel: num_accepted[idx] = max(num_sampled, 1),
    skipping -1 sentinel rows."""
    valid = idx_mapping >= 0
    num_accepted[idx_mapping[valid].long()] = num_sampled[valid].clamp_min(1)
