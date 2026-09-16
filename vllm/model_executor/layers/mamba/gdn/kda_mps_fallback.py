# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-native MPS/CPU execution path for the Kimi Delta Attention layer.

The KDA compute path (packed causal_conv1d, ``fused_recurrent_kda``,
``fused_recurrent_kda_packed_decode``, ``chunk_kda_with_fused_gate``) is
Triton-only; Metal has no Triton. These functions reproduce the numerics of
the Triton kernels they replace, in the contracts that
``KimiGatedDeltaNetAttention._forward_native`` exercises: plain decode,
speculative verify with state rollback, and varlen (chunked) prefill on the
continuous-batching state pools. They mirror ``gdn_mps_fallback.py`` (the
Qwen GDN precedent) in API shape.

Numerics contract (vllm/models/kimi_k3/amd/ops/third_party/kda/
fused_recurrent.py is the reference; the NVIDIA path re-exports it):
- Gate (``_kda_gate_beta_fwd_kernel`` :46-77, ``fused_recurrent_kda_fwd_kernel``
  :215-233, packed decode :493-503): ``x = raw_g + dt_bias`` (fp32),
  ``a = exp(A_log[h])``; with a lower bound ``gate = lower_bound *
  sigmoid(a * x)``, otherwise ``gate = -a * softplus(x)`` with the Triton
  softplus ``where(x > 20, x, log(1 + exp(x)))``. Gate is per key channel,
  shape ``[..., H, K]``, in log space. ``beta = sigmoid(raw_beta)``.
- q/k L2 norm in-kernel (:206-208): ``x / sqrt(sum(x*x) + 1e-6)``, fp32,
  then ``q *= K**-0.5``.
- Recurrence, fp32 state ``S[h, V, K]`` (:235-243):
  ``S *= exp(gate)[None, :]`` (decay along the KEY axis, broadcast over V);
  ``hk = S . k`` (sum over K); ``d = (v - hk) * beta``;
  ``S += outer(d, k)`` (``d[:, None] * k[None, :]``); ``o = S . q``.
- Spec verify (:186-192, :253-266): the initial state is read from
  ``ssm_state_indices[i, num_accepted_tokens[i] - 1]``; after EVERY
  position t the running state is stored to ``ssm_state_indices[i, t]`` when
  that slot is > 0. A resume slot <= 0 skips the row (zero output, no store).
- Decode (:447-533): state read/written in place at ``state_indices[i]``;
  slot <= 0 rows are skipped with zero output.
- Prefill replaces the chunked algorithm (``chunk_kda_with_fused_gate``) with
  the same sequential recurrence (the chunked form computes the same
  function; the chunk path additionally rounds l2norm(q/k) to the input
  dtype, which this path does not - fp32 throughout, as the recurrent
  kernels do). Initial state comes from the pool (zeros where
  ``has_initial_state`` is False, as ``gather_initial_states`` produces) and
  the final state is written back to ``state_indices[i]``.
- Conv (ops/causal_conv1d.py): depthwise causal conv over the packed
  [q|k|v] channels with SiLU; the non-spec ring is the FRONT ``width-1``
  columns of each pool row; spec verify (``_causal_conv1d_update_kernel``
  with IS_SPEC_DECODING, :836-928) reads the window at column offset
  ``num_accepted-1`` and rewrites the row from column 0 as
  ``[window[1:], x_0..x_{s-1}]``.

Hot-path shape discipline: decode and spec verify use only static-shape ops
(NULL slots are routed to slot 0 through ``torch.where`` and written back
with their own old contents, never boolean-masked). Prefill needs the host
copy of ``cu_seqlens`` / state indices (as the Qwen path does); it is
pulled once per step through ``KdaVarlenPlan`` and cached on the metadata.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

_L2NORM_EPS = 1e-6
_SOFTPLUS_THRESHOLD = 20.0


# --------------------------------------------------------------------------
# Elementwise pieces
# --------------------------------------------------------------------------


def kda_gate_native(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    """KDA log-decay gate, fp32, same shape as ``raw_g`` (``[..., H, K]``).

    Both gate forms of the Triton kernels: ``lower_bound * sigmoid(a * x)``
    when a lower bound is configured (GLM: -5.0), else ``-a * softplus(x)``
    with the kernels' threshold-20 softplus. ``a = exp(A_log)`` per head.
    """
    H, K = raw_g.shape[-2], raw_g.shape[-1]
    x = raw_g.float()
    if dt_bias is not None:
        x = x + dt_bias.float().reshape(H, K)
    a = A_log.float().exp().reshape(H, 1)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a * x)
    softplus = torch.where(x > _SOFTPLUS_THRESHOLD, x, torch.log(1.0 + torch.exp(x)))
    return -a * softplus


def kda_beta_native(raw_beta: torch.Tensor) -> torch.Tensor:
    """``beta = sigmoid(raw_beta)`` in fp32."""
    return torch.sigmoid(raw_beta.float())


def kda_l2norm_native(x: torch.Tensor) -> torch.Tensor:
    """In-kernel q/k normalization: ``x / sqrt(sum(x^2) + 1e-6)`` in fp32."""
    xf = x.float()
    return xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) + _L2NORM_EPS)


def gated_rmsnorm_sigmoid_native(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """``FusedRMSNormGated(activation="sigmoid").forward_native``:
    ``rmsnorm(x) * weight * sigmoid(g)``, fp32 math, output in ``x.dtype``."""
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xn = xf * torch.rsqrt(variance + eps)
    if weight is not None:
        xn = xn * weight.float()
    return (xn * torch.sigmoid(g.float())).to(x.dtype)


def _prep_qk(
    q: torch.Tensor, k: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 L2-normalized q (scaled) and k."""
    return kda_l2norm_native(q) * scale, kda_l2norm_native(k)


def _kda_step(
    S: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One out-of-place KDA recurrence step.

    S: ``[..., H, V, K]`` fp32; q/k: ``[..., H, K]``; v: ``[..., H, V]``;
    decay: ``[..., H, K]`` (``exp(gate)``); beta: ``[..., H]``.
    Returns ``(S_new, o)`` with ``o: [..., H, V]``.
    """
    S = S * decay.unsqueeze(-2)
    hk = torch.matmul(S, k.unsqueeze(-1))  # [..., H, V, 1]
    d = (v.unsqueeze(-1) - hk) * beta[..., None, None]
    S = S + d * k.unsqueeze(-2)
    o = torch.matmul(S, q.unsqueeze(-1)).squeeze(-1)
    return S, o


def _kda_step_(
    S: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """In-place variant of ``_kda_step`` (S mutated). Returns ``o``."""
    S.mul_(decay.unsqueeze(-2))
    hk = torch.matmul(S, k.unsqueeze(-1))
    d = (v.unsqueeze(-1) - hk) * beta[..., None, None]
    S.add_(d * k.unsqueeze(-2))
    return torch.matmul(S, q.unsqueeze(-1)).squeeze(-1)


def _route_null(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(valid mask, indices with NULL/pad slots (<= 0) routed to slot 0)."""
    idx = idx.long()
    valid = idx > 0
    return valid, torch.where(valid, idx, torch.zeros_like(idx))


def _split_packed_qkv(
    x: torch.Tensor, H: int, K: int, V: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed post-conv ``[..., H*K + H*K + H*V]`` rows into heads."""
    q, k, v = x.split([H * K, H * K, H * V], dim=-1)
    lead = x.shape[:-1]
    return q.reshape(*lead, H, K), k.reshape(*lead, H, K), v.reshape(*lead, H, V)


# --------------------------------------------------------------------------
# Varlen host plan (prefill)
# --------------------------------------------------------------------------


@dataclass
class KdaVarlenPlan:
    """Host-side view of a non-spec varlen batch, pulled once per step.

    ``groups`` maps a sequence length to the rows of that length (rows with
    zero length are dropped); ``tok_idx`` caches the flat token index tensor
    per multi-row group so the conv and recurrence share one gather.
    """

    starts: list[int]
    lens: list[int]
    slots: list[int]
    has_init: list[bool]
    groups: dict[int, list[int]] = field(default_factory=dict)
    _tok_idx: dict[int, torch.Tensor] = field(default_factory=dict)
    _row_slots: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    @classmethod
    def build(
        cls,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor | None,
    ) -> "KdaVarlenPlan":
        starts = [int(s) for s in cu_seqlens.tolist()]
        num_seqs = len(starts) - 1
        lens = [starts[i + 1] - starts[i] for i in range(num_seqs)]
        slots = [int(s) for s in state_indices[:num_seqs].tolist()]
        if has_initial_state is None:
            has_init = [True] * num_seqs
        else:
            has_init = [bool(b) for b in has_initial_state[:num_seqs].tolist()]
        groups: dict[int, list[int]] = {}
        for i in range(num_seqs):
            if lens[i] > 0:
                groups.setdefault(lens[i], []).append(i)
        return cls(starts[:-1], lens, slots, has_init, groups)

    def tok_idx(self, s: int, device: torch.device) -> torch.Tensor:
        """Flat token indices ``[n * s]`` of the rows in group ``s``."""
        t = self._tok_idx.get(s)
        if t is None:
            rows = self.groups[s]
            t = torch.tensor(
                [self.starts[i] + j for i in rows for j in range(s)],
                dtype=torch.long,
                device=device,
            )
            self._tok_idx[s] = t
        return t

    def row_slots(
        self, s: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(read slots with NULL routed to 0, ``[n]``; write pairs) for group
        ``s``. The write pairs are ``(group_row, slot)`` tensors restricted to
        rows whose slot is > 0."""
        hit = self._row_slots.get(s)
        if hit is None:
            rows = self.groups[s]
            read = torch.tensor(
                [max(self.slots[i], 0) for i in rows], dtype=torch.long, device=device
            )
            w_rows = [j for j, i in enumerate(rows) if self.slots[i] > 0]
            w_slots = [self.slots[rows[j]] for j in w_rows]
            write = (
                torch.tensor(w_rows, dtype=torch.long, device=device),
                torch.tensor(w_slots, dtype=torch.long, device=device),
            )
            hit = (read, write)
            self._row_slots[s] = hit
        return hit[0], hit[1]

    def no_init_rows(self, s: int) -> list[int]:
        rows = self.groups[s]
        return [
            j
            for j, i in enumerate(rows)
            if not (self.has_init[i] and self.slots[i] > 0)
        ]


def _gather_group(x: torch.Tensor, plan: KdaVarlenPlan, s: int) -> torch.Tensor:
    """``[n, s, ...]`` tokens of group ``s`` (a view when n == 1)."""
    rows = plan.groups[s]
    if len(rows) == 1:
        start = plan.starts[rows[0]]
        return x[start : start + s].unsqueeze(0)
    return x.index_select(0, plan.tok_idx(s, x.device)).view(len(rows), s, *x.shape[1:])


def _scatter_group(
    out: torch.Tensor, y: torch.Tensor, plan: KdaVarlenPlan, s: int
) -> None:
    """Write ``y [n, s, ...]`` of group ``s`` into the flat ``out [T, ...]``."""
    rows = plan.groups[s]
    if len(rows) == 1:
        start = plan.starts[rows[0]]
        out[start : start + s] = y[0]
        return
    out.index_copy_(0, plan.tok_idx(s, out.device), y.reshape(-1, *y.shape[2:]))


# --------------------------------------------------------------------------
# Convolution
# --------------------------------------------------------------------------


def _conv_windowed(
    padded: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    seq_len: int,
) -> torch.Tensor:
    """``out[..., t] = act(bias + sum_w weight[:, w] * padded[..., t + w])``
    for ``padded: [..., dim, width-1+seq_len]`` fp32 -> ``[..., dim, seq_len]``."""
    width = weight.size(-1)
    out = weight[:, 0:1] * padded[..., 0:seq_len]
    for w in range(1, width):
        out = out + weight[:, w : w + 1] * padded[..., w : w + seq_len]
    if bias is not None:
        out = out + bias[:, None]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out


def kda_conv_update_native(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor,
) -> torch.Tensor:
    """Single-token decode conv (``causal_conv1d_update``, ``[B, dim]``).

    ``conv_state`` is ``[num_slots, dim, L]`` (a transposed view is fine);
    only the front ``width-1`` ring columns are read/written. NULL rows
    (slot <= 0) produce their input unchanged (the Triton kernel returns
    before writing the output) and leave the pool untouched.
    """
    B = x.shape[0]
    width = weight.size(-1)
    ring = width - 1
    valid, slot0 = _route_null(conv_state_indices[:B])
    wf = weight.float()
    rows = conv_state[:, :, :ring].index_select(0, slot0).float()  # [B, dim, ring]
    xf = x.float()
    window = torch.cat([rows, xf.unsqueeze(-1)], dim=-1)  # [B, dim, width]
    out = (window * wf).sum(-1)
    if bias is not None:
        out = out + bias.float()
    if activation in ("silu", "swish"):
        out = F.silu(out)
    new_rows = window[..., 1:]
    vmask = valid.view(B, 1, 1)
    conv_state[slot0, :, :ring] = torch.where(vmask, new_rows, rows).to(
        conv_state.dtype
    )
    return torch.where(valid.view(B, 1), out, xf).to(x.dtype)


def kda_conv_prefill_native(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    plan: KdaVarlenPlan,
) -> torch.Tensor:
    """Varlen prefill conv (``causal_conv1d_fn``), ``x: [T, dim]`` -> same.

    Seeded from the front ``width-1`` ring of ``conv_state[slot]`` when the
    sequence has an initial state (and a real slot), zeros otherwise; the
    final ``width-1`` inputs are written back to rows with slot > 0.
    Sequences are batched by length (one gather per group).
    """
    width = weight.size(-1)
    ring = width - 1
    dim = x.shape[1]
    wf = weight.float()
    bf = bias.float() if bias is not None else None
    out = torch.empty_like(x)
    ring_view = conv_state[:, :, :ring]
    for s in plan.groups:
        read_slots, (w_rows, w_slots) = plan.row_slots(s, x.device)
        xs = _gather_group(x, plan, s).float().transpose(1, 2)  # [n, dim, s]
        n = xs.shape[0]
        prev = ring_view.index_select(0, read_slots).float()  # [n, dim, ring]
        no_init = plan.no_init_rows(s)
        if len(no_init) == n:
            prev = torch.zeros(n, dim, ring, dtype=torch.float32, device=x.device)
        elif no_init:
            prev[torch.tensor(no_init, dtype=torch.long, device=x.device)] = 0
        padded = torch.cat([prev, xs], dim=-1)  # [n, dim, ring + s]
        y = _conv_windowed(padded, wf, bf, activation, s)
        _scatter_group(out, y.transpose(1, 2).to(x.dtype), plan, s)
        if w_slots.numel():
            ring_view[w_slots] = padded[w_rows, :, -ring:].to(conv_state.dtype)
    return out


def _dense_spec_layout(
    cu_seqlens: torch.Tensor, max_len: int, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Static-shape varlen <-> dense maps for a spec batch of ``B`` rows.

    Returns ``(lens [B], tok [B, max_len] clamped token index per dense
    cell, flat [num_tokens] dense cell per real token)``.
    """
    B = cu_seqlens.numel() - 1
    starts = cu_seqlens[:-1].long()
    lens = cu_seqlens[1:].long() - starts
    ar = torch.arange(max_len, device=cu_seqlens.device)
    tok = (starts[:, None] + ar[None, :]).clamp_(max=max(num_tokens - 1, 0))
    pos = torch.arange(num_tokens, device=cu_seqlens.device)
    # row(p) = number of sequence ends <= p (sequence of each real token).
    row = (pos[:, None] >= cu_seqlens[None, 1:].long()).sum(1).clamp_(max=B - 1)
    flat = row * max_len + (pos - starts.index_select(0, row))
    return lens, tok, flat


def kda_conv_spec_update_native(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
) -> torch.Tensor:
    """Spec-verify conv with rollback (``causal_conv1d_update`` varlen +
    IS_SPEC_DECODING), ``x: [T, dim]`` flat spec tokens -> same shape.

    Row i (slot ``conv_state_indices[i]``, ``s_i`` tokens, ``off =
    num_accepted_tokens[i] - 1``): the window is
    ``state[:, off : off + width-1]``; afterwards the row is rewritten from
    column 0 as ``[state[:, off+1 : off+width-1], x_0..x_{s_i-1}]``
    (``width-2+s_i`` columns; the tail keeps its old contents). Rows with a
    NULL slot or zero length are untouched. Static shapes only.
    """
    B = query_start_loc.numel() - 1
    T, dim = x.shape
    width = weight.size(-1)
    L = conv_state.shape[-1]
    ncols = width - 2 + max_query_len
    assert ncols <= L, f"conv pool has {L} columns, spec verify needs {ncols}"
    device = x.device

    lens, tok, flat = _dense_spec_layout(query_start_loc, max_query_len, T)
    valid, slot0 = _route_null(conv_state_indices[:B])
    valid = valid & (lens > 0)
    acc = num_accepted_tokens[:B].long().clamp(min=1)
    off = acc - 1

    rows = conv_state.index_select(0, slot0).float()  # [B, dim, L]
    win_idx = (off[:, None, None] + torch.arange(width - 1, device=device)).expand(
        B, dim, width - 1
    )
    win = rows.gather(2, win_idx)  # [B, dim, width-1]
    xd = x.index_select(0, tok.reshape(-1)).view(B, max_query_len, dim)
    xd = xd.float().transpose(1, 2)  # [B, dim, max_len]
    padded = torch.cat([win, xd], dim=-1)
    y = _conv_windowed(
        padded,
        weight.float(),
        bias.float() if bias is not None else None,
        activation,
        max_query_len,
    )
    out = y.transpose(1, 2).reshape(B * max_query_len, dim).index_select(0, flat)

    new_cols = torch.cat([win[..., 1:], xd], dim=-1)  # [B, dim, ncols]
    col_ok = torch.arange(ncols, device=device)[None, :] < (width - 2 + lens)[:, None]
    new_rows = rows.clone()
    new_rows[..., :ncols] = torch.where(col_ok[:, None, :], new_cols, rows[..., :ncols])
    conv_state[slot0] = torch.where(valid.view(B, 1, 1), new_rows, rows).to(
        conv_state.dtype
    )
    return out.to(x.dtype)


# --------------------------------------------------------------------------
# Recurrence
# --------------------------------------------------------------------------


def kda_recurrent_decode_native(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """``fused_recurrent_kda_packed_decode``: one token per row, batched.

    mixed_qkv: ``[B, H*K + H*K + H*V]`` post-conv rows; raw_g ``[1, B, H, K]``;
    raw_beta ``[1, B, H]``; ssm_state ``[num_slots, H, V, K]`` read/written
    in place at ``state_indices[:B]`` (slot <= 0 rows: zero output, pool
    untouched). Returns ``[1, B, H, V]`` in ``mixed_qkv.dtype``.
    """
    _, H, V, K = ssm_state.shape
    B = mixed_qkv.shape[0]
    if scale is None:
        scale = K**-0.5
    q, k, v = _split_packed_qkv(mixed_qkv, H, K, V)
    qf, kf = _prep_qk(q, k, scale)
    vf = v.float()
    decay = torch.exp(kda_gate_native(raw_g[0, :B], A_log, dt_bias, lower_bound))
    beta = kda_beta_native(raw_beta[0, :B])

    valid, slot0 = _route_null(state_indices[:B])
    S = ssm_state.index_select(0, slot0).float()  # [B, H, V, K]
    S_new, o = _kda_step(S, qf, kf, vf, decay, beta)
    ssm_state[slot0] = torch.where(valid.view(B, 1, 1, 1), S_new, S).to(ssm_state.dtype)
    o = torch.where(valid.view(B, 1, 1), o, torch.zeros_like(o))
    return o.to(mixed_qkv.dtype).unsqueeze(0)


def kda_recurrent_prefill_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    ssm_state: torch.Tensor,
    plan: KdaVarlenPlan,
    scale: float | None = None,
) -> torch.Tensor:
    """``chunk_kda_with_fused_gate`` + state gather/scatter as a sequential
    fp32 recurrence over ``cu_seqlens`` segments.

    q/k ``[1, T, H, K]``, v ``[1, T, H, V]`` (post-conv, un-normalized);
    raw_g ``[1, T, H, K]``; raw_beta ``[1, T, H]``. The initial state of row
    i is ``ssm_state[slot_i]`` when ``has_initial_state[i]`` (zeros
    otherwise); the final state is written back to rows with slot > 0.
    Rows are batched by sequence length. Returns ``[1, T, H, V]`` in
    ``v.dtype``.
    """
    _, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    qf, kf = _prep_qk(q[0], k[0], scale)  # [T, H, K]
    vf = v[0].float()
    decay = torch.exp(kda_gate_native(raw_g[0], A_log, dt_bias, lower_bound))
    beta = kda_beta_native(raw_beta[0])
    out = torch.empty((T, H, V), dtype=v.dtype, device=v.device)

    for s in plan.groups:
        read_slots, (w_rows, w_slots) = plan.row_slots(s, q.device)
        n = read_slots.numel()
        S = ssm_state.index_select(0, read_slots).float()  # [n, H, V, K] (copy)
        no_init = plan.no_init_rows(s)
        if len(no_init) == n:
            S.zero_()
        elif no_init:
            S[torch.tensor(no_init, dtype=torch.long, device=q.device)] = 0
        q_g = _gather_group(qf, plan, s)
        k_g = _gather_group(kf, plan, s)
        v_g = _gather_group(vf, plan, s)
        d_g = _gather_group(decay, plan, s)
        b_g = _gather_group(beta, plan, s)
        y = torch.empty((n, s, H, V), dtype=torch.float32, device=q.device)
        for t in range(s):
            y[:, t] = _kda_step_(
                S, q_g[:, t], k_g[:, t], v_g[:, t], d_g[:, t], b_g[:, t]
            )
        _scatter_group(out, y.to(v.dtype), plan, s)
        if w_slots.numel():
            ssm_state[w_slots] = S[w_rows].to(ssm_state.dtype)
    return out.unsqueeze(0)


def kda_recurrent_prefill_metal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    ssm_state: torch.Tensor,
    plan: KdaVarlenPlan,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """``kda_recurrent_prefill_native`` with the per-token recurrence on the
    Metal ``kda_recur`` kernel (one simdgroup per request x head x Dv rows
    walking the request's tokens) instead of the Python step loop. Test /
    bench harness for the kernel (serving runs the recurrence inside the
    fused ``kda_step``): the prep (L2 norm, gate, beta) is the same
    vectorised torch as the native path; rows without an initial state
    have their pool slot zeroed before the kernel loads it. Same shapes and
    return contract as the native function; ``cu_seqlens`` /
    ``state_indices`` are the device int32 tensors the plan was built
    from."""
    from vllm.quixicore import quixicore_ops

    _, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    qf, kf = _prep_qk(q[0], k[0], scale)
    decay = torch.exp(kda_gate_native(raw_g[0], A_log, dt_bias, lower_bound))
    beta = kda_beta_native(raw_beta[0])
    num_seqs = len(plan.lens)
    zero_slots = [
        plan.slots[i]
        for i in range(num_seqs)
        if plan.slots[i] > 0 and plan.lens[i] > 0 and not plan.has_init[i]
    ]
    if zero_slots:
        ssm_state[torch.tensor(zero_slots, dtype=torch.long, device=q.device)] = 0
    y = quixicore_ops.kda_recur_prefill(
        qf.reshape(T, H * K).contiguous(),
        kf.reshape(T, H * K).contiguous(),
        v[0].float().reshape(T, H * V).contiguous(),
        decay.reshape(T, H * K).contiguous(),
        beta.reshape(T, H).contiguous(),
        ssm_state,
        cu_seqlens[: num_seqs + 1].to(torch.int32).contiguous(),
        state_indices[:num_seqs].to(torch.int32).contiguous(),
        K,
        V,
        True,
    )
    return y.view(T, H, V).to(v.dtype).unsqueeze(0)


def kda_recurrent_spec_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    ssm_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """``fused_recurrent_kda`` with ``num_accepted_tokens`` (spec verify).

    q/k ``[1, T, H, K]``, v ``[1, T, H, V]`` flat spec tokens of ``B =
    len(cu_seqlens) - 1`` rows; ``ssm_state_indices [B, >= max_query_len]``.
    Row i resumes from ``ssm_state[ssm_state_indices[i, num_accepted[i]-1]]``
    and, after every position t < len_i, stores its running state to
    ``ssm_state_indices[i, t]`` when that slot is > 0. A resume slot <= 0
    skips the row (zero output, nothing stored). Static shapes only (rows
    are padded to ``max_query_len`` and masked with ``torch.where``).
    Returns ``[1, T, H, V]`` in ``v.dtype``.
    """
    _, T, H, K = q.shape
    V = v.shape[-1]
    B = cu_seqlens.numel() - 1
    if max_query_len is None:
        max_query_len = ssm_state_indices.shape[1]
    if scale is None:
        scale = K**-0.5
    rows = ssm_state_indices[:B].long()
    lens, tok, flat = _dense_spec_layout(cu_seqlens, max_query_len, T)
    flat_tok = tok.reshape(-1)

    def dense(x: torch.Tensor) -> torch.Tensor:
        return x.index_select(0, flat_tok).view(B, max_query_len, *x.shape[1:])

    qf, kf = _prep_qk(q[0], k[0], scale)
    qd, kd, vd = dense(qf), dense(kf), dense(v[0].float())
    dd = dense(torch.exp(kda_gate_native(raw_g[0], A_log, dt_bias, lower_bound)))
    bd = dense(kda_beta_native(raw_beta[0]))

    acc = num_accepted_tokens[:B].long().clamp(min=1)
    resume = rows.gather(1, (acc - 1).unsqueeze(1)).squeeze(1)
    row_valid, resume0 = _route_null(resume)
    S = ssm_state.index_select(0, resume0).float()  # [B, H, V, K]
    o_dense = torch.zeros(
        (B, max_query_len, H, V), dtype=torch.float32, device=q.device
    )
    for t in range(max_query_len):
        active = row_valid & (lens > t)
        S_new, o = _kda_step(S, qd[:, t], kd[:, t], vd[:, t], dd[:, t], bd[:, t])
        S = torch.where(active.view(B, 1, 1, 1), S_new, S)
        o_dense[:, t] = torch.where(active.view(B, 1, 1), o, torch.zeros_like(o))
        slot_t = rows[:, t]
        ok = active & (slot_t > 0)
        slot0 = torch.where(ok, slot_t, torch.zeros_like(slot_t))
        old = ssm_state.index_select(0, slot0).float()
        ssm_state[slot0] = torch.where(ok.view(B, 1, 1, 1), S, old).to(ssm_state.dtype)
    out = o_dense.view(B * max_query_len, H, V).index_select(0, flat)
    return out.to(v.dtype).unsqueeze(0)
