# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``torch.ops._C`` implementations for Intel XPU.

The CUDA/HIP stable-ABI extension that normally populates ``torch.ops._C`` is
not built for XPU, yet the shared layers reach for a handful of its elementwise
ops by name (RMSNorm, SiluAndMul, rotary embedding, GELUs). Register those
schemas here with XPU kernels: QuixiCore-XPU SYCL where the vendored library
has the op (``rms_norm``), plain torch otherwise. Registered once from the
platform's ``check_and_update_config``.

Everything else the DeepSeek-V4 path needs from ``_C`` (the fused
qnorm/rope/kv-insert, the GGUF ops) is routed at the layer level to Triton or
the QuixiCore-XPU GGUF binding; see vllm/models/deepseek_v4/xpu.py and
vllm/model_executor/layers/quantization/gguf/ops.py.
"""

from __future__ import annotations

import torch
from torch.library import Library

from vllm.logger import init_logger

logger = init_logger(__name__)

_LIB: Library | None = None


def _rms_norm_torch(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


def _rms_norm(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor | None, epsilon: float) -> None:
    from vllm.quixicore import quixicore_ops

    hidden = input.shape[-1]
    if (
        weight is not None
        and input.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden % 4 == 0
        and quixicore_ops.is_available()
    ):
        out = quixicore_ops.rms_norm(
            input.reshape(-1, hidden).contiguous(), weight, epsilon
        )
        result.copy_(out.view(input.shape))
        return
    result.copy_(_rms_norm_torch(input, weight, epsilon))


def _fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor | None, epsilon: float) -> None:
    residual.add_(input)
    _rms_norm(input, residual, weight, epsilon)


def _silu_and_mul(result: torch.Tensor, input: torch.Tensor) -> None:
    d = input.shape[-1] // 2
    result.copy_(torch.nn.functional.silu(input[..., :d]) * input[..., d:])


def _mul_and_silu(result: torch.Tensor, input: torch.Tensor) -> None:
    d = input.shape[-1] // 2
    result.copy_(input[..., :d] * torch.nn.functional.silu(input[..., d:]))


def _silu_and_mul_with_clamp(result: torch.Tensor, input: torch.Tensor, limit: float, alpha: float = 1.0, beta: float = 0.0) -> None:
    d = input.shape[-1] // 2
    gate = torch.clamp(input[..., :d].float(), max=limit)
    up = torch.clamp(input[..., d:].float(), min=-limit, max=limit)
    result.copy_((gate * torch.sigmoid(alpha * gate) * (up + beta)).to(result.dtype))


def _swigluoai_and_mul(result: torch.Tensor, input: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> None:
    # gpt-oss: interleaved gate/up pairs (x[..., 0::2] gate, x[..., 1::2] up)
    gate = torch.clamp(input[..., 0::2].float(), max=limit)
    up = torch.clamp(input[..., 1::2].float(), min=-limit, max=limit)
    result.copy_((gate * torch.sigmoid(alpha * gate) * (up + 1.0)).to(result.dtype))


def _situ_and_mul(result: torch.Tensor, input: torch.Tensor, beta: float = 1.0, linear_beta: float = -1.0) -> None:
    d = input.shape[-1] // 2
    gate = input[..., :d].float()
    up = input[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)
    result.copy_((gate * up).to(result.dtype))


def _gelu_and_mul(result: torch.Tensor, input: torch.Tensor) -> None:
    d = input.shape[-1] // 2
    result.copy_(torch.nn.functional.gelu(input[..., :d]) * input[..., d:])


def _gelu_tanh_and_mul(result: torch.Tensor, input: torch.Tensor) -> None:
    d = input.shape[-1] // 2
    result.copy_(torch.nn.functional.gelu(input[..., :d], approximate="tanh") * input[..., d:])


def _gelu_new(result: torch.Tensor, input: torch.Tensor) -> None:
    result.copy_(torch.nn.functional.gelu(input, approximate="tanh"))


def _gelu_fast(result: torch.Tensor, input: torch.Tensor) -> None:
    x = input.float()
    result.copy_((0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))).to(input.dtype))


def _gelu_quick(result: torch.Tensor, input: torch.Tensor) -> None:
    x = input.float()
    result.copy_((x * torch.sigmoid(1.702 * x)).to(input.dtype))


def _apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool,
    rope_dim_offset: int,
    inverse: bool,
) -> None:
    """In-place RoPE on x viewed as [tokens, heads, head_size]."""
    num_tokens = positions.shape[0]
    rot_dim = cos.shape[-1] * 2
    xv = x.view(num_tokens, -1, head_size)
    xr = xv[..., rope_dim_offset : rope_dim_offset + rot_dim].float()
    c = cos.unsqueeze(1).float()  # [tokens, 1, rot/2]
    s = sin.unsqueeze(1).float()
    if inverse:
        s = -s
    if is_neox:
        x1, x2 = xr.chunk(2, dim=-1)
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
        out = torch.cat((o1, o2), dim=-1)
    else:
        x1 = xr[..., 0::2]
        x2 = xr[..., 1::2]
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
        out = torch.stack((o1, o2), dim=-1).flatten(-2)
    xv[..., rope_dim_offset : rope_dim_offset + rot_dim] = out.to(x.dtype)


def _rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rope_dim_offset: int = 0,
    inverse: bool = False,
) -> None:
    pos = positions.flatten()
    cache = cos_sin_cache.index_select(0, pos)
    cos, sin = cache.chunk(2, dim=-1)
    _apply_rope(query, pos, head_size, cos, sin, is_neox, rope_dim_offset, inverse)
    if key is not None:
        _apply_rope(key, pos, head_size, cos, sin, is_neox, rope_dim_offset, inverse)


def _topk_masked(logits: torch.Tensor, valid: torch.Tensor, topk: int) -> torch.Tensor:
    """Top-k column indices per row over the `valid` mask; -1 where fewer valid."""
    masked = logits.float().masked_fill(~valid, float("-inf"))
    k = min(topk, masked.shape[1])
    values, indices = torch.topk(masked, k=k, dim=-1)
    indices = torch.where(values == float("-inf"), torch.full_like(indices, -1), indices)
    out = torch.full((logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device)
    out[:, :k] = indices.to(torch.int32)
    return out


def _top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk: int,
) -> None:
    # Same contract as the CUDA kernel: row i is valid on [start, end) of its
    # logits row and the emitted indices are LOCAL to that range.
    rows = logits[:num_rows]
    cols = torch.arange(rows.shape[1], device=rows.device)[None, :]
    starts = row_starts[:num_rows].to(torch.int64)[:, None]
    ends = row_ends[:num_rows].to(torch.int64)[:, None]
    valid = (cols >= starts) & (cols < ends)
    top = _topk_masked(rows, valid, topk)
    top = torch.where(top < 0, top, top - starts.to(torch.int32))
    indices[:num_rows, :topk] = top


def _top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    workspace: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk: int,
) -> None:
    del workspace
    rows = logits[:num_rows]
    cols = torch.arange(rows.shape[1], device=rows.device)[None, :]
    row_idx = torch.arange(num_rows, device=rows.device)
    if seq_lens.dim() == 2:
        # per-row effective lengths, C-contiguous [B, next_n]
        ends = seq_lens.reshape(-1)[:num_rows].to(torch.int64)
    else:
        b = row_idx // next_n
        j = row_idx % next_n
        ends = seq_lens.to(torch.int64)[b] - next_n + j + 1
    ends = ends.clamp(min=0)[:, None]
    valid = cols < ends
    indices[:num_rows, :topk] = _topk_masked(rows, valid, topk)


_SCHEMAS = {
    "rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) -> ()": _rms_norm,
    "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, float epsilon) -> ()": _fused_add_rms_norm,
    "silu_and_mul(Tensor! result, Tensor input) -> ()": _silu_and_mul,
    "mul_and_silu(Tensor! result, Tensor input) -> ()": _mul_and_silu,
    "silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit, float alpha=1.0, float beta=0.0) -> ()": _silu_and_mul_with_clamp,
    "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float limit=7.0) -> ()": _swigluoai_and_mul,
    "situ_and_mul(Tensor! out, Tensor input, float beta=1.0, float linear_beta=-1.0) -> ()": _situ_and_mul,
    "gelu_and_mul(Tensor! out, Tensor input) -> ()": _gelu_and_mul,
    "gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()": _gelu_tanh_and_mul,
    "gelu_new(Tensor! out, Tensor input) -> ()": _gelu_new,
    "gelu_fast(Tensor! out, Tensor input) -> ()": _gelu_fast,
    "gelu_quick(Tensor! out, Tensor input) -> ()": _gelu_quick,
    (
        "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, "
        "int head_size, Tensor cos_sin_cache, bool is_neox, "
        "int rope_dim_offset=0, bool inverse=False) -> ()"
    ): _rotary_embedding,
    (
        "top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
        "Tensor! indices, int numRows, int stride0, int stride1, int topK) -> ()"
    ): _top_k_per_row_prefill,
    (
        "top_k_per_row_decode(Tensor logits, int next_n, Tensor seq_lens, "
        "Tensor! indices, Tensor! workspace, int numRows, int stride0, "
        "int stride1, int topK) -> ()"
    ): _top_k_per_row_decode,
}


def register_xpu_c_ops() -> None:
    """Populate torch.ops._C with XPU implementations (idempotent)."""
    global _LIB
    if _LIB is not None:
        return
    lib = Library("_C", "FRAGMENT")
    _LIB = lib  # set first: imports below can re-enter through the platform
    registered = []
    for schema, fn in _SCHEMAS.items():
        name = schema.split("(", 1)[0]
        if hasattr(torch.ops._C, name):
            # Already provided (another extension defined it); only add the
            # XPU kernel if the schema exists without one.
            try:
                lib.impl(name, fn, "XPU")
                registered.append(name)
            except RuntimeError:
                pass
            continue
        lib.define(schema)
        lib.impl(name, fn, "XPU")
        registered.append(name)
    logger.info_once("Registered %d torch.ops._C XPU ops: %s", len(registered), ", ".join(registered))
    # The DeepSeek indexer logits ops (torch.ops.vllm.xpu_fp8_*).
    import vllm.v1.attention.ops.xpu_mla_sparse  # noqa: F401
