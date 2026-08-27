# SPDX-License-Identifier: Apache-2.0
"""TurboQuant k8v4 main-KV path: store + sparse-gather parity."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _dequant_slab(slab: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference dequant of a TQ k8v4 slab [b, bs, h, slot]."""
    b = slab.to(torch.int32)
    kb = b[..., :head_dim]
    sign = (kb >> 7) & 1
    exp = (kb >> 3) & 0xF
    man = kb & 0x7
    if torch.cuda.get_device_capability()[:2] < (8, 9):
        bias = 15  # Triton float8e4b15 on Ampere/Ada
        sub_scale = 2.0 ** (1 - bias) / 8.0
    else:
        bias = 7  # e4m3fn (e4nv) on SM89+
        sub_scale = 2.0 ** (1 - bias) / 8.0
    normal = (1.0 + man.float() / 8.0) * torch.pow(2.0, (exp - bias).float())
    subnormal = man.float() * sub_scale
    k = torch.where(exp > 0, normal, subnormal)
    k = torch.where(sign == 1, -k, k)

    val_bytes = head_dim // 2
    vb = b[..., head_dim : head_dim + val_bytes]
    lo = vb & 0xF
    hi = (vb >> 4) & 0xF
    nib = torch.stack((lo, hi), dim=-1).reshape(*vb.shape[:-1], head_dim)
    sc_off = head_dim + val_bytes
    scale = (
        (b[..., sc_off] | (b[..., sc_off + 1] << 8))
        .to(torch.uint16)
        .view(torch.float16)
        .float()
    )
    zero = (
        (b[..., sc_off + 2] | (b[..., sc_off + 3] << 8))
        .to(torch.uint16)
        .view(torch.float16)
        .float()
    )
    v = zero[..., None] + scale[..., None] * nib.float()
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


def test_qsa_sparse_paged_attention_tq_kv_matches_dequant_reference():
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.v1.attention.ops.triton_turboquant_store import (
        triton_turboquant_store,
    )

    torch.manual_seed(23)
    num_tokens, num_heads, kv_heads, head_dim = 12, 8, 1, 256
    page, pages, topk = 64, 32, 128
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", head_dim)
    slot = cfg.slot_size_aligned
    assert slot >= head_dim + head_dim // 2 + 4

    q = torch.randn(
        num_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        pages, page, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    slab = torch.zeros(pages, page, kv_heads, slot, device="cuda", dtype=torch.uint8)

    n = pages * page
    empty = torch.empty(0, dtype=torch.float32, device="cuda")
    triton_turboquant_store(
        k.reshape(n, kv_heads, head_dim),
        v.reshape(n, kv_heads, head_dim),
        slab,
        torch.arange(n, device="cuda", dtype=torch.int32),
        empty,
        empty,
        mse_bits=cfg.mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.value_quant_bits,
        key_fp8=True,
    )

    block_table = torch.arange(pages, device="cuda", dtype=torch.int32).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, device="cuda", dtype=torch.int32)
    idx = torch.randint(
        0, pages * page, (num_tokens, topk), device="cuda", dtype=torch.int32
    )

    out_tq = qsa_sparse_paged_attention(
        q, slab, slab, idx, block_table, token_to_req, tq_slot_size=slot
    )

    # Reference: torch-side dequant of the same slab through the bf16 kernel,
    # so the only difference is the in-kernel decode.
    k_ref, v_ref = _dequant_slab(slab, head_dim)
    out_ref = qsa_sparse_paged_attention(
        q, k_ref.contiguous(), v_ref.contiguous(), idx, block_table, token_to_req
    )
    assert torch.allclose(out_tq.float(), out_ref.float(), atol=2e-2, rtol=1e-2), (
        (out_tq.float() - out_ref.float()).abs().max().item()
    )

    # And the end-to-end quantization error against the raw bf16 cache stays
    # bounded (V is 4-bit per-vector uniform).
    out_bf16 = qsa_sparse_paged_attention(q, k, v, idx, block_table, token_to_req)
    err = (out_tq.float() - out_bf16.float()).abs().max().item()
    assert err < 0.5, err
