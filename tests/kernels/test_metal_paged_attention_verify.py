# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle for the multi-query verify attention kernel (paged_attention_verify).

The m rows of a speculative verify block share one request's KV cache; row i
sees context positions [0, ctx - m + i + 1). The kernel must match a dense
fp32 reference on that causal window, and it must treat an unmapped block
(block_table entry < 0) the way paged_attention_partition does: skip the
tokens, rather than score a zero tile and inflate the softmax denominator.

Run directly: .venv/bin/python tests/kernels/test_metal_paged_attention_verify.py
"""
import math

import torch

try:  # collected by pytest in CI; hand-run on the serving box without it
    import pytest

    pytestmark = pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="requires Apple Metal (MPS)",
    )
except ModuleNotFoundError:
    pass

DEV = "mps"


def _case(
    *,
    m: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    ctx_base: int,
    dtype: torch.dtype,
    unmapped_blocks: tuple[int, ...] = (),
    seed: int = 0,
):
    torch.manual_seed(seed)
    ctx = ctx_base + m
    n_slots = (ctx + block_size - 1) // block_size
    n_blocks = n_slots + 4
    # non-identity block table so a stride bug cannot hide behind block == col
    perm = torch.randperm(n_blocks)[:n_slots].to(torch.int32)
    block_table = perm.clone()
    for col in unmapped_blocks:
        block_table[col] = -1
    block_table = block_table[None, :].contiguous().to(DEV)
    key_cache = torch.randn(n_blocks, block_size, kv_heads, head_dim).to(dtype).to(DEV)
    value_cache = torch.randn(n_blocks, block_size, kv_heads, head_dim).to(dtype).to(DEV)
    q = torch.randn(m, heads, head_dim).to(dtype).to(DEV)
    context_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    scale = 1.0 / math.sqrt(head_dim)
    return q, key_cache, value_cache, block_table, context_lens, scale


def _reference(q, key_cache, value_cache, block_table, ctx: int, scale: float):
    """Dense fp32 attention over the mapped tokens of each row's causal window."""
    m, heads, head_dim = q.shape
    kv_heads = key_cache.shape[2]
    block_size = key_cache.shape[1]
    group = heads // kv_heads
    bt = block_table[0].cpu()
    tokens = torch.arange(ctx)
    blocks = bt[tokens // block_size]
    mapped = blocks >= 0
    slots = tokens % block_size
    k = key_cache.cpu().float()[blocks.clamp(min=0), slots]  # [ctx, KV, D]
    v = value_cache.cpu().float()[blocks.clamp(min=0), slots]
    qf = q.cpu().float()
    out = torch.zeros(m, heads, head_dim)
    for row in range(m):
        row_end = ctx - m + row + 1
        keep = mapped[:row_end]
        for h in range(heads):
            kh = h // group
            kk = k[:row_end, kh][keep]
            vv = v[:row_end, kh][keep]
            s = (kk @ qf[row, h]) * scale
            p = torch.softmax(s, dim=0)
            out[row, h] = p @ vv
    return out


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    # output is rounded to the activation dtype once; bf16 keeps ~3 digits
    return (2e-2, 2e-2) if dtype == torch.bfloat16 else (5e-3, 5e-3)


def _run(**kw):
    from vllm.quixicore.ops import quixicore_ops

    q, kc, vc, bt, cl, scale = _case(**kw)
    out = quixicore_ops.paged_attention_verify(q, kc, vc, bt, cl, scale)
    out2 = quixicore_ops.paged_attention_verify(q, kc, vc, bt, cl, scale)
    torch.mps.synchronize()
    assert torch.equal(out, out2), "verify kernel is not deterministic"
    ref = _reference(q, kc, vc, bt, int(cl[0]), scale)
    atol, rtol = _tol(kw["dtype"])
    torch.testing.assert_close(out.cpu().float(), ref, atol=atol, rtol=rtol)
    return out


def test_verify_matches_reference_d128() -> None:
    # the metallib instantiates bf16 at D=128 and both dtypes at D=256
    _run(m=4, heads=8, kv_heads=2, head_dim=128, block_size=16,
         ctx_base=700, dtype=torch.bfloat16)


def test_verify_matches_reference_d256() -> None:
    # Qwen3.8's full-attention layers: bf16 KV at head_dim 256, m up to k+1;
    # the fp16 twin is the Muse-Glimmer route
    for dtype in (torch.bfloat16, torch.float16):
        _run(m=4, heads=4, kv_heads=2, head_dim=256, block_size=16,
             ctx_base=1100, dtype=dtype)


def test_verify_single_row_and_max_rows() -> None:
    _run(m=1, heads=8, kv_heads=2, head_dim=128, block_size=16,
         ctx_base=40, dtype=torch.bfloat16)
    _run(m=32, heads=4, kv_heads=4, head_dim=256, block_size=16,
         ctx_base=520, dtype=torch.float16)


def test_verify_skips_unmapped_blocks() -> None:
    # An unmapped block inside the context is a host-invariant violation, but
    # the kernel's defensive behaviour must agree with paged_attention_partition:
    # skip those tokens. Scoring the zero tile they were staged as would add
    # exp(0 - max) per token to the denominator and attenuate every row.
    # short context so the two unmapped blocks are ~30% of the window and
    # the attenuation is far outside the bf16 tolerance
    _run(m=4, heads=8, kv_heads=2, head_dim=128, block_size=16,
         ctx_base=100, dtype=torch.bfloat16, unmapped_blocks=(1, 4))


def main() -> None:
    test_verify_matches_reference_d128()
    test_verify_matches_reference_d256()
    test_verify_single_row_and_max_rows()
    print("verify kernel matches the dense reference on mapped context")
    test_verify_skips_unmapped_blocks()
    print("verify kernel skips unmapped blocks like the partition kernel")


if __name__ == "__main__":
    main()
