# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuixiCore gfx942 sparse MLA decode vs a pure-torch oracle.

The kernel's reason to exist is the packed cross-layer KV slab: each layer's
cache is a BLOCK-STRIDED view of one slab (rows contiguous inside a block,
blocks slab-stride apart), which aiter's mla_decode_fwd rejects. Every case
here therefore runs on a genuinely strided view, plus one contiguous control.
"""

import pytest
import torch

qc = pytest.importorskip("vllm._quixicore_C")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and hasattr(qc, "mla_sparse_decode_fwd")),
    reason="requires the ROCm QuixiCore sparse MLA op",
)

ENTRY, LATENT = 576, 512


def oracle(q, kv_rows, indptr, indices, scale):
    """[T,H,576] x gathered rows -> [T,H,512], fp32 softmax."""
    T, H, _ = q.shape
    out = torch.zeros(T, H, LATENT, dtype=torch.float32, device=q.device)
    for t in range(T):
        span = indices[indptr[t] : indptr[t + 1]]
        span = span[span >= 0]
        if span.numel() == 0:
            continue
        rows = kv_rows[span.long()].float()  # [K, 576]
        scores = (q[t].float() @ rows.T) * scale  # [H, K]
        probs = torch.softmax(scores, dim=-1)
        out[t] = probs @ rows[:, :LATENT]
    return out


def make_slab(num_blocks, block_size, layer_offset_rows, total_row_width, seed):
    """A packed-slab layer view: [NB, bs, 576] with block dim strided."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    slab = torch.randn(
        num_blocks,
        total_row_width,
        generator=g,
        device="cuda",
        dtype=torch.bfloat16,
    )
    start = layer_offset_rows * ENTRY
    view = slab[:, start : start + block_size * ENTRY].view(
        num_blocks, block_size, ENTRY
    )
    assert not view.is_contiguous()
    assert view.stride(1) == ENTRY and view.stride(2) == 1
    return slab, view


def flat_rows(view):
    """The view's rows as a contiguous [NB*bs, 576] copy (oracle input)."""
    return view.reshape(-1, ENTRY).contiguous()


@pytest.mark.parametrize("heads", [7, 12, 16])
@pytest.mark.parametrize("block_size", [64, 256])
def test_sparse_decode_matches_oracle_on_strided_slab(heads, block_size):
    torch.manual_seed(0)
    num_blocks, T, topk = 24, 5, 96
    slab, view = make_slab(
        num_blocks,
        block_size,
        layer_offset_rows=3,
        total_row_width=(block_size + 7) * ENTRY,
        seed=1,
    )
    q = torch.randn(T, heads, ENTRY, dtype=torch.bfloat16, device="cuda")
    total = num_blocks * block_size
    lens = torch.randint(1, topk, (T,))
    indptr = torch.zeros(T + 1, dtype=torch.int32)
    indices = []
    for t in range(T):
        n = int(lens[t])
        sel = torch.randperm(total)[:n].int()
        # Sprinkle -1 padding inside the span.
        if n > 2:
            sel[n // 2] = -1
        indices.append(sel)
        indptr[t + 1] = indptr[t] + n
    indices = torch.cat(indices).cuda()
    indptr = indptr.cuda()

    out = torch.empty(T, heads, LATENT, dtype=torch.bfloat16, device="cuda")
    qc.mla_sparse_decode_fwd(q, view, indptr, indices, out, 0.1, topk)
    ref = oracle(q, flat_rows(view), indptr.cpu(), indices.cpu(), 0.1)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_sparse_decode_contiguous_control_and_empty_span():
    torch.manual_seed(0)
    num_blocks, block_size, T, heads = 8, 64, 3, 12
    kv = torch.randn(num_blocks, block_size, ENTRY, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(T, heads, ENTRY, dtype=torch.bfloat16, device="cuda")
    # Token 1 has an empty span; its output must be zeros.
    indptr = torch.tensor([0, 40, 40, 100], dtype=torch.int32, device="cuda")
    indices = (
        torch.cat(
            [
                torch.randperm(num_blocks * block_size)[:40],
                torch.randperm(num_blocks * block_size)[:60],
            ]
        )
        .int()
        .cuda()
    )
    out = torch.empty(T, heads, LATENT, dtype=torch.bfloat16, device="cuda")
    qc.mla_sparse_decode_fwd(q, kv, indptr, indices, out, 0.13, 128)
    ref = oracle(q, kv.reshape(-1, ENTRY), indptr.cpu(), indices.cpu(), 0.13)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    assert torch.all(out[1] == 0)


def test_sparse_decode_deep_topk_splits():
    """topk 2048 (GLM's setting) exercises the multi-split reduce."""
    torch.manual_seed(0)
    num_blocks, block_size, T, heads, topk = 40, 64, 2, 12, 2048
    slab, view = make_slab(
        num_blocks,
        block_size,
        layer_offset_rows=1,
        total_row_width=(block_size + 3) * ENTRY,
        seed=2,
    )
    q = torch.randn(T, heads, ENTRY, dtype=torch.bfloat16, device="cuda")
    total = num_blocks * block_size
    indptr = torch.tensor([0, topk, 2 * topk], dtype=torch.int32, device="cuda")
    idx = (
        torch.cat(
            [
                torch.randint(0, total, (topk,)),
                torch.randint(0, total, (topk,)),
            ]
        )
        .int()
        .cuda()
    )
    out = torch.empty(T, heads, LATENT, dtype=torch.bfloat16, device="cuda")
    qc.mla_sparse_decode_fwd(q, view, indptr, idx, out, 0.042, topk)
    ref = oracle(q, flat_rows(view), indptr.cpu(), idx.cpu(), 0.042)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_blocks", [160, 256])
def test_sparse_decode_long_kv_high_block_ids(num_blocks):
    """GLM at 10K-16K context: block ids beyond 128, topk 2048 spans drawn
    from the whole range and from the far tail (residual garbling bisect,
    perf/optimization_status.md 2026-09-01 late entry)."""
    torch.manual_seed(1)
    block_size, T, heads, topk = 64, 3, 12, 2048
    slab, view = make_slab(
        num_blocks,
        block_size,
        layer_offset_rows=2,
        total_row_width=(block_size + 5) * ENTRY,
        seed=7,
    )
    q = torch.randn(T, heads, ENTRY, dtype=torch.bfloat16, device="cuda")
    total = num_blocks * block_size
    spans = [
        torch.randperm(total)[:topk].sort().values,  # whole range, sorted
        torch.arange(total - topk, total),  # far tail only
        torch.randint(total // 2, total, (topk,)),  # upper half, dups
    ]
    indptr = torch.tensor(
        [0, topk, 2 * topk, 3 * topk], dtype=torch.int32, device="cuda"
    )
    idx = torch.cat(spans).int().cuda()
    out = torch.empty(T, heads, LATENT, dtype=torch.bfloat16, device="cuda")
    qc.mla_sparse_decode_fwd(q, view, indptr, idx, out, 0.042, topk)
    ref = oracle(q, flat_rows(view), indptr.cpu(), idx.cpu(), 0.042)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
