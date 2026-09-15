# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-native GLM-5.3-Flash pooled indexer (Metal / CPU) against an
independent python reference.

The native producer (``glm5_next_indexer._pooled_select_native`` and
``_insert_rows_native``) must reproduce what the Triton kernels do on CUDA:

* pool keys ``softmax_m(gate(t_m) + ape[m]) . k(t_m)`` over each complete
  ``kp``-token pool (rounded through bf16 where the kernel feeds its dot),
* ``logit(q, p) = sum_h w_h relu(scale q_h . pool_key(p))`` over pools whose
  last token is visible,
* top-``ksel`` pools, expanded to raw tokens, then the incomplete tail
  pool's tokens IMMEDIATELY after the last valid pool, then ``-1`` padding
  to the row width ``roundup32(topk + kp - 1)``.

No CUDA here: the reference is float64 python over logical token rows and
never touches the kernels. Ties at the top-k boundary are tolerated (any
valid top-k set passes); everything else is exact.

Runs on CPU always and on MPS when available.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers import glm5_next_indexer as gi

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


@pytest.fixture(autouse=True)
def _force_native_on_cpu(monkeypatch):
    # CPU tensors keep the Triton route by default (the CUDA unit tests mock
    # those kernels); VLLM_METAL_GLM_INDEXER=native selects the torch producer.
    monkeypatch.setenv("VLLM_METAL_GLM_INDEXER", "native")

KP = 4
D = 128
H = 8
ROW = gi._ROW_DIM


# ----------------------------------------------------------------- reference


def _ref_pool_logits(rows: torch.Tensor, ape: torch.Tensor, q: torch.Tensor,
                     w: torch.Tensor, scale: float, visible: int,
                     kp: int) -> torch.Tensor:
    """float64 logits over the complete pools visible to one query row.
    rows: [T, 2D] (k | gate) in logical token order."""
    n_pools = visible // kp
    if n_pools == 0:
        return torch.empty(0, dtype=torch.float64)
    data = rows[: n_pools * kp].double().view(n_pools, kp, 2 * D)
    probs = torch.softmax(data[:, :, D:] + ape.double()[None], dim=1)
    keys = (probs * data[:, :, :D]).sum(dim=1)
    # The CUDA kernel dots a bf16 pool key; the native path mirrors that.
    keys = keys.to(torch.bfloat16).double()
    scores = torch.relu(keys @ q.double().T * scale)  # [P, H]
    return (scores * w.double()[None, :]).sum(dim=1)


def _check_row(out_row: list[int], logits: torch.Tensor, visible: int,
               ksel: int, kp: int) -> None:
    n_pools = visible // kp
    n_sel = min(n_pools, ksel)
    tail_count = visible - n_pools * kp
    tail_start = n_pools * kp
    width = len(out_row)
    # 1) valid pools first and contiguous (the sparse kernel walks
    #    [0, last_valid + 1)): n_sel runs of kp consecutive tokens.
    chosen = []
    for s in range(n_sel):
        run = out_row[s * kp : (s + 1) * kp]
        assert run[0] % kp == 0 and run == list(range(run[0], run[0] + kp)), run
        chosen.append(run[0] // kp)
    assert len(set(chosen)) == n_sel
    # 2) tail IMMEDIATELY after the valid pools (inside the pool region
    #    when fewer than ksel pools exist, exactly like _expand_topk_kernel).
    tcol = n_sel * kp
    assert out_row[tcol : tcol + tail_count] == list(
        range(tail_start, tail_start + tail_count)
    ), (out_row[tcol : tcol + kp], tail_start, tail_count)
    # 3) -1 padding for the rest of the row.
    assert all(v == -1 for v in out_row[tcol + tail_count : width])
    # 4) the chosen pools are a valid top-k set of the reference logits.
    if n_pools > ksel:
        order = torch.argsort(logits, descending=True)
        threshold = logits[order[ksel - 1]].item()
        tol = 1e-4 * max(1.0, logits.abs().max().item())
        must = {p for p in range(n_pools) if logits[p] > threshold + tol}
        may = {p for p in range(n_pools) if logits[p] >= threshold - tol}
        got = set(chosen)
        assert must <= got <= may, (sorted(must - got), sorted(got - may))
    else:
        assert set(chosen) == set(range(n_pools))
    # 5) the query's own token is selected whenever the reference
    #    guarantees it (in the tail, or every pool selected).
    own = visible - 1
    if visible > 0 and (tail_count > 0 or n_pools <= ksel):
        assert own in out_row


# --------------------------------------------------------------- fixtures


def _paged_cache(device, rows_per_req, block_size, page_padding, seed=0):
    """Scatter each request's logical rows into random physical pages of a
    (possibly strided) page cache; returns (cache, block_table, phys)."""
    g = torch.Generator().manual_seed(seed)
    nblk_req = [(len(r) + block_size - 1) // block_size for r in rows_per_req]
    total = sum(nblk_req) + 1  # block 0 stays the null block
    perm = torch.randperm(total - 1, generator=g) + 1
    backing = torch.full(
        (total, 1 + page_padding, block_size, ROW), float("nan"),
        dtype=torch.bfloat16,
    )
    cache = backing[:, 0]
    bt = torch.zeros((len(rows_per_req), max(nblk_req)), dtype=torch.int32)
    phys = []
    k = 0
    for r, rows in enumerate(rows_per_req):
        blocks = perm[k : k + nblk_req[r]]
        k += nblk_req[r]
        bt[r, : len(blocks)] = blocks.to(torch.int32)
        phys.append(blocks)
        for t in range(len(rows)):
            cache[blocks[t // block_size], t % block_size] = rows[t]
    return backing.to(device)[:, 0], bt.to(device), phys


def _inputs(R, seed):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(R, H, D, generator=g) * 0.5).to(torch.bfloat16)
    w = torch.randn(R, H, generator=g) * H**-0.5
    ape = torch.randn(KP, D, generator=g)
    return q, w, ape


def _rows(T, seed):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(T, D, generator=g)
    gate = torch.randn(T, D, generator=g)
    return torch.cat([k, gate], -1).to(torch.bfloat16)


# ------------------------------------------------------------------- tests


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("page_padding", [0, 2])
@pytest.mark.parametrize("topk", [16, 64])
@torch.no_grad()
def test_native_decode_rows(device, page_padding, topk):
    """Decode rows: one query per request at assorted contexts, including
    a 0-context pad row, pool-exact lengths (no tail), lengths below one
    pool, and contexts that prune pools."""
    ksel = topk // KP
    width = (topk + KP - 1 + 31) // 32 * 32
    visible = [0, 1, 3, 4, 5, 7, 16, 61, 64, 67, 133, 200, 261]
    block_size = 16
    rows = [_rows(v, 100 + i) for i, v in enumerate(visible)]
    cache, bt, _ = _paged_cache(device, rows, block_size, page_padding)
    R = len(visible)
    q, w, ape = _inputs(R, 7)
    scale = D**-0.5
    out = torch.full((R, width), 7, dtype=torch.int32, device=device)
    vis_t = torch.tensor(visible, dtype=torch.int32, device=device)
    row_req = torch.arange(R, dtype=torch.int32, device=device)
    max_pools = max(1, max(visible) // KP)
    gi._pooled_select(
        q.to(device), w.to(device), ape.to(device), cache, bt, row_req, vis_t,
        None, max_pools, block_size, scale, ksel, out, KP,
    )
    for r in range(R):
        logits = _ref_pool_logits(rows[r], ape, q[r], w[r], scale, visible[r], KP)
        _check_row(out[r].tolist(), logits, visible[r], ksel, KP)


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_native_pool_bound_and_tiling(device, monkeypatch):
    """A pool bound below the row's context must not change the result of
    rows within the bound, and a tiny tile forces the streaming top-k
    merge across many tiles."""
    monkeypatch.setattr(gi, "_NATIVE_TILE_BYTES", 1)  # -> 64-pool tiles
    monkeypatch.setattr(gi, "_NATIVE_ROW_CHUNK", 3)
    ksel, block_size = 32, 64
    width = (ksel * KP + KP - 1 + 31) // 32 * 32
    visible = [1000, 517, 64, 4099]
    rows = [_rows(v, 300 + i) for i, v in enumerate(visible)]
    cache, bt, _ = _paged_cache(device, rows, block_size, 1)
    R = len(visible)
    q, w, ape = _inputs(R, 9)
    scale = D**-0.5
    out = torch.full((R, width), -1, dtype=torch.int32, device=device)
    gi._pooled_select_native(
        q.to(device), w.to(device), ape.to(device), cache, bt,
        torch.arange(R, dtype=torch.int32, device=device),
        torch.tensor(visible, dtype=torch.int32, device=device),
        block_size, scale, ksel, out, KP, pool_bound=max(visible) // KP,
    )
    for r in range(R):
        logits = _ref_pool_logits(rows[r], ape, q[r], w[r], scale, visible[r], KP)
        _check_row(out[r].tolist(), logits, visible[r], ksel, KP)


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_native_expand_layout_matches_triton_contract(device):
    """Pure expansion: -1 pools, pools >= n_pools, tail placement, padding."""
    ksel, kp = 4, KP
    width = (ksel * kp + kp - 1 + 31) // 32 * 32
    sel = torch.tensor(
        [
            [3, 0, -1, -1],   # pool 3 is past the 3 pools -> dropped; tail of 2
            [5, 1, 2, 0],     # 4 of 6 pools, no tail
            [-1, -1, -1, -1], # empty context
            [0, -1, -1, -1],  # single pool, tail of 3
            [9, 2, 1, 0],     # a pool index past n_pools must be dropped
        ],
        dtype=torch.int32, device=device,
    )
    visible = torch.tensor([14, 24, 0, 7, 12], dtype=torch.int32, device=device)
    out = torch.full((5, width), 99, dtype=torch.int32, device=device)
    gi._expand_topk_native(sel, visible, out, kp, ksel)
    rows = out.tolist()
    # n_sel = min(3 pools, ksel) = 3, so the tail lands at column 12 even
    # though slot 0 was dropped (a real top-k never emits such a slot).
    assert rows[0][:16] == [-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, 12, 13, -1, -1]
    assert rows[0][16:] == [-1] * (width - 16)
    assert rows[1][:16] == [20, 21, 22, 23, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3]
    assert rows[1][16:] == [-1] * (width - 16)
    assert rows[2] == [-1] * width
    assert rows[3][:7] == [0, 1, 2, 3, 4, 5, 6] and rows[3][7:] == [-1] * (width - 7)
    # row 4: pool 9 >= 3 pools -> dropped; the valid pools (2, 1) keep
    # their slots; the (empty: 12 % 4 == 0) tail store lands on slot
    # n_sel=3 and clobbers pool 0 there, exactly as _expand_topk_kernel
    # does for this (never produced by a real top-k) input.
    assert rows[4][:16] == [-1, -1, -1, -1, 8, 9, 10, 11, 4, 5, 6, 7, -1, -1, -1, -1]


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_native_insert_rows(device):
    """Slot insert through a strided page view; PAD (-1) slots land on the
    null block (block 0, row 0) and touch nothing else."""
    block_size = 16
    backing = torch.full((5, 3, block_size, ROW), 2.0, dtype=torch.bfloat16)
    cache = backing.to(device)[:, 1]
    packed = _rows(6, 11).to(device)
    slots = torch.tensor([4 * 16 + 3, -1, 2 * 16 + 15, -1, 1 * 16, 4 * 16 + 2],
                         dtype=torch.int64, device=device)
    gi._insert_rows_native(packed, cache, slots)
    full = cache.cpu()
    assert torch.equal(full[4, 3], packed[0].cpu())
    assert torch.equal(full[2, 15], packed[2].cpu())
    assert torch.equal(full[1, 0], packed[4].cpu())
    assert torch.equal(full[4, 2], packed[5].cpu())
    written = {(4, 3), (2, 15), (1, 0), (4, 2), (0, 0)}
    for b in range(5):
        for t in range(block_size):
            if (b, t) not in written:
                assert (full[b, t] == 2.0).all(), (b, t)
    # the other page planes of the packed slab are untouched
    b_cpu = backing  # original host copy still all 2.0
    assert (b_cpu[:, 0] == 2.0).all() and (b_cpu[:, 2] == 2.0).all()


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_native_op_prefill_with_cached_prefix_and_decode(device):
    """The registered op body end to end on a mixed step: two decode rows
    (one plain, one 2-token spec verify) plus a prefill chunk of two
    requests, one extending a cached 40-token prefix by 20 tokens and one
    fresh 30-token prompt. Rows are inserted by the op through the slot
    mapping; the expected selection is the python reference over each
    request's logical rows."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
        DeepseekV32IndexerPrefillChunkMetadata,
        DeepseekV32IndexerPrefillMetadata,
    )

    topk, block_size = 32, 16
    ksel = topk // KP
    width = (topk + KP - 1 + 31) // 32 * 32
    scale = D**-0.5
    # request layout: dec0 (ctx 50 -> 1 new), dec1 (ctx 70 -> 2 new, spec),
    # pre0 (40 cached + 20 new), pre1 (30 new)
    ctx = [50, 70, 60, 30]
    new = [1, 2, 20, 30]
    rows_all = [_rows(c, 500 + i) for i, c in enumerate(ctx)]
    cached = [rows[: c - n] for rows, c, n in zip(rows_all, ctx, new)]
    cache, bt, phys = _paged_cache(device, cached, block_size, 1, seed=3)
    # the pages for the new tokens exist (allocated) but hold NaN until the
    # op inserts them; _paged_cache sized the pages from the cached rows,
    # so extend block tables where a request grows into a new page.
    need = [(c + block_size - 1) // block_size for c in ctx]
    if max(need) > bt.shape[1] or any(need[r] > len(phys[r]) for r in range(4)):
        extra = sum(max(0, need[r] - len(phys[r])) for r in range(4))
        nb_old = cache.shape[0]
        grown = torch.full(
            (nb_old + extra, 2, block_size, ROW), float("nan"),
            dtype=torch.bfloat16, device=device,
        )
        grown[:nb_old, 0] = cache
        cache = grown[:, 0]
        bt_new = torch.zeros((4, max(need)), dtype=torch.int32)
        nxt = nb_old
        for r in range(4):
            blocks = phys[r].tolist()
            while len(blocks) < need[r]:
                blocks.append(nxt)
                nxt += 1
            bt_new[r, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        bt = bt_new.to(device)
    T = sum(new)
    packed = torch.cat([rows_all[r][ctx[r] - new[r] :] for r in range(4)]).to(device)
    slots = []
    for r in range(4):
        for t in range(ctx[r] - new[r], ctx[r]):
            slots.append(int(bt[r, t // block_size]) * block_size + t % block_size)
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)
    q, w, ape = _inputs(T, 21)
    q, w, ape = q.to(device), w.to(device), ape.to(device)

    # decode metadata: 2-D seq_lens (B=2, next_n=2); dec0 pads its 2nd slot.
    dm = DeepSeekV32IndexerDecodeMetadata(
        block_table=bt[:2],
        seq_lens=torch.tensor([[50, 0], [69, 70]], dtype=torch.int32, device=device),
        decode_lens=torch.tensor([1, 2], dtype=torch.int32, device=device),
        requires_padding=True,
        schedule_metadata=torch.empty(0),
    )
    # NOTE: with 2-D seq_lens the op reads R = num_decode_tokens rows in
    # (b, j) order; dec0's padded j=1 row is a real query row here (a
    # scheduler would not emit it), so give it a 0 context (=> all -1).
    q_dec = torch.cat([q[0:1], torch.zeros_like(q[0:1]), q[1:3]])
    w_dec = torch.cat([w[0:1], torch.zeros_like(w[0:1]), w[1:3]])
    packed_dec = torch.cat([packed[0:1], packed[0:1], packed[1:3]])
    slots_dec = torch.tensor([slots[0], -1, slots[1], slots[2]],
                             dtype=torch.int64, device=device)
    # prefill chunk: rows follow the 4 decode rows.
    pre_rows = new[2] + new[3]
    ks = [0] * new[2] + [ctx[2]] * new[3]
    ke = [ctx[2] - new[2] + i + 1 for i in range(new[2])] + [
        ctx[2] + j + 1 for j in range(new[3])
    ]
    chunk = DeepseekV32IndexerPrefillChunkMetadata(
        block_table=bt[2:4],
        cu_seqlen_ks=torch.tensor(ks, dtype=torch.int32, device=device),
        cu_seqlen_ke=torch.tensor(ke, dtype=torch.int32, device=device),
        cu_seq_lens=torch.tensor([0, ctx[2], ctx[2] + ctx[3]], dtype=torch.int32,
                                 device=device),
        token_to_seq=torch.tensor([0] * ctx[2] + [1] * ctx[3], dtype=torch.int32,
                                  device=device),
        total_seq_lens=ctx[2] + ctx[3],
        max_seq_len=max(ctx[2], ctx[3]),
        token_start=4,
        token_end=4 + pre_rows,
        num_reqs=2,
    )
    md = DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor(ctx, dtype=torch.int32, device=device),
        max_seq_len=max(ctx),
        slot_mapping=torch.cat([slots_dec, slot_mapping[3:]]),
        num_decodes=2,
        num_decode_tokens=4,
        num_prefills=2,
        num_prefill_tokens=pre_rows,
        decode=dm,
        prefill=DeepseekV32IndexerPrefillMetadata([chunk]),
    )
    q_all = torch.cat([q_dec, q[3:]])
    w_all = torch.cat([w_dec, w[3:]])
    packed_all = torch.cat([packed_dec, packed[3:]])
    R = q_all.shape[0]
    buf = torch.full((R + 3, width), 5, dtype=torch.int32, device=device)
    tlen = torch.zeros((R + 3,), dtype=torch.int32, device=device)
    decode_logits = torch.empty((8, 4096 // KP), dtype=torch.float32, device=device)
    prefix = "model.layers.3.self_attn.indexer.k_cache"
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config), set_forward_context(
        {prefix: md}, vllm_config
    ):
        gi.glm5_next_pooled_indexer(
            q_all.contiguous(), packed_all.contiguous(), w_all.contiguous(),
            ape.contiguous(), prefix, cache, buf, decode_logits,
            decode_logits.shape[1], ksel, KP, scale, tlen,
        )
    out = buf.cpu().tolist()
    # rows beyond the batch untouched
    assert all(v == 5 for row in out[R:] for v in row)
    # the op inserted every new row (NaN pages become data)
    for r in range(4):
        for t in range(ctx[r] - new[r], ctx[r]):
            got = cache[int(bt[r, t // block_size]), t % block_size].cpu()
            assert torch.equal(got, rows_all[r][t])
    # decode rows: dec0 j=0 (vis 50), dec0 j=1 (vis 0), dec1 j=0 (69), j=1 (70)
    expect = [(0, 50, 0), (0, 0, 1), (1, 69, 2), (1, 70, 3)]
    for req, vis, row in expect:
        logits = _ref_pool_logits(rows_all[req], ape.cpu(), q_all[row].cpu(),
                                  w_all[row].cpu(), scale, vis, KP)
        _check_row(out[row], logits, vis, ksel, KP)
    # prefill rows
    for i in range(pre_rows):
        row = 4 + i
        req = 2 if i < new[2] else 3
        vis = ke[i] - ks[i]
        logits = _ref_pool_logits(rows_all[req], ape.cpu(), q_all[row].cpu(),
                                  w_all[row].cpu(), scale, vis, KP)
        _check_row(out[row], logits, vis, ksel, KP)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
