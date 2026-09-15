# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-native sparse NoPE MLA backend (Apple Metal / CPU) against a plain
reference attention over each query row's selected latent positions.

Covers, on CPU and MPS:
* the gathered path row by row (decode rows, spec-verify rows with several
  query rows per request, long prefill rows), including rows with -1 pads
  and a row with no valid position,
* dense == sparse when every position is selected (the short-prefill
  ``selected_limit`` path), including a cached prefix and forced row chunking,
* the latent insert through a strided page view with PAD slots,
* the metadata builder + impl end to end on a mixed batch through the
  ``MLAAttentionImpl`` contract (``do_kv_cache_update`` then ``forward_mqa``).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import metal_mla_sparse as mm
from vllm.v1.kv_cache_interface import MLAAttentionSpec

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")

H = 4          # query heads
DK = 64        # latent width (kv_lora_rank; qk_rope_head_dim == 0)
BS = 16        # block size
SCALE = 1 / math.sqrt(256)


# ----------------------------------------------------------------- reference


def _ref_attend(q_row: torch.Tensor, rows: torch.Tensor, positions: list[int],
                scale: float) -> torch.Tensor:
    """Plain float64 attention of one query row [H, DK] over the latent rows
    at the given positions (the request's logical rows [T, DK])."""
    if not positions:
        return torch.zeros(q_row.shape[0], rows.shape[1], dtype=torch.float64)
    kv = rows[positions].double()
    s = q_row.double() @ kv.T * scale
    p = torch.softmax(s, dim=-1)
    return p @ kv


def _paged(device, rows_per_req, page_padding=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    nblk = [(len(r) + BS - 1) // BS for r in rows_per_req]
    total = sum(nblk) + 1
    perm = torch.randperm(total - 1, generator=g) + 1
    backing = torch.full((total, 1 + page_padding, BS, DK), float("nan"),
                         dtype=torch.bfloat16)
    cache = backing[:, 0]
    bt = torch.zeros((len(rows_per_req), max(nblk)), dtype=torch.int32)
    k = 0
    for r, rows in enumerate(rows_per_req):
        blocks = perm[k : k + nblk[r]]
        k += nblk[r]
        bt[r, : len(blocks)] = blocks.to(torch.int32)
        for t in range(len(rows)):
            cache[blocks[t // BS], t % BS] = rows[t]
    return backing.to(device)[:, 0], bt.to(device)


def _rows(T, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(T, DK, generator=g) * 0.7).to(torch.bfloat16)


def _q(R, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(R, H, DK, generator=g)).to(torch.bfloat16)


def _sel(visible: int, k: int, width: int, seed: int) -> list[int]:
    """A request-local selection of min(k, visible) positions padded to width
    with -1 (interleaved pads, like a real -1-filled top-k row may carry)."""
    g = torch.Generator().manual_seed(seed)
    n = min(k, visible)
    chosen = torch.randperm(visible, generator=g)[:n].tolist() if n else []
    row = [-1] * width
    slots = torch.randperm(width, generator=g)[:n].tolist()
    for s, c in zip(slots, chosen):
        row[s] = c
    return row


# ------------------------------------------------------------------- tests


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_sparse_rows_vs_reference(device, monkeypatch):
    monkeypatch.setattr(mm, "_SPARSE_ROW_CHUNK", 3)  # force row chunking
    ctx = [37, 5, 70, 0, 129]
    rows = [_rows(c, 10 + i) for i, c in enumerate(ctx)]
    cache, bt = _paged(device, rows)
    width = 24
    sel = [_sel(c, 20, width, 50 + i) for i, c in enumerate(ctx)]
    # duplicate a request to get two query rows against one block table row
    req_of_row = [0, 1, 2, 3, 4, 2]
    sel.append(_sel(70, 20, width, 99))
    q = _q(len(req_of_row), 3)
    out = mm.sparse_attend_rows(
        q.to(device), cache, bt[req_of_row], torch.tensor(sel, dtype=torch.int32,
                                                         device=device),
        BS, SCALE, DK,
    )
    assert out.shape == (len(req_of_row), H, DK) and out.dtype == q.dtype
    for i, r in enumerate(req_of_row):
        pos = [p for p in sel[i] if p >= 0]
        ref = _ref_attend(q[i], rows[r], pos, SCALE)
        torch.testing.assert_close(out[i].cpu().double(), ref, atol=2e-2, rtol=2e-2)
    assert torch.equal(out[3].cpu(), torch.zeros(H, DK, dtype=out.dtype))


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_dense_equals_sparse_when_everything_selected(device, monkeypatch):
    monkeypatch.setattr(mm, "_DENSE_ROW_CHUNK", 7)
    monkeypatch.setattr(mm, "_SPARSE_ROW_CHUNK", 5)
    S, Q = 53, 20  # 33 cached, 20 new query rows
    rows = _rows(S, 21)
    cache, bt = _paged(device, [rows])
    q = _q(Q, 22)
    nblk = (S + BS - 1) // BS
    dense = mm.dense_causal_attend(q.to(device), cache, bt[0, :nblk], S, BS, SCALE, DK)
    width = 64
    sel = []
    for i in range(Q):
        vis = S - Q + i + 1
        sel.append(list(range(vis)) + [-1] * (width - vis))
    sparse = mm.sparse_attend_rows(
        q.to(device), cache, bt[[0] * Q],
        torch.tensor(sel, dtype=torch.int32, device=device), BS, SCALE, DK,
    )
    torch.testing.assert_close(dense.cpu().float(), sparse.cpu().float(),
                               atol=1e-2, rtol=1e-2)
    for i in range(Q):
        ref = _ref_attend(q[i], rows, list(range(S - Q + i + 1)), SCALE)
        torch.testing.assert_close(dense[i].cpu().double(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_insert_latent_rows(device):
    backing = torch.full((4, 3, BS, DK), 2.0, dtype=torch.bfloat16)
    cache = backing.to(device)[:, 2]
    latent = _rows(5, 5).to(device)
    slots = torch.tensor([3 * BS + 1, -1, 2 * BS + 15, 1 * BS, -1],
                         dtype=torch.int64, device=device)
    mm.insert_latent_rows(cache, latent, slots)
    full = cache.cpu()
    assert torch.equal(full[3, 1], latent[0].cpu())
    assert torch.equal(full[2, 15], latent[2].cpu())
    assert torch.equal(full[1, 0], latent[3].cpu())
    touched = {(3, 1), (2, 15), (1, 0), (0, 0)}  # (0, 0): null block for PADs
    for b in range(4):
        for t in range(BS):
            if (b, t) not in touched:
                assert (full[b, t] == 2.0).all(), (b, t)
    assert (backing[:, 0] == 2.0).all() and (backing[:, 1] == 2.0).all()


def _vllm_config(index_topk, kpool, num_spec):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_topk=index_topk, index_kpool=kpool),
            dtype=torch.bfloat16,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=256, async_scheduling=False
        ),
        speculative_config=(
            None if num_spec == 0
            else SimpleNamespace(
                num_speculative_tokens=num_spec, parallel_drafting=False
            )
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_builder_and_impl_mixed_batch(device):
    """decode (ctx 37 -> +1) | spec verify (ctx 20 -> +3, threshold 3 so it
    is a decode) | short prefill with cached prefix (ctx 30 -> +10, dense)
    | fresh short prefill (ctx 12 -> +12, dense) | long prefill with cached
    prefix (ctx 40 -> +5, above the limit -> gathered rows)."""
    index_topk, kpool = 16, 4
    limit = index_topk + kpool - 1  # 19
    width = (index_topk + kpool - 1 + 31) // 32 * 32  # 32
    ctx = [37, 20, 30, 12, 40]
    new = [1, 3, 10, 12, 5]
    rows_all = [_rows(c, 200 + i) for i, c in enumerate(ctx)]
    # pages sized for the full context; the cached prefix is pre-written,
    # the new rows are NaN until do_kv_cache_update inserts them.
    cache, bt = _paged(device, rows_all, seed=8)
    for r in range(5):
        for t in range(ctx[r] - new[r], ctx[r]):
            cache[int(bt[r, t // BS]), t % BS] = float("nan")
    T = sum(new)
    qsl = [0]
    for n in new:
        qsl.append(qsl[-1] + n)
    slots = []
    for r in range(5):
        for t in range(ctx[r] - new[r], ctx[r]):
            slots.append(int(bt[r, t // BS]) * BS + t % BS)
    vllm_config = _vllm_config(index_topk, kpool, num_spec=2)
    spec = MLAAttentionSpec(block_size=BS, num_kv_heads=1, head_size=DK,
                            dtype=torch.bfloat16)
    builder = mm.MetalMLASparseMetadataBuilder(spec, ["l"], vllm_config,
                                               torch.device(device))
    assert builder.reorder_batch_threshold == 3
    assert builder.selected_limit == limit
    cam = CommonAttentionMetadata(
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        seq_lens=torch.tensor(ctx, dtype=torch.int32, device=device),
        num_reqs=5,
        num_actual_tokens=T,
        max_query_len=max(new),
        max_seq_len=max(ctx),
        block_table_tensor=bt,
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
        seq_lens_cpu_upper_bound=torch.tensor(ctx, dtype=torch.int32),
    )
    md = builder.build(0, cam)
    assert (md.num_decodes, md.num_prefills, md.num_decode_tokens) == (2, 3, 4)
    assert md.prefill_max_seq_len == 40 and md.topk_tokens == index_topk
    assert md.bt_per_token.shape == (T, bt.shape[1])
    assert torch.equal(md.bt_per_token[3].cpu(), bt[1].cpu())
    assert torch.equal(md.bt_per_token[T - 1].cpu(), bt[4].cpu())

    # per-row selections: everything for rows within the limit (what the
    # pooled indexer produces there); a random 16-subset for the long request.
    sel = []
    for r in range(5):
        for j in range(new[r]):
            vis = ctx[r] - new[r] + j + 1
            if vis <= limit:
                sel.append(list(range(vis)) + [-1] * (width - vis))
            else:
                sel.append(_sel(vis, index_topk, width, 300 + len(sel)))
    topk_buf = torch.full((T + 2, width), -1, dtype=torch.int32, device=device)
    topk_buf[:T] = torch.tensor(sel, dtype=torch.int32, device=device)

    impl = mm.MetalMLASparseImpl(
        num_heads=H, head_size=DK, scale=SCALE, num_kv_heads=1, alibi_slopes=None,
        sliding_window=None, kv_cache_dtype="auto", logits_soft_cap=None,
        attn_type="decoder", kv_sharing_target_layer_name=None,
        topk_indices_buffer=topk_buf, indexer=None,
        q_lora_rank=32, kv_lora_rank=DK, qk_nope_head_dim=DK, qk_rope_head_dim=0,
        qk_head_dim=DK, v_head_dim=DK, kv_b_proj=None,
    )
    latent = torch.cat([rows_all[r][ctx[r] - new[r] :] for r in range(5)]).to(device)
    k_pe = torch.empty((T, 1, 0), dtype=torch.bfloat16, device=device)
    impl.do_kv_cache_update(latent, k_pe, cache, md.slot_mapping, "auto", None)
    for r in range(5):
        for t in range(ctx[r] - new[r], ctx[r]):
            got = cache[int(bt[r, t // BS]), t % BS].cpu()
            assert torch.equal(got, rows_all[r][t]), (r, t)
    q = _q(T + 1, 44)  # one padded row beyond num_actual_tokens
    q_pe = torch.empty((T + 1, H, 0), dtype=torch.bfloat16, device=device)
    out, lse = impl.forward_mqa((q.to(device), q_pe), cache, md)
    assert lse is None and out.shape == (T, H, DK) and out.is_contiguous()
    for r in range(5):
        for j in range(new[r]):
            i = qsl[r] + j
            pos = [p for p in sel[i] if p >= 0]
            ref = _ref_attend(q[i], rows_all[r], pos, SCALE)
            torch.testing.assert_close(out[i].cpu().double(), ref, atol=2e-2,
                                       rtol=2e-2, msg=f"req {r} row {j}")


def test_backend_surface():
    b = mm.MetalMLASparseBackend
    assert b.is_mla() and b.is_sparse()
    assert b.get_kv_cache_shape(7, BS, 1, 512) == (7, BS, 512)
    assert b.get_kv_cache_block_dim(BS, 1, 512) == 0
    assert b.supports_kv_cache_dtype("auto") and b.supports_kv_cache_dtype("bfloat16")
    assert not b.supports_kv_cache_dtype("fp8_e4m3")
    with pytest.raises(NotImplementedError):
        mm.MetalMLASparseImpl(
            num_heads=H, head_size=DK, scale=SCALE, num_kv_heads=1,
            alibi_slopes=None, sliding_window=None, kv_cache_dtype="fp8",
            logits_soft_cap=None, attn_type="decoder",
            kv_sharing_target_layer_name=None, kv_lora_rank=DK,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
