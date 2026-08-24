# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metal (MPS) long-context producer for the DeepSeek-V4 sparse indexer.

On CUDA, SparseAttnIndexer runs fp8 logits kernels + top_k_per_row_* to fill
``topk_indices_buffer`` once a request's compressed candidate count exceeds
``index_topk``. None of those kernels exist on Metal, so this module computes
the same selection in torch ops, sync-free by construction (fixed shapes from
python-int metadata; no ``.item()``, ``nonzero`` or boolean compaction):

    logits[t, j] = k_scale[j] * sum_h w[t, h] * relu(q[t, h, :] . code[j, :])
    topk_indices[t] = topk(logits[t], index_topk)   # request-local, -1 pad

This matches ``fp8_mqa_logits_torch`` / ``fp8_paged_mqa_logits_torch`` (the
DeepGEMM reference oracles) under the Metal weight-fold convention from
``_fused_indexer_q_rope_quant_metal``: q arrives as value / q_scale in model
dtype (fp8 rounding skipped), with q_scale folded into ``weights``.

The indexer K cache on Metal is written by ``dsv4_indexer_kv_insert`` as
per-slot 132-byte records: [128 e4m3 codes][1 float32 power-of-two scale].
The e4m3 bytes are decoded here through a 256-entry fp32 LUT (MPS has no fp8
casts); decode-by-LUT is exact.
"""

import os
from typing import Any, cast

import torch

from vllm.forward_context import get_forward_context

# Long-context prefill mitigation (default on): MPSGraph's encode queue
# degrades pathologically when >64K-token prefill einsums run with
# unbounded encode-ahead. One bounded synchronize per producer call keeps
# the queue shallow; timing-only, bit-exact. VLLM_QC_LONGCTX_SYNC=0
# disables. Retire via a real MPSGraph encode-queue fix (see
# optimization_status).
_LONGCTX_PREFILL_SYNC = os.environ.get("VLLM_QC_LONGCTX_SYNC", "1") != "0"

# Rows and candidates per fallback score tile: together they bound the fp32
# intermediate to 32 MiB instead of scaling it to the configured context
# length (which would be about 2 GiB at 128 rows x 64 heads x 65,536 keys).
_SCORE_ROW_CHUNK = 128
_SCORE_K_CHUNK = 1024

_E4M3_LUT: dict[str, torch.Tensor] = {}


def _e4m3_lut(device: torch.device) -> torch.Tensor:
    key = str(device)
    lut = _E4M3_LUT.get(key)
    if lut is None:
        lut = (
            torch.arange(256, dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .to(torch.float32)
        )
        # 0x7f/0xff decode to NaN; the writer never emits them, but a NaN
        # from a stale slot would poison top-k, so decode them as 0.
        lut = torch.nan_to_num(lut, nan=0.0).to(device)
        _E4M3_LUT[key] = lut
    return lut


def _topk_desc_stable(
    logits: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k with the native kernels' documented (logit desc, index asc)
    tie order. torch.topk's tie order is unspecified and empirically
    differs between backends and between the full and streaming eager
    paths; ties are reachable in production (relu zeroes, stale e4m3
    slots decode to 0), so the fallbacks must break them the same way
    the kernel does."""
    vals, idx = torch.sort(logits, dim=-1, descending=True, stable=True)
    return vals[..., :k], idx[..., :k]


def _score_and_select(
    q: torch.Tensor,  # [rows, H, 128] model dtype (value / q_scale)
    weights: torch.Tensor,  # [rows, H] fp32 (q_scale folded)
    k_vals: torch.Tensor,  # [n_k, 128] or [rows, n_k, 128] fp32 codes
    k_scale: torch.Tensor,  # [n_k] or [rows, n_k] fp32
    lo: torch.Tensor | None,  # [rows] first valid column, or None for 0
    hi: torch.Tensor,  # [rows] one past last valid column
    k_eff: int,
) -> torch.Tensor:
    """relu-weighted MQA logits + top-k. Returns [rows, k_eff] int64 column
    indices into the K axis, -1 where fewer than k_eff candidates exist."""
    n_k = k_vals.shape[-2]
    col = torch.arange(n_k, device=q.device)
    if k_vals.dim() == 2:
        scores = torch.einsum("thd,kd->thk", q.float(), k_vals)
    else:
        scores = torch.einsum("thd,tkd->thk", q.float(), k_vals)
    logits = torch.einsum("thk,th->tk", torch.relu(scores), weights)
    logits = logits * k_scale if k_scale.dim() == 2 else logits * k_scale[None, :]
    invalid = col[None, :] >= hi[:, None]
    if lo is not None:
        invalid |= col[None, :] < lo[:, None]
    logits = logits.masked_fill(invalid, float("-inf"))
    vals, idx = _topk_desc_stable(logits, k_eff)
    return idx.masked_fill(vals == float("-inf"), -1)


def _score_and_select_streaming(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_vals: torch.Tensor,
    k_scale: torch.Tensor,
    lo: torch.Tensor | None,
    hi: torch.Tensor,
    k_eff: int,
) -> torch.Tensor:
    """Candidate-tiled fallback with O(rows * heads * 1024) score memory."""
    rows = q.shape[0]
    n_k = k_vals.shape[-2]
    best_vals = torch.full(
        (rows, k_eff), float("-inf"), dtype=torch.float32, device=q.device
    )
    best_idx = torch.full((rows, k_eff), -1, dtype=torch.long, device=q.device)

    for k0 in range(0, n_k, _SCORE_K_CHUNK):
        k1 = min(k0 + _SCORE_K_CHUNK, n_k)
        tile_vals = k_vals[k0:k1]
        tile_scale = k_scale[k0:k1]
        col = torch.arange(k0, k1, device=q.device)
        scores = torch.einsum("thd,kd->thk", q.float(), tile_vals)
        logits = torch.einsum("thk,th->tk", torch.relu(scores), weights)
        logits *= tile_scale[None, :]
        invalid = col[None, :] >= hi[:, None]
        if lo is not None:
            invalid |= col[None, :] < lo[:, None]
        logits.masked_fill_(invalid, float("-inf"))

        tile_k = min(k_eff, k1 - k0)
        vals, idx = _topk_desc_stable(logits, tile_k)
        idx += k0
        # Retained survivors precede the new tile positionally and always
        # carry smaller global indices (tiles ascend), so the stable merge
        # preserves (logit desc, index asc) globally by induction.
        merged_vals = torch.cat((best_vals, vals), dim=1)
        merged_idx = torch.cat((best_idx, idx), dim=1)
        best_vals, keep = _topk_desc_stable(merged_vals, k_eff)
        best_idx = merged_idx.gather(1, keep)

    return best_idx.masked_fill(best_vals == float("-inf"), -1)


def _gather_k(
    kv_cache: torch.Tensor,  # [num_blocks, block_size, 132] uint8 (may be
    # block-stride padded for alignment — index, never view(-1))
    blocks: torch.Tensor,  # [...] int64 block ids (all >= 0)
    offs: torch.Tensor,  # [...] int64 in-block offsets, broadcastable
    lut: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = kv_cache[blocks, offs]  # [..., 132] contiguous gather
    k_vals = lut[rows[..., :128].to(torch.long)]
    k_scale = rows[..., 128:].contiguous().view(torch.float32).reshape(rows.shape[:-1])
    return k_vals, k_scale


def _native_topk_decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    dec,
    buf: torch.Tensor,
    num_decode: int,
    width: int,
    k_eff: int,
) -> bool:
    """Single-dispatch decode producer (logits + masked top-k straight into
    ``buf``); mirrors the eager branch below to reduction-order ulps, with a
    deterministic (logit desc, index asc) tie order and the exact e4m3-LUT
    decode. Returns False when shapes/dtypes fall outside the kernel."""
    if not (
        q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[1] == 64
        and q.shape[2] == 128
        and kv_cache.shape[-1] == 132
        and dec.block_table.dtype == torch.int32
        and dec.seq_lens.dtype == torch.int32
        and buf.dtype == torch.int32
        and buf.stride(1) == 1
    ):
        return False
    if width > 1024 and k_eff > 512:
        # The streaming kernel keeps at most 512 survivors between tiles;
        # the native op TORCH_CHECKs this and would raise out of the forward
        # pass. Checkpoints with index_topk > 512 fall back to eager.
        return False
    from vllm.quixicore.ops import quixicore_ops

    if not (
        quixicore_ops.is_available() and quixicore_ops.has("dsv4_indexer_topk_decode")
    ):
        return False
    quixicore_ops.dsv4_indexer_topk_decode(
        q[:num_decode].contiguous(),
        weights[:num_decode].float().contiguous(),
        kv_cache,
        dec.block_table[:num_decode].contiguous(),
        dec.seq_lens.reshape(-1)[:num_decode].contiguous(),
        buf,
        width,
        k_eff,
    )
    return True


def _native_topk_prefill(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    chunk,
    buf: torch.Tensor,
    t0: int,
    n_q: int,
    topk: int,
) -> bool:
    """Single-dispatch prefill producer: the decode top-k kernel with the
    block-table row taken from each token's request and request-local
    candidate windows. DSV4-Flash indexer windows are full causal prefixes
    (cu_seqlen_ks[t] == cu_seq_lens[req(t)]), so the request-local columns,
    logits, deterministic tie order, and rebased output indices mirror the
    eager chain below to the decode kernel's ULP class. Returns False when
    shapes/dtypes fall outside the kernel."""
    if not (
        q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[1] == 64
        and q.shape[2] == 128
        and kv_cache.shape[-1] == 132
        and chunk.block_table.dtype == torch.int32
        and buf.dtype == torch.int32
        and buf.stride(1) == 1
    ):
        return False
    from vllm.quixicore.ops import quixicore_ops

    if not (
        quixicore_ops.is_available() and quixicore_ops.has("dsv4_indexer_topk_prefill")
    ):
        return False
    # This is exact CPU scheduler metadata for prefill requests.  Do not gate
    # on the profile's configured maximum: that disabled the native path for
    # every request when a 262K context was configured.
    max_cand = chunk.max_seq_len
    ks = chunk.cu_seqlen_ks[:n_q]
    ke = chunk.cu_seqlen_ke[:n_q]
    cu = chunk.cu_seq_lens
    tok_req = (torch.searchsorted(cu, ks, right=True).to(torch.int32) - 1).contiguous()
    cand = (ke - ks).to(torch.int32).contiguous()
    k_eff = min(topk, max_cand)
    if max_cand > 1024 and k_eff > 512:
        # Same capacity contract as the decode wrapper: the streaming merge
        # holds 512 survivors, so wider top-k on a >1024 window must take
        # the eager fallback instead of tripping the op's TORCH_CHECK.
        return False
    quixicore_ops.dsv4_indexer_topk_prefill(
        q[t0 : t0 + n_q].contiguous(),
        weights[t0 : t0 + n_q].float().contiguous(),
        kv_cache,
        chunk.block_table.contiguous(),
        tok_req,
        cand,
        buf.narrow(0, t0, n_q),
        max_cand,
        k_eff,
    )
    return True


def metal_sparse_attn_indexer(
    indexer,  # DeepseekV4Indexer
    q: torch.Tensor,  # [T, H, 128]
    weights: torch.Tensor,  # [T, H] fp32
) -> torch.Tensor:
    buf = indexer.topk_indices_buffer
    assert buf is not None
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return buf
    md = cast(Any, attn_metadata[indexer.k_cache.prefix])
    kv_cache = indexer.k_cache.kv_cache
    block_size = kv_cache.shape[1]
    lut = _e4m3_lut(q.device)
    topk = indexer.topk_tokens
    ratio = indexer.compress_ratio

    buf[: q.shape[0]] = -1

    num_decode = md.num_decode_tokens
    if num_decode > 0 and md.decode is not None:
        dec = md.decode
        width = max(1, md.max_seq_len // ratio)
        k_eff = min(topk, width)
        native = _native_topk_decode(
            q, weights, kv_cache, dec, buf, num_decode, width, k_eff
        )
        if not native:
            # The Metal builder path expands decode metadata per token
            # (block_table row and compressed candidate count per decode
            # token).
            cand = dec.seq_lens.reshape(-1)[:num_decode].to(torch.long)
            bt = dec.block_table[:num_decode].to(torch.long)
            j = torch.arange(width, device=q.device)
            cols = torch.div(j, block_size, rounding_mode="floor").clamp(
                max=max(bt.shape[1] - 1, 0)
            )
            blocks = bt.gather(1, cols.unsqueeze(0).expand(num_decode, width))
            offs = torch.remainder(j, block_size).unsqueeze(0)
            k_vals, k_scale = _gather_k(kv_cache, blocks, offs, lut)
            idx = _score_and_select(
                q[:num_decode],
                weights[:num_decode],
                k_vals,
                k_scale,
                None,
                cand,
                k_eff,
            )
            buf[:num_decode, :k_eff] = idx.to(buf.dtype)

    if md.num_prefill_tokens > 0 and md.prefill is not None:
        for chunk in md.prefill.chunks:
            t0, t1 = chunk.token_start, chunk.token_end
            n_q = t1 - t0
            n_k = chunk.total_seq_lens
            if n_q <= 0 or n_k <= 0:
                continue
            if _native_topk_prefill(
                q,
                weights,
                kv_cache,
                chunk,
                buf,
                t0,
                n_q,
                topk,
            ):
                continue
            # Concatenated per-request K rows, exactly the reference layout:
            # rows [cu[r], cu[r+1]) belong to chunk-relative request r.
            req_of_row = chunk.token_to_seq[:n_k].to(torch.long)
            cu = chunk.cu_seq_lens.to(torch.long)
            local_j = torch.arange(n_k, device=q.device) - cu.index_select(
                0, req_of_row
            )
            bt = chunk.block_table.to(torch.long)
            cols = torch.div(local_j, block_size, rounding_mode="floor").clamp(
                min=0, max=max(bt.shape[1] - 1, 0)
            )
            k_vals, k_scale = _gather_k(
                kv_cache,
                bt[req_of_row, cols],
                torch.remainder(local_j, block_size),
                lut,
            )
            ks = chunk.cu_seqlen_ks[:n_q].to(torch.long)
            ke = chunk.cu_seqlen_ke[:n_q].to(torch.long)
            k_eff = min(topk, n_k)
            for s in range(0, n_q, _SCORE_ROW_CHUNK):
                e = min(s + _SCORE_ROW_CHUNK, n_q)
                idx = _score_and_select_streaming(
                    q[t0 + s : t0 + e],
                    weights[t0 + s : t0 + e],
                    k_vals,
                    k_scale,
                    ks[s:e],
                    ke[s:e],
                    k_eff,
                )
                # Rebase concatenated columns to request-local candidate
                # positions; -1 pads must stay -1.
                idx = torch.where(idx >= 0, idx - ks[s:e, None], idx)
                buf[t0 + s : t0 + e, :k_eff] = idx.to(buf.dtype)
        if _LONGCTX_PREFILL_SYNC:
            torch.mps.synchronize()

    return buf
