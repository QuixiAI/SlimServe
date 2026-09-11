# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooled DSA indexer for GLM-5.3-Flash (glm5_next).

Reference: transformers ``Glm5NextTextIndexer``. Unlike the DeepSeek-V3.2
per-token indexer, this one scores COMPRESSED POOLS of ``index_kpool`` (4)
consecutive tokens:

    pool_key(p) = sum_m softmax_m(gate(t_m) + ape[m]) * k(t_m)
    logit(q, p) = sum_h w_h * relu(scale * q_h . pool_key(p))

selects ``index_topk // index_kpool`` pools per query, expands each pool
back into its token indices, and always appends the current incomplete
tail pool's raw tokens (``index_kpool_always_select_tail``). Only complete
pools whose last token is visible to the query are candidates.

Serving design (no eager paths): the per-token indexer state
``[k_norm(wk x) | gate = x @ compress_gate^T]`` (256 bf16, 512 B/token) is
a paged KV-cache group registered like the DeepSeek-V3.2 indexer cache;
pooled logits are computed by a paged Triton kernel that reads the four
member rows of each pool straight from the cache (block table), so prefill
and decode share one path and nothing is gathered or materialized per
token; top-k runs on the existing per-row top-k kernels in POOL units;
a second Triton kernel expands pools to tokens and appends the tail into
``topk_indices_buffer``. The whole forward is one custom op so it stays
opaque to torch.compile and captures into decode CUDA graphs.
"""

from __future__ import annotations

import os
from functools import cache as cache_once

import torch
from torch import nn

from slimserve.canonical_indexer import maybe_ordered_topk
from slimserve.index_journal import instrument_pooled_indexer, instrument_topk

from vllm import _custom_ops as ops
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

logger = init_logger(__name__)

# Keep the observer outside the intervention: record the actual selection order
# passed to pool expansion. Both factories are identity functions when disabled.
top_k_per_row_prefill = instrument_topk(maybe_ordered_topk(ops.top_k_per_row_prefill))

# Row layout of the cached indexer state.
_K_DIM = 128
_ROW_DIM = 2 * _K_DIM  # [k | gate]
_POOL_PROGRAMS = 128  # programs per row on the pool axis (stride loop inside)
# Prefill path: pooled keys once per (request, pool), then a tensor-core
# matmul of [_ROW_TILE * H, D] query rows against [_POOL_TILE, D] pooled keys.
# Kill switch for A/B runs only (read here, so it is not a compile factor):
# VLLM_GLM5_INDEXER_PREFILL_MATMUL=0 scores prefill rows with the decode-shaped
# per-row kernel instead.
_PREFILL_MATMUL = os.getenv("VLLM_GLM5_INDEXER_PREFILL_MATMUL", "1") != "0"
# Profile-selected SM120 geometry: fewer query rows and more pooled keys avoid the
# original tile's register spills. Arithmetic and selection are unchanged.
# Keep other profiles on their measured geometry; set before worker import.
_SM120_TILES = os.getenv("VLLM_GLM5_INDEXER_SM120_TILES", "0") == "1"
_ROW_TILE = 2 if _SM120_TILES else 8
_POOL_TILE = 128 if _SM120_TILES else 64
_TP_PREFILL_SHARD = os.getenv("VLLM_GLM53_INDEXER_TP_PREFILL", "0") == "1"


def _prefill_row_sharding_enabled(chunks) -> bool:
    # This fork stores host-side context lengths on each chunk, not on the
    # enclosing prefill metadata (unlike the newer upstream implementation).
    return bool(
        _TP_PREFILL_SHARD and _PREFILL_MATMUL and chunks
        and chunks[-1].token_end - chunks[0].token_start >= 2048
        and max(c.max_seq_len for c in chunks) >= 32768
    )


@cache_once
def _prefill_shard_group():
    # Explicit SM120/TP4 opt-in. Decode, short prefills and other platforms
    # retain the existing zero-communication path.
    if (not _TP_PREFILL_SHARD or not torch.cuda.is_available()
            or torch.cuda.get_device_capability() != (12, 0)):
        return None
    group = get_tp_group()
    if (group.world_size != 4 or get_dcp_group().world_size != 1
            or get_pcp_group().world_size != 1):
        return None
    return group


class Glm5NextIndexerBackend(DeepseekV32IndexerBackend):
    """DSV3.2 indexer metadata (slot mapping, prefill chunks, decode block
    tables) over a 256-wide bf16 row instead of the fp8 128+scale row."""

    @staticmethod
    def get_name() -> str:
        return "GLM5_NEXT_INDEXER"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_ROW_DIM]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Runtime block size everywhere (insert rows via slot mapping, pooled
        # logits via block table // BLOCK_SIZE); the kernel block equals the
        # KV-manager block so the packed slab view is a single stride.
        return [MultipleOf(64)]


class Glm5NextIndexerCache(DeepseekV32IndexerCache):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Glm5NextIndexerBackend


# --------------------------------------------------------------------- kernels


@triton.jit
def _insert_rows_kernel(
    src_ptr,
    cache_ptr,
    slot_ptr,
    num_tokens,
    page_stride,    # elements between consecutive blocks' pages
    BLOCK_SIZE: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """cache[slot[t]] = src[t] for slot[t] >= 0. Pages may be strided (the
    packed cross-layer slab interleaves every layer's page per block), so a
    slot resolves to block * page_stride + (slot % BLOCK_SIZE) * ROW."""
    pid = tl.program_id(0)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < num_tokens
    slot = tl.load(slot_ptr + t, mask=tmask, other=-1)
    valid = tmask & (slot >= 0)
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = slot - blk * BLOCK_SIZE
    cols = tl.arange(0, ROW)
    vals = tl.load(
        src_ptr + t[:, None] * ROW + cols[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    tl.store(
        cache_ptr + (blk * page_stride + off * ROW)[:, None] + cols[None, :],
        vals,
        mask=valid[:, None],
    )


@triton.jit
def _pooled_logits_kernel(
    q_ptr,          # [R, H, D] bf16
    w_ptr,          # [R, H] fp32 (already * n_heads^-0.5)
    ape_ptr,        # [KP, D] fp32
    cache_ptr,      # [num_slots, ROW] bf16
    bt_ptr,         # [num_bt_rows, bt_stride] int32
    row_req_ptr,    # [R] int32: block-table row per query row
    vis_ptr,        # [R] int32: visible tokens per query row
    out_ptr,        # [R, max_pools] fp32
    max_pools,
    bt_stride,
    page_stride,    # elements between consecutive blocks' pages
    softmax_scale,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    KP: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Pooled indexer logits for one query row. The grid's second axis is a
    FIXED number of programs (CUDA-graph safe); each strides over the
    row's pool tiles up to its actual visible count, so the work is
    proportional to the context, not to max_model_len. Sizing the grid by
    max_model_len // KP cost 16,384 programs per row per layer at the
    1M-token default, each reloading q and APE and storing -inf for pools
    the top-k never reads: 20 ms per sequence per decode step."""
    r = tl.program_id(0)
    pb = tl.program_id(1)
    G = tl.num_programs(1)
    vis = tl.load(vis_ptr + r)
    n_pools = vis // KP
    n_tiles = (n_pools + BLOCK_P - 1) // BLOCK_P
    req = tl.load(row_req_ptr + r)

    m = tl.arange(0, KP)  # [KP]
    d = tl.arange(0, D)
    h = tl.arange(0, H)
    ape = tl.load(ape_ptr + m[:, None] * D + d[None, :])  # [KP, D]
    q = tl.load(q_ptr + (r.to(tl.int64) * H + h[:, None]) * D + d[None, :])
    qb = q.to(tl.bfloat16)  # [H, D]
    w = tl.load(w_ptr + r * H + h)  # [H]

    for tile in range(pb, n_tiles, G):
        p = tile * BLOCK_P + tl.arange(0, BLOCK_P)  # [P]
        pmask = p < n_pools
        tok = p[:, None] * KP + m[None, :]  # [P, KP]
        blk = tl.load(
            bt_ptr + req.to(tl.int64) * bt_stride + tok // BLOCK_SIZE,
            mask=pmask[:, None],
            other=0,
        )
        base = cache_ptr + (
            blk.to(tl.int64) * page_stride + (tok % BLOCK_SIZE) * ROW
        )[:, :, None]  # [P, KP, 1]
        k = tl.load(base + d[None, None, :], mask=pmask[:, None, None], other=0.0)
        g = tl.load(
            base + D + d[None, None, :], mask=pmask[:, None, None], other=0.0
        )
        logits_g = g.to(tl.float32) + ape[None, :, :]  # [P, KP, D]
        # softmax over the pool members (axis 1), per channel
        mx = tl.max(logits_g, axis=1)  # [P, D]
        e = tl.exp(logits_g - mx[:, None, :])
        probs = e / tl.sum(e, axis=1)[:, None, :]
        pool_key = tl.sum(probs * k.to(tl.float32), axis=1)  # [P, D]
        # scores[P, H] = relu(scale * pool_key . q_h)
        scores = tl.dot(pool_key.to(tl.bfloat16), tl.trans(qb))
        scores = tl.maximum(scores * softmax_scale, 0.0)
        logit = tl.sum(scores * w[None, :], axis=1)  # [P]
        logit = tl.where(pmask, logit, float("-inf"))
        omask = p < max_pools
        tl.store(out_ptr + r.to(tl.int64) * max_pools + p, logit, mask=omask)


@triton.jit
def _expand_topk_kernel(
    sel_ptr,        # [R, KSEL] int32 pool indices (-1 invalid), valid first
    vis_ptr,        # [R] int32
    out_ptr,        # [R, OUT_W] int32
    sel_stride,
    KP: tl.constexpr,
    KSEL: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Expand selected pools to tokens, then append the incomplete tail
    pool's tokens IMMEDIATELY after the last valid expanded entry, then
    -1 padding. The sparse decode kernel walks [0, last_valid + 1), so
    valid entries must stay contiguous: a tail parked at column
    KSEL*KP would force a full 2048-slot walk for every query."""
    r = tl.program_id(0)
    vis = tl.load(vis_ptr + r)
    n_pools = vis // KP
    tail_count = vis - n_pools * KP
    tail_start = n_pools * KP
    n_sel = tl.minimum(n_pools, KSEL)  # top-k writes valid pools first
    # Only the selected slots are written here; the tail store and the
    # padding loop below own the columns from n_sel * KP on. Threads of one
    # program are not ordered against each other, so a slot written by two
    # stores keeps whichever lands last: writing -1 over [0, KSEL * KP) here
    # raced the tail store and dropped the tail (often the query's own
    # token) from a fraction of rows.
    for s0 in range(0, KSEL, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        smask = s < n_sel
        pool = tl.load(sel_ptr + r.to(tl.int64) * sel_stride + s, mask=smask, other=-1)
        ok = smask & (pool >= 0) & (pool < n_pools)
        m = tl.arange(0, KP)
        tokens = tl.where(ok[:, None], pool[:, None] * KP + m[None, :], -1)
        tl.store(
            out_ptr + r.to(tl.int64) * OUT_W + s[:, None] * KP + m[None, :],
            tokens,
            mask=smask[:, None],
        )
    # tail right after the valid pools, then pad the rest of the row
    tcol = n_sel * KP + tl.arange(0, KP)
    tail = tl.arange(0, KP)
    tval = tl.where(tail < tail_count, tail_start + tail, -1)
    tl.store(out_ptr + r.to(tl.int64) * OUT_W + tcol, tval, mask=tcol < OUT_W)
    pad0 = n_sel * KP + KP
    for c0 in range(0, OUT_W, 256):
        c = c0 + tl.arange(0, 256)
        cm = (c >= pad0) & (c < OUT_W)
        tl.store(out_ptr + r.to(tl.int64) * OUT_W + c, tl.full((256,), -1, tl.int32), mask=cm)


@triton.jit(do_not_specialize=["max_pools"])
def _pool_keys_kernel(
    ape_ptr,        # [KP, D] fp32
    cache_ptr,      # [num_slots, ROW] bf16
    bt_ptr,         # [num_bt_rows, bt_stride] int32
    req_pools_ptr,  # [num_bt_rows] int32: pools to build per block-table row
    pk_ptr,         # [num_bt_rows, max_pools, D] bf16
    max_pools,
    bt_stride,
    page_stride,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    KP: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """pk[req, p, :] = sum_m softmax_m(gate(t_m) + ape[m]) * k(t_m) over the
    KP members of pool p of the request in block-table row req. The pooled
    key depends only on the cache and APE, never on the query, so the
    prefill path builds it once per request instead of once per query row
    (`_pooled_logits_kernel` re-pools it for every row: rows x visible
    softmaxes gathered through the block table). Same arithmetic and the
    same bf16 rounding as that kernel's pool_key."""
    req = tl.program_id(0)
    pt = tl.program_id(1)
    n_pools = tl.load(req_pools_ptr + req)
    p = pt * BLOCK_P + tl.arange(0, BLOCK_P)
    pmask = p < n_pools
    m = tl.arange(0, KP)
    d = tl.arange(0, D)
    ape = tl.load(ape_ptr + m[:, None] * D + d[None, :])  # [KP, D]
    tok = p[:, None] * KP + m[None, :]  # [P, KP]
    blk = tl.load(
        bt_ptr + req.to(tl.int64) * bt_stride + tok // BLOCK_SIZE,
        mask=pmask[:, None],
        other=0,
    )
    base = cache_ptr + (
        blk.to(tl.int64) * page_stride + (tok % BLOCK_SIZE) * ROW
    )[:, :, None]  # [P, KP, 1]
    k = tl.load(base + d[None, None, :], mask=pmask[:, None, None], other=0.0)
    g = tl.load(base + D + d[None, None, :], mask=pmask[:, None, None], other=0.0)
    logits_g = g.to(tl.float32) + ape[None, :, :]  # [P, KP, D]
    mx = tl.max(logits_g, axis=1)  # [P, D]
    e = tl.exp(logits_g - mx[:, None, :])
    probs = e / tl.sum(e, axis=1)[:, None, :]
    pool_key = tl.sum(probs * k.to(tl.float32), axis=1)  # [P, D]
    tl.store(
        pk_ptr + (req.to(tl.int64) * max_pools + p)[:, None] * D + d[None, :],
        pool_key.to(tl.bfloat16),
        mask=pmask[:, None],
    )


@triton.jit
def _row_tiles_kernel(
    start_ptr,      # [num_req] int32: first query row of the request
    count_ptr,      # [num_req] int32: query rows of the request
    first_ptr,      # [num_req] int32: first tile slot of the request
    row0_ptr,       # [max_tiles] int32 out
    end_ptr,        # [max_tiles] int32 out
    req_ptr,        # [max_tiles] int32 out
    RT: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Tiles of RT consecutive query rows of one request (the last tile of a
    request is short), written at the request's slots of the tile table.
    Sized on the host from an upper bound, so no sync: unused slots keep
    their (0, 0) fill and the matmul programs that draw them exit."""
    req = tl.program_id(0)
    start = tl.load(start_ptr + req)
    count = tl.load(count_ptr + req)
    first = tl.load(first_ptr + req)
    n_tiles = (count + RT - 1) // RT
    for t0 in range(0, n_tiles, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        tmask = t < n_tiles
        row0 = start + t * RT
        tl.store(row0_ptr + first + t, row0, mask=tmask)
        tl.store(end_ptr + first + t, tl.minimum(row0 + RT, start + count), mask=tmask)
        req_v = tl.full((BLOCK_T,), 0, tl.int32) + req
        tl.store(req_ptr + first + t, req_v, mask=tmask)


@triton.jit(do_not_specialize=["max_pools"])
def _pooled_logits_matmul_kernel(
    q_ptr,          # [R, H, D] bf16
    w_ptr,          # [R, H] fp32 (already * n_heads^-0.5)
    pk_ptr,         # [num_bt_rows, max_pools, D] bf16 pooled keys
    row0_ptr,       # [max_tiles] int32
    end_ptr,        # [max_tiles] int32
    req_ptr,        # [max_tiles] int32
    vis_ptr,        # [R] int32
    out_ptr,        # [R, max_pools] fp32
    max_pools,
    softmax_scale,
    H: tl.constexpr,
    D: tl.constexpr,
    KP: tl.constexpr,
    RT: tl.constexpr,
    PT: tl.constexpr,
):
    """logits[r, p] = sum_h w[r, h] * relu(scale * <pk[req(r), p], q[r, h]>)
    for p < vis(r) // KP. One program scores a row tile (rows [row0, end)
    of one request) against PT pools as a [RT * H, D] x [D, PT] tensor-core
    product; pool tiles past every row's visibility exit at once."""
    rt = tl.program_id(0)
    pt = tl.program_id(1)
    r0 = tl.load(row0_ptr + rt)
    r_end = tl.load(end_ptr + rt)
    rows = r0 + tl.arange(0, RT)
    rmask = rows < r_end
    vis = tl.load(vis_ptr + rows, mask=rmask, other=0)
    n_pools = vis // KP
    p0 = pt * PT
    if p0 >= tl.max(n_pools, axis=0):
        return
    req = tl.load(req_ptr + rt)
    p = p0 + tl.arange(0, PT)
    d = tl.arange(0, D)
    pk = tl.load(
        pk_ptr + (req.to(tl.int64) * max_pools + p)[:, None] * D + d[None, :],
        mask=(p < max_pools)[:, None],
        other=0.0,
    )  # [PT, D]
    qi = tl.arange(0, RT * H)
    qrow = r0 + qi // H
    qmask = qrow < r_end
    q = tl.load(
        q_ptr + (qrow.to(tl.int64) * H + qi % H)[:, None] * D + d[None, :],
        mask=qmask[:, None],
        other=0.0,
    )  # [RT * H, D]
    scores = tl.dot(q, tl.trans(pk))  # [RT * H, PT] fp32
    scores = tl.maximum(scores * softmax_scale, 0.0)
    w = tl.load(w_ptr + qrow * H + qi % H, mask=qmask, other=0.0)  # [RT * H]
    scores = tl.reshape(scores * w[:, None], (RT, H, PT))
    logit = tl.sum(scores, axis=1)  # [RT, PT]
    valid = (p[None, :] < n_pools[:, None]) & rmask[:, None]
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(
        out_ptr + rows.to(tl.int64)[:, None] * max_pools + p[None, :],
        logit,
        mask=rmask[:, None] & (p[None, :] < max_pools),
    )


# --------------------------------------------------------------------- core op


def _pooled_logits_by_request(
    q: torch.Tensor,
    weights: torch.Tensor,
    ape: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    row_req: torch.Tensor,
    n_pools: torch.Tensor,     # [R] int32: visible // kp per row
    visible: torch.Tensor,
    logits: torch.Tensor,
    max_pools: int,
    block_size: int,
    softmax_scale: float,
    kp: int,
) -> None:
    """Fill `logits` for rows grouped by request (rows of a request are
    contiguous, requests in block-table order, as in a prefill chunk):
    pooled keys once per request, then the tiled matmul. Every size that
    shapes a launch is a host int, so nothing syncs."""
    R, H, D = q.shape
    RT, PT = _ROW_TILE, _POOL_TILE
    n_req = block_table.shape[0]
    dev = q.device
    i32 = torch.int32
    req_ids = torch.arange(n_req, device=dev, dtype=row_req.dtype)
    start = torch.searchsorted(row_req, req_ids).to(i32)
    end = torch.searchsorted(row_req, req_ids, right=True).to(i32)
    count = end - start
    req_pools = torch.zeros(n_req, dtype=i32, device=dev)
    req_pools.scatter_reduce_(0, row_req.long(), n_pools, reduce="amax")
    n_tiles = (count + (RT - 1)) // RT
    tile_first = torch.cumsum(n_tiles, 0, dtype=i32) - n_tiles
    # sum over requests of ceil(count / RT) <= ceil(R / RT) + n_req; padded
    # to 16 slots so the three table rows stay 16-byte aligned (Triton
    # specialises pointer args on alignment: one compile per boot, not one
    # per tile count).
    max_tiles = -(-(triton.cdiv(R, RT) + n_req) // 16) * 16
    tiles = torch.zeros((3, max_tiles), dtype=i32, device=dev)
    _row_tiles_kernel[(n_req,)](
        start, count, tile_first, tiles[0], tiles[1], tiles[2],
        RT=RT, BLOCK_T=256,
    )
    pk = torch.empty((n_req, max_pools, D), dtype=torch.bfloat16, device=dev)
    BLOCK_P = 16
    _pool_keys_kernel[(n_req, triton.cdiv(max_pools, BLOCK_P))](
        ape, cache, block_table, req_pools, pk, max_pools,
        block_table.stride(0), cache.stride(0),
        BLOCK_SIZE=block_size, D=D, KP=kp, ROW=_ROW_DIM, BLOCK_P=BLOCK_P,
    )
    _pooled_logits_matmul_kernel[(max_tiles, triton.cdiv(max_pools, PT))](
        q, weights, pk, tiles[0], tiles[1], tiles[2], visible, logits,
        max_pools, softmax_scale, H=H, D=D, KP=kp, RT=RT, PT=PT,
    )


def _prefill_row_req(chunk, R: int) -> torch.Tensor:
    """Block-table row of each query row of a prefill chunk. A request's
    rows are contiguous and in block-table order, and all of them carry
    the request's row start in `cu_seqlen_ks`, so a request begins
    wherever ks changes. (`chunk.token_to_seq` is indexed by KV token, not
    query row: a request with a cached prefix or an earlier chunk has more
    KV tokens than query rows, and reading it by row shifts every request
    after it onto the wrong block-table row.)"""
    ks = chunk.cu_seqlen_ks[:R]
    new_req = torch.ones(R, dtype=torch.int32, device=ks.device)
    new_req[1:] = ks[1:] != ks[:-1]
    return torch.cumsum(new_req, 0, dtype=torch.int32) - 1


def _pooled_topk(
    q: torch.Tensor,           # [R, H, D] bf16
    weights: torch.Tensor,     # [R, H] fp32
    ape: torch.Tensor,         # [KP, D] fp32
    cache: torch.Tensor,       # [num_blocks, BS, ROW] bf16 (pages may be strided)
    block_table: torch.Tensor, # [rows_bt, stride] int32
    row_req: torch.Tensor,     # [R] int32
    visible: torch.Tensor,     # [R] int32
    logits: torch.Tensor,      # [R, max_pools] fp32 workspace
    max_pools: int,
    block_size: int,
    softmax_scale: float,
    ksel: int,
    kp: int,
    by_request: bool = False,  # rows grouped by request: prefill chunks
) -> torch.Tensor:
    R, H, D = q.shape
    if R == 0:
        return torch.empty((0, ksel), dtype=torch.int32, device=q.device)
    n_pools = torch.div(visible, kp, rounding_mode="floor").to(torch.int32)
    if by_request:
        _pooled_logits_by_request(
            q, weights, ape, cache, block_table, row_req, n_pools, visible,
            logits, max_pools, block_size, softmax_scale, kp,
        )
    else:
        BLOCK_P = 16
        # Fixed program count per row; each program strides over the row's
        # actual pool tiles (see the kernel docstring).
        grid = (R, min(triton.cdiv(max_pools, BLOCK_P), _POOL_PROGRAMS))
        _pooled_logits_kernel[grid](
            q, weights, ape, cache, block_table, row_req, visible,
            logits, max_pools, block_table.stride(0), cache.stride(0),
            softmax_scale,
            BLOCK_SIZE=block_size, H=H, D=D, KP=kp, ROW=_ROW_DIM,
            BLOCK_P=BLOCK_P,
        )
    # top-k over pools: prefill-style ranges [0, n_pools) per row.
    zeros = torch.zeros_like(n_pools)
    sel = torch.empty((R, ksel), dtype=torch.int32, device=q.device)
    top_k_per_row_prefill(
        logits[:R], zeros, n_pools, sel, R, logits.stride(0), logits.stride(1),
        ksel,
    )
    return sel


def _pooled_select(
    q, weights, ape, cache, block_table, row_req, visible, logits,
    max_pools, block_size, softmax_scale, ksel, topk_out, kp,
    by_request=False,
) -> None:
    R = q.shape[0]
    if R == 0:
        return
    sel = _pooled_topk(
        q, weights, ape, cache, block_table, row_req, visible, logits,
        max_pools, block_size, softmax_scale, ksel, kp, by_request,
    )
    _expand_topk_kernel[(R,)](
        sel, visible, topk_out, sel.stride(0),
        KP=kp, KSEL=ksel, OUT_W=topk_out.shape[1], BLOCK_S=64,
    )


def _pooled_prefill_tp_select(
    q, weights, ape, cache, chunks, topk_out, block_size,
    softmax_scale, ksel, kp, group,
) -> None:
    """Disjoint query rows; one pool-ID exchange before token expansion.

    Adapted from vLLM #54951's row-ownership design. Pool IDs are 4x smaller
    than expanded indices for GLM53. No score reduction/quantization change.
    Chunk offsets exclude leading decode rows and trailing graph padding.
    """
    first, last = chunks[0].token_start, chunks[-1].token_end
    owned = triton.cdiv(last - first, group.world_size)
    start = first + group.rank_in_group * owned
    stop = min(start + owned, last)
    local = torch.full((owned, ksel), -1, dtype=torch.int32, device=q.device)
    for chunk in chunks:
        lo, hi = max(start, chunk.token_start), min(stop, chunk.token_end)
        if lo >= hi:
            continue
        R = chunk.token_end - chunk.token_start
        offset = slice(lo - chunk.token_start, hi - chunk.token_start)
        visible = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).to(torch.int32)
        row_req = _prefill_row_req(chunk, R)
        max_pools = max(1, chunk.max_seq_len // kp)
        logits = torch.empty((hi - lo, max_pools), dtype=torch.float32, device=q.device)
        local[lo - start:hi - start] = _pooled_topk(
            q[lo:hi], weights[lo:hi], ape, cache, chunk.block_table,
            row_req[offset], visible[offset], logits, max_pools, block_size,
            softmax_scale, ksel, kp, by_request=True,
        )
    # Keep local alive until all dependent work is enqueued. Empty owners still
    # participate with padding; every rank issues exactly one collective.
    gathered = group.all_gather(local, dim=0)
    for chunk in chunks:
        R = chunk.token_end - chunk.token_start
        if R <= 0:
            continue
        selected = gathered[chunk.token_start - first:chunk.token_end - first]
        visible = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).to(torch.int32)
        _expand_topk_kernel[(R,)](
            selected, visible, topk_out[chunk.token_start:chunk.token_end],
            selected.stride(0), KP=kp, KSEL=ksel, OUT_W=topk_out.shape[1], BLOCK_S=64,
        )


@instrument_pooled_indexer
def glm5_next_pooled_indexer(
    q: torch.Tensor,
    packed: torch.Tensor,
    weights: torch.Tensor,
    ape: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    decode_logits: torch.Tensor,
    max_pools_total: int,
    ksel: int,
    kp: int,
    softmax_scale: float,
) -> None:
    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if not isinstance(attn_metadata, dict):
        # Profiling / dummy run: nothing to index.
        return
    md = attn_metadata[k_cache_prefix]
    assert isinstance(md, DeepseekV32IndexerMetadata)
    num_tokens = md.slot_mapping.shape[0]
    # [num_blocks, block_size, ROW]; in the packed cross-layer slab the
    # block dim is strided (stride(0) > block_size * ROW), so kernels
    # address pages as block * stride(0) + offset * ROW, never a flat view.
    cache = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], _ROW_DIM)
    assert cache.stride(2) == 1 and cache.stride(1) == _ROW_DIM
    block_size = kv_cache.shape[1]

    # 1) insert this step's rows.
    BLOCK_T = 64
    _insert_rows_kernel[(triton.cdiv(num_tokens, BLOCK_T),)](
        packed[:num_tokens], cache, md.slot_mapping, num_tokens, cache.stride(0),
        BLOCK_SIZE=block_size, ROW=_ROW_DIM, BLOCK_T=BLOCK_T,
    )
    topk_indices_buffer[: q.shape[0]] = -1

    # 2) prefill chunks.
    if md.num_prefills > 0:
        assert md.prefill is not None
        chunks = md.prefill.chunks
        group = None
        if _prefill_row_sharding_enabled(chunks):
            group = _prefill_shard_group()
        if group is not None:
            logger.info_once(
                "GLM53 TP4 prefill indexer row sharding active (pool-ID exchange)."
            )
            _pooled_prefill_tp_select(
                q, weights, ape, cache, chunks, topk_indices_buffer,
                block_size, softmax_scale, ksel, kp, group,
            )
        for chunk in (() if group is not None else chunks):
            R = chunk.token_end - chunk.token_start
            if R <= 0:
                continue
            visible = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).to(torch.int32)
            row_req = _prefill_row_req(chunk, R)
            max_pools = max(1, chunk.max_seq_len // kp)
            logits = torch.empty(
                (R, max_pools), dtype=torch.float32, device=q.device
            )
            _pooled_select(
                q[chunk.token_start:chunk.token_end],
                weights[chunk.token_start:chunk.token_end],
                ape, cache, chunk.block_table, row_req, visible, logits,
                max_pools, block_size, softmax_scale, ksel,
                topk_indices_buffer[chunk.token_start:chunk.token_end], kp,
                by_request=_PREFILL_MATMUL,
            )

    # 3) decode rows (fixed-size workspace: CUDA-graph safe).
    if md.num_decodes > 0:
        assert md.decode is not None
        dm = md.decode
        R = md.num_decode_tokens
        seq_lens = dm.seq_lens
        if seq_lens.dim() == 2:
            visible = seq_lens.reshape(-1)[:R].to(torch.int32)
            next_n = seq_lens.shape[1]
        else:
            next_n = max(1, R // max(1, seq_lens.shape[0]))
            j = torch.arange(R, device=q.device, dtype=torch.int32) % next_n
            visible = (
                seq_lens.repeat_interleave(next_n)[:R] - next_n + j + 1
            ).to(torch.int32)
        row_req = (
            torch.arange(R, device=q.device, dtype=torch.int32) // next_n
        )
        max_pools = decode_logits.shape[1]
        _pooled_select(
            q[:R], weights[:R], ape, cache, dm.block_table, row_req, visible,
            decode_logits, max_pools, block_size, softmax_scale, ksel,
            topk_indices_buffer[:R], kp,
        )


def glm5_next_pooled_indexer_fake(
    q, packed, weights, ape, k_cache_prefix, kv_cache, topk_indices_buffer,
    decode_logits, max_pools_total, ksel, kp, softmax_scale,
) -> None:
    return None


direct_register_custom_op(
    op_name="glm5_next_pooled_indexer",
    op_func=glm5_next_pooled_indexer,
    mutates_args=["kv_cache", "topk_indices_buffer", "decode_logits"],
    fake_impl=glm5_next_pooled_indexer_fake,
)


# --------------------------------------------------------------------- module


class Glm5NextPooledIndexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor,
        prefix: str = "",
        fold_input_projections: bool = False,
    ) -> None:
        super().__init__()
        # When the owning attention layer folds wk, the kpool compress gate
        # and weights_proj into its fused_qkv_a_proj, forward() receives their
        # outputs as ``precomputed`` and this module holds no such weights.
        self.fold_input_projections = fold_input_projections
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        assert self.head_dim == _K_DIM
        self.index_topk = config.index_topk
        self.kp = config.index_kpool
        assert config.index_kpool_compress and config.index_kpool_always_select_tail
        self.ksel = self.index_topk // self.kp
        # Output width: expanded pools + tail, padded to a multiple of 32.
        self.topk_tokens = topk_indices_buffer.shape[1]
        self.softmax_scale = self.head_dim**-0.5
        self.n_head_scale = self.n_heads**-0.5
        self.topk_indices_buffer = topk_indices_buffer

        self.wq_b = ReplicatedLinear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wq_b",
        )
        if not fold_input_projections:
            self.wk = ReplicatedLinear(
                config.hidden_size, self.head_dim, bias=False,
                quant_config=quant_config, prefix=f"{prefix}.wk",
            )
            self.weights_proj = ReplicatedLinear(
                config.hidden_size, self.n_heads, bias=False,
                quant_config=quant_config, prefix=f"{prefix}.weights_proj",
            )
            self.index_kpool_compress_gate = nn.Parameter(
                torch.zeros(self.head_dim, config.hidden_size), requires_grad=False
            )
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.kp, self.head_dim), requires_grad=False
        )
        self.k_cache = Glm5NextIndexerCache(
            head_dim=_ROW_DIM,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        max_model_len = vllm_config.model_config.max_model_len
        self.max_pools_total = max(1, triton.cdiv(max_model_len, self.kp))
        sched = vllm_config.scheduler_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        decode_rows = sched.max_num_seqs * (1 + num_spec)
        self.decode_logits = torch.empty(
            (decode_rows, self.max_pools_total),
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        self._ape_f32: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb=None,
        precomputed: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim).to(torch.bfloat16)
        if precomputed is not None:
            k, gate, weights = precomputed
        else:
            k, _ = self.wk(hidden_states)
            gate = torch.nn.functional.linear(
                hidden_states,
                self.index_kpool_compress_gate.to(hidden_states.dtype),
            )
            weights, _ = self.weights_proj(hidden_states)
        k = torch.nn.functional.layer_norm(
            k.float(),
            (self.head_dim,),
            self.k_norm.weight.float(),
            self.k_norm.bias.float(),
            self.k_norm.eps,
        ).to(torch.bfloat16)
        packed = torch.cat([k, gate.to(torch.bfloat16)], dim=-1).contiguous()
        weights = (weights.float() * self.n_head_scale).contiguous()
        if self._ape_f32 is None or self._ape_f32.device != q.device:
            self._ape_f32 = self.index_kpool_compress_ape.float().contiguous()
        torch.ops.vllm.glm5_next_pooled_indexer(
            q.contiguous(),
            packed,
            weights,
            self._ape_f32,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            self.topk_indices_buffer,
            self.decode_logits,
            self.max_pools_total,
            self.ksel,
            self.kp,
            self.softmax_scale,
        )
        return self.topk_indices_buffer
