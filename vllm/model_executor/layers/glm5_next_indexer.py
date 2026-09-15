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

On Apple Metal (mps tensors; see ``_use_native_producer``) the same op body
runs a torch-native producer (``_insert_rows_native`` /
``_pooled_select_native``): fixed-shape
pool tiles gathered through the block table, the identical pool-softmax /
relu / head-weight arithmetic, ``torch.topk`` over pools, and a torch
expansion that writes the same ``[expanded pools | tail | -1 pad]`` row
layout the Triton ``_expand_topk_kernel`` produces. No host readback, no
boolean-mask compaction, no data-dependent shapes.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.glm5_next_indexer_workspace import (
    Glm5NextIndexerWorkspace,
)
from vllm.model_executor.layers.glm5_next_pool_cache import (
    POOL_CACHE_HEAD_DIM,
    cached_pool_logits,
    update_pool_cache,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.metal_phaseprof import phase as _qc_phase

logger = init_logger(__name__)


def glm5_next_device() -> torch.device:
    """Device for the model's step buffers (top-k index buffer, decode
    logits workspace). CUDA/ROCm keep ``cuda:<current>`` exactly as before;
    other platforms (Metal ``mps``, CPU) use their platform device type."""
    device_type = current_platform.device_type
    if device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device_type)


# Row layout of the cached indexer state.
_K_DIM = 128
_ROW_DIM = 2 * _K_DIM  # [k | gate]
_POOL_PROGRAMS = 128  # programs per row on the pool axis (stride loop inside)


def _compact_decode_options(vllm_config, heads):
    extra = vllm_config.additional_config or {}
    flags = []
    for key in ("glm5_next_adaptive_pool_score", "glm5_next_singleton_pool_update"):
        enabled = extra.get(key, False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{key} must be a boolean")
        if enabled and not extra.get("glm5_next_compact_indexer_cache", False):
            raise ValueError(f"{key} requires compact indexer cache")
        flags.append(enabled)
    if flags[0] and heads != 32:
        raise ValueError("Adaptive GLM pool scoring requires 32 heads")
    return tuple(flags)


def _row_shard_option(vllm_config, heads):
    extra = vllm_config.additional_config or {}
    enabled = extra.get("glm5_next_indexer_row_shard", False)
    if not isinstance(enabled, bool):
        raise ValueError("glm5_next_indexer_row_shard must be a boolean")
    if not enabled:
        return False
    parallel = vllm_config.parallel_config
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability((8, 0))
        or heads != 32
        or parallel.tensor_parallel_size not in (4, 8)
        or parallel.pipeline_parallel_size != 1
        or not extra.get("glm5_next_compact_indexer_cache", False)
    ):
        # Rows are sharded over the replica's own TP group, so DP replicas
        # are independent; TP4 shards 16/32 rows as 4/8 per rank.
        raise ValueError(
            "Indexer row sharding requires SM80, TP4 or TP8 with PP1, "
            "32 indexer heads and the compact cache"
        )
    return True


def _row_shard_dispatch(enabled, rows, num_prefills, next_n):
    # Capture-stable shape policy. Do NOT inspect context length here: a
    # Python branch would be frozen at capture, not reevaluated on replay.
    # Rows are independent query rows (row_req/visible are per row), so a
    # speculative batch of reqs x next_n rows shards the same way; only the
    # row count must be one of the qualified shapes.
    return enabled and rows in (16, 32) and num_prefills == 0


def _cache_row_dim(vllm_config: VllmConfig, kp: int) -> int:
    extra = vllm_config.additional_config
    enabled = extra.get("glm5_next_compact_indexer_cache", False) if isinstance(extra, dict) else False
    if not isinstance(enabled, bool):
        raise ValueError("glm5_next_compact_indexer_cache must be a boolean")
    if not enabled:
        return _ROW_DIM
    if not (current_platform.is_cuda() and current_platform.is_device_capability((8, 0))):
        raise ValueError("Compact GLM indexer cache is currently qualified only on SM80")
    num_spec = (vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config else 0)
    if kp != 4 or num_spec > 5:
        raise ValueError("Compact GLM indexer cache requires kpool=4 and at most five speculative tokens")
    return POOL_CACHE_HEAD_DIM


class Glm5NextIndexerBackend(DeepseekV32IndexerBackend):
    """DSV3.2 indexer metadata (slot mapping, prefill chunks, decode block
    tables) over a 256-wide bf16 row instead of the fp8 128+scale row."""

    @staticmethod
    def get_name() -> str:
        return "GLM5_NEXT_INDEXER"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_ROW_DIM, POOL_CACHE_HEAD_DIM]

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
    for s0 in range(0, KSEL, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        smask = s < KSEL
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

# ------------------------------------------------- native (Metal / CPU) producer
#
# Everything below is the torch-only counterpart of the three Triton kernels
# above, used whenever the tensors are not on a CUDA/ROCm device. Contract:
# fixed-shape tiles sized from python-int metadata, gathers through the
# block table, no ``.item()``/``nonzero``/boolean compaction, no host copies.

# Byte budget for one gathered pool tile ([bt_rows, P, KP, ROW] bf16) and the
# number of query rows scored per tile; both bound the fp32 intermediates.
_NATIVE_TILE_BYTES = 32 << 20
_NATIVE_ROW_CHUNK = 256
# MPS index kernels have been observed to fold element offsets into signed
# 32-bit integers for strided sources (see metal_attn.py's range gather);
# a page cache spanning more elements than that is gathered window by
# window with the tensor's own (64-bit) storage offset doing the addressing.
_INDEX32_LIMIT = 2**31 - 1


def _use_native_producer(t: torch.Tensor) -> bool:
    """Whether the torch-native producer serves this call.

    MPS tensors always (Triton does not target Metal). ``VLLM_METAL_GLM_INDEXER``
    mirrors ``VLLM_METAL_KDA``: ``native`` forces the torch path on any
    device (CPU parity tests), ``0`` pins the Triton route. CPU tensors
    otherwise keep the Triton route, which the CUDA unit tests drive with
    mocked kernels."""
    flag = os.environ.get("VLLM_METAL_GLM_INDEXER", "1").strip().lower()
    if flag == "native":
        return True
    if flag in ("0", "false", "off"):
        return False
    return t.device.type == "mps"


def _cache_extent(cache: torch.Tensor) -> int:
    """Elements spanned by a [num_blocks, BS, ROW] page view (pages may be
    strided by the packed cross-layer slab)."""
    if cache.shape[0] == 0:
        return 0
    return (cache.shape[0] - 1) * cache.stride(0) + cache.shape[1] * cache.shape[2]


def _gather_cache_rows(
    cache: torch.Tensor, blk: torch.Tensor, off: torch.Tensor
) -> torch.Tensor:
    """rows[...] = cache[blk[...], off[...]] (int64 indices, any leading
    shape). Windowed when the page view exceeds the 32-bit offset range."""
    if cache.device.type != "mps" or _cache_extent(cache) <= _INDEX32_LIMIT:
        return cache[blk, off]
    win = max(1, _INDEX32_LIMIT // max(1, cache.stride(0)))
    out = torch.zeros(
        (*blk.shape, cache.shape[2]), dtype=cache.dtype, device=cache.device
    )
    for b0 in range(0, cache.shape[0], win):
        b1 = min(cache.shape[0], b0 + win)
        inwin = (blk >= b0) & (blk < b1)
        local = torch.where(inwin, blk - b0, 0)
        out = torch.where(inwin[..., None], cache[b0:b1][local, off], out)
    return out


def _scatter_cache_rows(
    cache: torch.Tensor, blk: torch.Tensor, off: torch.Tensor, vals: torch.Tensor
) -> None:
    """cache[blk[t], off[t]] = vals[t]. Windowed like the gather when the
    page view exceeds the 32-bit offset range; rows outside a window
    rewrite the window's row 0 with its own contents (no valid row ever
    targets block 0, the KV manager's null block)."""
    if cache.device.type != "mps" or _cache_extent(cache) <= _INDEX32_LIMIT:
        cache[blk, off] = vals
        return
    win = max(1, _INDEX32_LIMIT // max(1, cache.stride(0)))
    for b0 in range(0, cache.shape[0], win):
        b1 = min(cache.shape[0], b0 + win)
        sub = cache[b0:b1]
        inwin = (blk >= b0) & (blk < b1)
        local = torch.where(inwin, blk - b0, 0)
        loff = torch.where(inwin, off, 0)
        sub[local, loff] = torch.where(inwin[:, None], vals, sub[0, 0][None, :])


_METAL_IDX: bool | None = None


def _metal_indexer_kernels() -> bool:
    """quixicore Metal indexer kernels (pool logits, top-k expand, paged row
    insert). VLLM_METAL_INDEXER_KERNEL=0 pins the torch producer."""
    global _METAL_IDX
    if _METAL_IDX is None:
        _METAL_IDX = False
        if current_platform.is_metal() and os.environ.get(
            "VLLM_METAL_INDEXER_KERNEL", "1"
        ) != "0":
            try:
                from vllm.quixicore import quixicore_ops

                _METAL_IDX = quixicore_ops.is_available() and (
                    quixicore_ops.has("glm5_indexer_pool_logits")
                )
            except Exception:
                _METAL_IDX = False
    return _METAL_IDX


def _insert_rows_native(
    packed: torch.Tensor, cache: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    """Torch counterpart of ``_insert_rows_kernel``: cache[slot[t]] = src[t]
    for slot[t] >= 0, slots resolved as (slot // BS, slot % BS). PAD rows
    (slot -1) land on block 0 row 0, the KV manager's null block that no
    request reads (the same convention as metal_attn's cache update);
    a data-dependent mask would need a host sync."""
    if packed.shape[0] == 0:
        return
    if (
        _metal_indexer_kernels()
        and cache.device.type == "mps"
        and cache.dim() == 3
        and cache.stride(2) == 1
        and cache.stride(1) == cache.shape[2]
    ):
        from vllm.quixicore import quixicore_ops

        rows = packed if packed.dtype == cache.dtype else packed.to(cache.dtype)
        slots = slot_mapping
        if slots.dtype not in (torch.int32, torch.int64):
            slots = slots.to(torch.int64)
        quixicore_ops.paged_row_insert(rows, cache, slots.contiguous())
        return
    bs = cache.shape[1]
    slot = slot_mapping.to(torch.int64).clamp_min(0)
    blk = torch.div(slot, bs, rounding_mode="floor")
    off = slot - blk * bs
    _scatter_cache_rows(cache, blk, off, packed.to(cache.dtype))


def _pool_keys_native(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    ape: torch.Tensor,
    kp: int,
    p0: int,
    p1: int,
) -> torch.Tensor:
    """Pool keys for pools [p0, p1) of every block-table row, [B, P, D]
    fp32, rounded through bf16 exactly where the Triton kernel feeds them
    to its bf16 dot. Pools past a row's context read whatever their
    (clamped) block-table entry points at; the caller masks them."""
    D = ape.shape[1]
    dev = cache.device
    B = block_table.shape[0]
    P = p1 - p0
    tok = torch.arange(p0, p1, device=dev)[:, None] * kp + torch.arange(
        kp, device=dev
    )[None, :]  # [P, KP]
    col = torch.div(tok, block_size, rounding_mode="floor").clamp_(
        max=max(block_table.shape[1] - 1, 0)
    )
    blk = (
        block_table.to(torch.int64)
        .index_select(1, col.reshape(-1))
        .view(B, P, kp)
        .clamp_(0, cache.shape[0] - 1)
    )
    off = torch.remainder(tok, block_size)[None].expand(B, P, kp)
    rows = _gather_cache_rows(cache, blk, off)  # [B, P, KP, ROW]
    k = rows[..., :D].float()
    g = rows[..., D:].float() + ape[None, None, :, :]
    probs = torch.softmax(g, dim=2)
    return (probs * k).sum(dim=2).to(torch.bfloat16).float()


_METAL_INDEXER_IDENTITY: bool | None = None
_IDENTITY_ARANGE: dict[tuple[str, int], torch.Tensor] = {}


def _metal_indexer_identity() -> bool:
    """VLLM_METAL_INDEXER_IDENTITY=1 (glm53f-q2-1): below the selection
    limit the pooled top-k is the identity; skip scoring it."""
    global _METAL_INDEXER_IDENTITY
    if _METAL_INDEXER_IDENTITY is None:
        _METAL_INDEXER_IDENTITY = (
            os.environ.get("VLLM_METAL_INDEXER_IDENTITY", "0") == "1"
        )
    return _METAL_INDEXER_IDENTITY


def _identity_arange(dev: torch.device, n: int) -> torch.Tensor:
    key = (str(dev), n)
    t = _IDENTITY_ARANGE.get(key)
    if t is None:
        t = torch.arange(n, dtype=torch.int32, device=dev)
        _IDENTITY_ARANGE[key] = t
    return t


def _pooled_select_native(
    q: torch.Tensor,           # [R, H, D] bf16
    weights: torch.Tensor,     # [R, H] fp32 (already * n_heads^-0.5)
    ape: torch.Tensor,         # [KP, D] fp32
    cache: torch.Tensor,       # [num_blocks, BS, ROW] bf16 (pages may be strided)
    block_table: torch.Tensor, # [rows_bt, stride] int
    row_req: torch.Tensor,     # [R] block-table row per query row
    visible: torch.Tensor,     # [R] visible tokens per query row
    block_size: int,
    softmax_scale: float,
    ksel: int,
    topk_out: torch.Tensor,    # [R, OUT_W] int32
    kp: int,
    pool_bound: int,
    tlen_out: torch.Tensor | None = None,  # [R] int32 valid prefix length
) -> None:
    """Native pooled logits + top-k pools + expansion. ``pool_bound`` is a
    python-int bound on every row's pool count (batch max_seq_len // kp).
    ``tlen_out`` receives each row's valid prefix length (the sparse decode
    kernel stops its scan there)."""
    R, H, D = q.shape
    assert cache.shape[-1] == _ROW_DIM, "native GLM indexer serves the 256-wide row"
    dev = q.device
    pool_bound = max(1, pool_bound)
    if (
        _metal_indexer_kernels()
        and dev.type == "mps"
        and kp == 4
        and D == 128
        and H <= 64
        and q.dtype in (torch.bfloat16, torch.float16)
        and cache.dtype == q.dtype
        and cache.stride(2) == 1
        and cache.stride(1) == _ROW_DIM
        and block_size % 4 == 0
    ):
        # Metal: pool logits in one launch, torch top-k (sorted, so valid
        # pools come first), expand + tail in one launch.
        from vllm.quixicore import quixicore_ops

        vis = visible if visible.dtype == torch.int32 else visible.to(torch.int32)
        if pool_bound <= ksel and _metal_indexer_identity():
            # Every pool of every row fits the selection (context below
            # index_topk): the top-k is the identity, so skip the pool
            # logits and the torch top-k chain and emit each row's pools in
            # position order (valid prefix, then -1). Same selected SET as
            # the scored path; only the list order differs, which the
            # sparse decode kernel's fp32 partition merge sees as rounding
            # (W16c precedent). Opt-in per profile
            # (VLLM_METAL_INDEXER_IDENTITY=1, the glm53f-q2-1 env block).
            if quixicore_ops.has("glm5_indexer_expand_identity"):
                quixicore_ops.glm5_indexer_expand_identity(
                    vis.contiguous(), topk_out, kp, ksel, tlen_out
                )
                return
            n_pools = torch.div(vis, kp, rounding_mode="floor")
            ar = _identity_arange(dev, ksel)
            sel = torch.where(
                ar[None, :] < n_pools[:, None], ar[None, :].expand(R, ksel), -1
            )
            quixicore_ops.glm5_indexer_expand_topk(
                sel.contiguous(), vis.contiguous(), topk_out, kp, tlen_out
            )
            return
        bt = block_table if block_table.dtype == torch.int32 else block_table.to(
            torch.int32
        )
        rr = row_req if row_req.dtype == torch.int32 else row_req.to(torch.int32)
        logits = quixicore_ops.glm5_indexer_pool_logits(
            q.contiguous(), weights.contiguous(), ape.contiguous(), cache,
            bt.contiguous(), rr.contiguous(), vis.contiguous(), pool_bound,
            block_size, softmax_scale,
        )
        tk = min(ksel, pool_bound)
        vals, idx = torch.topk(logits, tk, dim=-1, sorted=True)
        sel = idx.to(torch.int32).masked_fill(vals == float("-inf"), -1)
        if tk < ksel:
            sel = torch.nn.functional.pad(sel, (0, ksel - tk), value=-1)
        quixicore_ops.glm5_indexer_expand_topk(
            sel.contiguous(), vis.contiguous(), topk_out, kp, tlen_out
        )
        return
    n_pools = torch.div(visible.to(torch.int64), kp, rounding_mode="floor")
    row_req64 = row_req.to(torch.int64)
    qf = q.float()
    tile = _NATIVE_TILE_BYTES // max(
        1, block_table.shape[0] * kp * cache.shape[-1] * cache.element_size()
    )
    tile = max(64, min(2048, tile))

    best_vals = torch.full((R, ksel), float("-inf"), dtype=torch.float32, device=dev)
    best_idx = torch.full((R, ksel), -1, dtype=torch.int64, device=dev)
    for p0 in range(0, pool_bound, tile):
        p1 = min(p0 + tile, pool_bound)
        with _qc_phase("idx_pool_keys"):
            pk = _pool_keys_native(cache, block_table, block_size, ape, kp, p0, p1)
        pcol = torch.arange(p0, p1, device=dev)
        chunk_vals, chunk_idx = [], []
        for r0 in range(0, R, _NATIVE_ROW_CHUNK):
            r1 = min(r0 + _NATIVE_ROW_CHUNK, R)
            pk_rows = pk.index_select(0, row_req64[r0:r1])  # [r, P, D]
            scores = torch.einsum("rpd,rhd->rph", pk_rows, qf[r0:r1])
            scores = torch.relu(scores * softmax_scale)
            logits = (scores * weights[r0:r1, None, :]).sum(dim=-1)  # [r, P]
            logits = logits.masked_fill(
                pcol[None, :] >= n_pools[r0:r1, None], float("-inf")
            )
            tk = min(ksel, p1 - p0)
            vals, idx = torch.topk(logits, tk, dim=-1)
            chunk_vals.append(vals)
            chunk_idx.append(idx + p0)
        vals = torch.cat(chunk_vals, dim=0)
        idx = torch.cat(chunk_idx, dim=0)
        merged_vals, keep = torch.topk(
            torch.cat((best_vals, vals), dim=1), ksel, dim=1
        )
        best_idx = torch.cat((best_idx, idx), dim=1).gather(1, keep)
        best_vals = merged_vals
    sel = best_idx.masked_fill(best_vals == float("-inf"), -1)
    with _qc_phase("idx_expand"):
        _expand_topk_native(sel, visible, topk_out, kp, ksel, tlen_out)


def _expand_topk_native(
    sel: torch.Tensor,      # [R, KSEL] pool indices (-1 invalid), valid first
    visible: torch.Tensor,  # [R]
    out: torch.Tensor,      # [R, OUT_W] int32
    kp: int,
    ksel: int,
    tlen_out: torch.Tensor | None = None,
) -> None:
    """Torch counterpart of ``_expand_topk_kernel``: expanded pools, then
    the incomplete tail pool's tokens right after the last valid pool,
    then -1 padding (see that kernel for why the tail is not parked at a
    fixed column)."""
    R, out_w = out.shape
    dev = out.device
    assert ksel * kp + kp - 2 < out_w, "top-k output row too narrow for tail"
    vis = visible.to(torch.int64)
    n_pools = torch.div(vis, kp, rounding_mode="floor")
    tail_count = vis - n_pools * kp
    tail_start = n_pools * kp
    n_sel = n_pools.clamp(max=ksel)
    pool = sel.to(torch.int64)
    ok = (pool >= 0) & (pool < n_pools[:, None])
    m = torch.arange(kp, device=dev)
    tokens = torch.where(ok[:, :, None], pool[:, :, None] * kp + m, -1)
    out.fill_(-1)
    out[:, : ksel * kp] = tokens.reshape(R, ksel * kp).to(out.dtype)
    # Tail slots n_sel*kp .. n_sel*kp+kp-1, exactly the kernel's store
    # (it also overwrites slot n_sel, which a real top-k leaves at -1).
    # The last slot is always -1 (tail_count <= kp-1); it is skipped only
    # when it would fall off the row, where the kernel masks its store.
    nt = kp if ksel * kp + kp - 1 < out_w else kp - 1
    if nt > 0:
        mt = m[:nt]
        tcol = n_sel[:, None] * kp + mt[None, :]
        tval = torch.where(
            mt[None, :] < tail_count[:, None], tail_start[:, None] + mt[None, :], -1
        )
        out.scatter_(1, tcol, tval.to(out.dtype))
    if tlen_out is not None:
        # Same bound the Metal expand kernel writes: every valid entry is
        # below n_sel * kp + min(tail_count, nt).
        tlen_out[:R] = (n_sel * kp + tail_count.clamp(max=nt)).to(tlen_out.dtype)


# --------------------------------------------------------------------- core op


def _prefill_query_requests(chunk) -> torch.Tensor:
    # token_to_seq maps the gathered KV span, not the current query span.
    # With cached prefixes (or a sliced query chunk), its first R entries
    # need not belong to the R queries. Each query's cu_seqlen_ks identifies
    # its request's start in that KV span, so gather the map at those starts.
    # No device-to-host readback; int32 index_select preserves the row dtype.
    assert (chunk.local_cu_seq_lens is None
            or chunk.local_cu_seq_lens is chunk.cu_seq_lens), (
        "GLM pooled prefill query mapping requires unsharded KV row bounds"
    )
    return torch.index_select(chunk.token_to_seq, 0, chunk.cu_seqlen_ks)


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
    adaptive_score: bool = False,
    group: int = 1,
) -> torch.Tensor:
    R, H, D = q.shape
    assert R > 0
    BLOCK_P = 16
    # Fixed program count per row; each program strides over the row's
    # actual pool tiles (see the kernel docstring).
    grid = (R, min(triton.cdiv(max_pools, BLOCK_P), _POOL_PROGRAMS))
    if cache.shape[-1] == POOL_CACHE_HEAD_DIM:
        assert kp == 4 and D == 128 and softmax_scale == 128**-0.5
        if adaptive_score and R <= 64:
            from vllm.model_executor.layers.glm5_next_pool_score import (
                adaptive_pool_logits,
            )

            adaptive_pool_logits(q, weights, cache, block_table, row_req, visible, logits)
        elif 2 <= group <= 8 and R % group == 0:
            # Speculative verify rows of one request share their history:
            # read each pool tile once for all `group` rows.
            from vllm.model_executor.layers.glm5_next_pool_cache import (
                cached_pool_logits_grouped,
            )

            cached_pool_logits_grouped(
                q, weights, cache, block_table, row_req, visible, logits, group
            )
        else:
            cached_pool_logits(q, weights, cache, block_table, row_req, visible, logits)
    else:
        _pooled_logits_kernel[grid](
            q, weights, ape, cache, block_table, row_req, visible,
            logits, max_pools, block_table.stride(0), cache.stride(0), softmax_scale,
            BLOCK_SIZE=block_size, H=H, D=D, KP=kp, ROW=_ROW_DIM, BLOCK_P=BLOCK_P,
        )
    # top-k over pools: prefill-style ranges [0, n_pools) per row.
    n_pools = torch.div(visible, kp, rounding_mode="floor").to(torch.int32)
    zeros = torch.zeros_like(n_pools)
    sel = torch.empty((R, ksel), dtype=torch.int32, device=q.device)
    ops.top_k_per_row_prefill(
        logits[:R], zeros, n_pools, sel, R, logits.stride(0), logits.stride(1),
        ksel,
    )
    return sel


def _pooled_select(
    q, weights, ape, cache, block_table, row_req, visible, logits,
    max_pools, block_size, softmax_scale, ksel, topk_out, kp,
    adaptive_score=False, row_shard=False, row_group=1, pool_bound=None,
    tlen_out=None,
) -> None:
    R = q.shape[0]
    if R == 0:
        return
    if _use_native_producer(q):
        # Metal (mps): torch-native producer, same outputs.
        assert not row_shard and not adaptive_score
        _pooled_select_native(
            q, weights, ape, cache, block_table, row_req, visible,
            block_size, softmax_scale, ksel, topk_out, kp,
            max_pools if pool_bound is None else min(max_pools, pool_bound),
            tlen_out,
        )
        return
    if row_shard:
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        assert group.world_size in (4, 8) and 0 <= group.rank_in_group < group.world_size
        assert R in (16, 32) and q.shape[1:] == (32, 128)
        assert cache.shape[-1] == POOL_CACHE_HEAD_DIM and kp == 4 and ksel == 512
        assert not adaptive_score
        communicator = group.device_communicator
        assert communicator is not None
        communicator.wait_for_comm_init()
        pynccl = communicator.pynccl_comm
        assert pynccl is not None and not pynccl.disabled
        local_rows = R // group.world_size
        lo = group.rank_in_group * local_rows
        hi = lo + local_rows
        local_sel = _pooled_topk(
            q[lo:hi], weights[lo:hi], ape, cache, block_table,
            row_req[lo:hi], visible[lo:hi], logits[:local_rows],
            max_pools, block_size, softmax_scale, ksel, kp,
            group=row_group if local_rows % max(row_group, 1) == 0 else 1,
        )
        sel = torch.empty((R, ksel), dtype=torch.int32, device=q.device)
        # Reuse the live serving communicator on the caller stream. Creating
        # another ProcessGroupNCCL communicator cost 484 MiB/rank on TP8 A100.
        # Resolve it inside this opaque op, never serialize its pointer in AOT.
        pynccl.all_gather(sel, local_sel)
    else:
        sel = _pooled_topk(
            q, weights, ape, cache, block_table, row_req, visible, logits,
            max_pools, block_size, softmax_scale, ksel, kp, adaptive_score,
            group=row_group,
        )
    _expand_topk_kernel[(R,)](
        sel, visible, topk_out, sel.stride(0),
        KP=kp, KSEL=ksel, OUT_W=topk_out.shape[1], BLOCK_S=64,
    )


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
    topk_len_buffer: torch.Tensor,
    adaptive_score: bool = False,
    singleton_fused_update: bool = False,
    row_shard_decode: bool = False,
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
    row_dim = kv_cache.shape[-1]
    assert row_dim in (_ROW_DIM, POOL_CACHE_HEAD_DIM)
    cache = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], row_dim)
    assert cache.stride(2) == 1 and cache.stride(1) == row_dim
    block_size = kv_cache.shape[1]

    # 1) insert this step's rows.
    BLOCK_T = 64
    if row_dim == POOL_CACHE_HEAD_DIM:
        update_pool_cache(packed[:num_tokens], md.slot_mapping, ape, cache,
                          singleton_fused=singleton_fused_update)
    elif _use_native_producer(q):
        with _qc_phase("idx_insert"):
            _insert_rows_native(packed[:num_tokens], cache, md.slot_mapping)
    else:
        _insert_rows_kernel[(triton.cdiv(num_tokens, BLOCK_T),)](
            packed[:num_tokens], cache, md.slot_mapping, num_tokens, cache.stride(0),
            BLOCK_SIZE=block_size, ROW=_ROW_DIM, BLOCK_T=BLOCK_T,
        )
    topk_indices_buffer[: q.shape[0]] = -1
    # Valid prefix length per row (Metal native producer writes the exact
    # bound; anything left unwritten keeps the full padded width).
    topk_len_buffer[: q.shape[0]] = topk_indices_buffer.shape[1]

    # 2) prefill chunks.
    if md.num_prefills > 0:
        assert md.prefill is not None
        for chunk in md.prefill.chunks:
            R = chunk.token_end - chunk.token_start
            if R <= 0:
                continue
            visible = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).to(torch.int32)
            row_req = _prefill_query_requests(chunk)
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
                tlen_out=topk_len_buffer[chunk.token_start:chunk.token_end],
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
            adaptive_score=adaptive_score,
            row_shard=_row_shard_dispatch(row_shard_decode, R, md.num_prefills, next_n),
            # Decode rows are request-major (row_req = arange // next_n), so a
            # request's k+1 verify rows are contiguous.
            row_group=next_n if R % max(next_n, 1) == 0 else 1,
            tlen_out=topk_len_buffer[:R],
            # Native path only: the batch's longest context (CPU metadata)
            # bounds the pool tiles; the Triton kernel strides by `visible`.
            pool_bound=(
                max(1, md.max_seq_len // kp)
                if getattr(md, "max_seq_len", None) is not None
                else None
            ),
        )


def glm5_next_pooled_indexer_fake(
    q, packed, weights, ape, k_cache_prefix, kv_cache, topk_indices_buffer,
    decode_logits, max_pools_total, ksel, kp, softmax_scale, topk_len_buffer,
    adaptive_score=False, singleton_fused_update=False, row_shard_decode=False,
) -> None:
    return None


direct_register_custom_op(
    op_name="glm5_next_pooled_indexer",
    op_func=glm5_next_pooled_indexer,
    mutates_args=[
        "kv_cache", "topk_indices_buffer", "decode_logits", "topk_len_buffer"
    ],
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
        workspace: Glm5NextIndexerWorkspace | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.adaptive_score, self.singleton_fused_update = _compact_decode_options(
            vllm_config, self.n_heads
        )
        self.row_shard_decode = _row_shard_option(vllm_config, self.n_heads)
        if self.row_shard_decode and self.adaptive_score:
            raise ValueError("Row sharding does not support experimental adaptive scoring")
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
        # Per-row valid prefix length of the expanded top-k row (written
        # beside the indices; the sparse decode stops scanning there).
        self.topk_len_buffer = torch.full(
            (topk_indices_buffer.shape[0],),
            self.topk_tokens,
            dtype=torch.int32,
            device=topk_indices_buffer.device,
        )

        self.wq_b = ReplicatedLinear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            config.hidden_size, self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wk",
        )
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            config.hidden_size, self.n_heads, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.weights_proj",
        )
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.kp, self.head_dim), requires_grad=False
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(self.head_dim, config.hidden_size), requires_grad=False
        )
        self.k_cache = Glm5NextIndexerCache(
            head_dim=_cache_row_dim(vllm_config, self.kp),
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
        device = glm5_next_device()
        if workspace is None:
            self.decode_logits = torch.empty(
                (decode_rows, self.max_pools_total), dtype=torch.float32, device=device
            )
        else:
            self.decode_logits = workspace.get_decode_logits(
                decode_rows, self.max_pools_total, device
            )
        self._ape_f32: torch.Tensor | None = None
        # Metal: one [k | gate | weights] linear over hidden_states plus the
        # glm5_indexer_pack kernel replace wk + gate + weights_proj and their
        # float/layer_norm/to/cat/mul glue (built lazily after weight load).
        self._fused_kgw: torch.Tensor | None = None
        self._k_norm_f32: tuple[torch.Tensor, torch.Tensor] | None = None

    def _metal_pack_inputs(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            hidden_states.device.type != "mps"
            or hidden_states.dtype not in (torch.bfloat16, torch.float16)
            or not _metal_indexer_kernels()
            or os.getenv("VLLM_METAL_INDEXER_PACK", "1") == "0"
        ):
            return None
        from vllm.quixicore import quixicore_ops

        if self._fused_kgw is None or self._fused_kgw.device != hidden_states.device:
            if not quixicore_ops.has("glm5_indexer_pack"):
                return None
            self._fused_kgw = (
                torch.cat(
                    [
                        self.wk.weight.to(hidden_states.dtype),
                        self.index_kpool_compress_gate.to(hidden_states.dtype),
                        self.weights_proj.weight.to(hidden_states.dtype),
                    ],
                    dim=0,
                )
                .contiguous()
                .to(hidden_states.device)
            )
            self._k_norm_f32 = (
                self.k_norm.weight.detach().float().contiguous(),
                self.k_norm.bias.detach().float().contiguous(),
            )
        fused = torch.nn.functional.linear(hidden_states, self._fused_kgw)
        T = fused.shape[0]
        packed = torch.empty(
            (T, 2 * self.head_dim), dtype=fused.dtype, device=fused.device
        )
        weights = torch.empty((T, self.n_heads), dtype=torch.float32, device=fused.device)
        norm_w, norm_b = self._k_norm_f32
        quixicore_ops.glm5_indexer_pack(
            fused,
            norm_w,
            norm_b,
            packed,
            weights,
            self.head_dim,
            self.k_norm.eps,
            self.n_head_scale,
        )
        return packed, weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb=None,
    ) -> torch.Tensor:
        with _qc_phase("idx_wq_b"):
            q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim).to(torch.bfloat16)
        with _qc_phase("idx_wk"):
            fused_pack = self._metal_pack_inputs(hidden_states)
        if fused_pack is not None:
            packed, weights = fused_pack
        else:
            k, _ = self.wk(hidden_states)
            k = torch.nn.functional.layer_norm(
                k.float(),
                (self.head_dim,),
                self.k_norm.weight.float(),
                self.k_norm.bias.float(),
                self.k_norm.eps,
            ).to(torch.bfloat16)
            gate = torch.nn.functional.linear(
                hidden_states, self.index_kpool_compress_gate.to(hidden_states.dtype)
            ).to(torch.bfloat16)
            packed = torch.cat([k, gate], dim=-1).contiguous()
            weights, _ = self.weights_proj(hidden_states)
            weights = (weights.float() * self.n_head_scale).contiguous()
        if self._ape_f32 is None or self._ape_f32.device != q.device:
            self._ape_f32 = self.index_kpool_compress_ape.float().contiguous()
        with _qc_phase("idx_op"):
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
                self.topk_len_buffer,
                self.adaptive_score,
                self.singleton_fused_update,
                self.row_shard_decode,
            )
        return self.topk_indices_buffer
