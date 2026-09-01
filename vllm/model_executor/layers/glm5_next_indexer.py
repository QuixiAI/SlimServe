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

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

logger = init_logger(__name__)

# Row layout of the cached indexer state.
_K_DIM = 128
_ROW_DIM = 2 * _K_DIM  # [k | gate]


class Glm5NextIndexerBackend(DeepseekV32IndexerBackend):
    """DSV3.2 indexer metadata (slot mapping, prefill chunks, decode block
    tables) over a 256-wide bf16 row instead of the fp8 128+scale row."""

    @staticmethod
    def get_name() -> str:
        return "GLM5_NEXT_INDEXER"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_ROW_DIM]


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
    ROW: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """cache[slot[t]] = src[t] for slot[t] >= 0."""
    pid = tl.program_id(0)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < num_tokens
    slot = tl.load(slot_ptr + t, mask=tmask, other=-1)
    valid = tmask & (slot >= 0)
    cols = tl.arange(0, ROW)
    vals = tl.load(
        src_ptr + t[:, None] * ROW + cols[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    tl.store(
        cache_ptr + slot[:, None].to(tl.int64) * ROW + cols[None, :],
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
    softmax_scale,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    KP: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    r = tl.program_id(0)
    pb = tl.program_id(1)
    vis = tl.load(vis_ptr + r)
    n_pools = vis // KP
    req = tl.load(row_req_ptr + r)

    p = pb * BLOCK_P + tl.arange(0, BLOCK_P)  # [P]
    pmask = p < n_pools
    m = tl.arange(0, KP)  # [KP]
    tok = p[:, None] * KP + m[None, :]  # [P, KP]
    blk = tl.load(
        bt_ptr + req.to(tl.int64) * bt_stride + tok // BLOCK_SIZE,
        mask=pmask[:, None],
        other=0,
    )
    slot = blk.to(tl.int64) * BLOCK_SIZE + (tok % BLOCK_SIZE)  # [P, KP]
    d = tl.arange(0, D)
    base = cache_ptr + slot[:, :, None] * ROW  # [P, KP, 1]
    k = tl.load(base + d[None, None, :], mask=pmask[:, None, None], other=0.0)
    g = tl.load(
        base + D + d[None, None, :], mask=pmask[:, None, None], other=0.0
    )
    ape = tl.load(ape_ptr + m[:, None] * D + d[None, :])  # [KP, D]
    logits_g = g.to(tl.float32) + ape[None, :, :]  # [P, KP, D]
    # softmax over the pool members (axis 1), per channel
    mx = tl.max(logits_g, axis=1)  # [P, D]
    e = tl.exp(logits_g - mx[:, None, :])
    probs = e / tl.sum(e, axis=1)[:, None, :]
    pool_key = tl.sum(probs * k.to(tl.float32), axis=1)  # [P, D]

    h = tl.arange(0, H)
    q = tl.load(q_ptr + (r.to(tl.int64) * H + h[:, None]) * D + d[None, :])
    q = q.to(tl.float32)  # [H, D]
    w = tl.load(w_ptr + r * H + h)  # [H]
    # scores[P, H] = relu(scale * pool_key . q_h)
    scores = tl.dot(pool_key.to(tl.bfloat16), tl.trans(q.to(tl.bfloat16)))
    scores = tl.maximum(scores * softmax_scale, 0.0)
    logit = tl.sum(scores * w[None, :], axis=1)  # [P]
    logit = tl.where(pmask, logit, float("-inf"))
    omask = p < max_pools
    tl.store(out_ptr + r.to(tl.int64) * max_pools + p, logit, mask=omask)


@triton.jit
def _expand_topk_kernel(
    sel_ptr,        # [R, KSEL] int32 pool indices (-1 invalid)
    vis_ptr,        # [R] int32
    out_ptr,        # [R, OUT_W] int32
    sel_stride,
    KP: tl.constexpr,
    KSEL: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    r = tl.program_id(0)
    vis = tl.load(vis_ptr + r)
    n_pools = vis // KP
    tail_count = vis - n_pools * KP
    tail_start = n_pools * KP
    for s0 in range(0, KSEL, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        smask = s < KSEL
        pool = tl.load(sel_ptr + r.to(tl.int64) * sel_stride + s, mask=smask, other=-1)
        ok = smask & (pool >= 0) & (pool < n_pools)
        m = tl.arange(0, KP)
        tokens = pool[:, None] * KP + m[None, :]
        tokens = tl.where(ok[:, None], tokens, -1)
        tl.store(
            out_ptr + r.to(tl.int64) * OUT_W + s[:, None] * KP + m[None, :],
            tokens,
            mask=smask[:, None],
        )
    # tail + padding: columns [KSEL*KP, OUT_W)
    c = KSEL * KP + tl.arange(0, KP)
    tail = tl.arange(0, KP)
    tval = tl.where(tail < tail_count, tail_start + tail, -1)
    tl.store(out_ptr + r.to(tl.int64) * OUT_W + c, tval, mask=c < OUT_W)
    # any remaining columns (OUT_W padded past KSEL*KP + KP) are -1
    extra = KSEL * KP + KP + tl.arange(0, 32)
    tl.store(
        out_ptr + r.to(tl.int64) * OUT_W + extra,
        tl.full((32,), -1, tl.int32),
        mask=extra < OUT_W,
    )


# --------------------------------------------------------------------- core op


def _pooled_select(
    q: torch.Tensor,           # [R, H, D] bf16
    weights: torch.Tensor,     # [R, H] fp32
    ape: torch.Tensor,         # [KP, D] fp32
    cache: torch.Tensor,       # [num_slots, ROW] bf16 (flat)
    block_table: torch.Tensor, # [rows_bt, stride] int32
    row_req: torch.Tensor,     # [R] int32
    visible: torch.Tensor,     # [R] int32
    logits: torch.Tensor,      # [R, max_pools] fp32 workspace
    max_pools: int,
    block_size: int,
    softmax_scale: float,
    ksel: int,
    topk_out: torch.Tensor,    # [R, OUT_W] int32
    kp: int,
) -> None:
    R, H, D = q.shape
    if R == 0:
        return
    BLOCK_P = 16
    grid = (R, triton.cdiv(max_pools, BLOCK_P))
    _pooled_logits_kernel[grid](
        q, weights, ape, cache, block_table, row_req, visible,
        logits, max_pools, block_table.stride(0), softmax_scale,
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
) -> None:
    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if not isinstance(attn_metadata, dict):
        # Profiling / dummy run: nothing to index.
        return
    md = attn_metadata[k_cache_prefix]
    assert isinstance(md, DeepseekV32IndexerMetadata)
    num_tokens = md.slot_mapping.shape[0]
    cache = kv_cache.view(-1, _ROW_DIM)
    block_size = kv_cache.shape[1]

    # 1) insert this step's rows.
    BLOCK_T = 64
    _insert_rows_kernel[(triton.cdiv(num_tokens, BLOCK_T),)](
        packed[:num_tokens], cache, md.slot_mapping, num_tokens,
        ROW=_ROW_DIM, BLOCK_T=BLOCK_T,
    )
    topk_indices_buffer[: q.shape[0]] = -1

    # 2) prefill chunks.
    if md.num_prefills > 0:
        assert md.prefill is not None
        for chunk in md.prefill.chunks:
            R = chunk.token_end - chunk.token_start
            if R <= 0:
                continue
            visible = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).to(torch.int32)
            row_req = chunk.token_to_seq[:R].to(torch.int32)
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
    ) -> None:
        super().__init__()
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
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim).to(torch.bfloat16)
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
