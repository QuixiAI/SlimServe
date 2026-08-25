# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense/GQA attention on the Apple GPU.

Two paths, chosen per forward:

- Pure decode -- every request contributing exactly one query token -- goes to
  one fused QuixiCore paged-attention dispatch for the whole batch.
- Everything else (prefill, and any mixed batch) falls back to a per-request
  ``scaled_dot_product_attention`` over keys gathered out of the paged cache.
  That loop is the known throughput ceiling on this platform; it is correct,
  and it is what a batched Metal kernel would replace.

The KV layout is chosen so ``kv_cache[0]`` and ``kv_cache[1]`` are each exactly
the ``(num_blocks, block_size, H_kv, D)`` tensor the paged kernel consumes, with
no per-step repacking.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.quixicore import quixicore_ops
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Head sizes the fused paged decode kernel is instantiated for. Anything else
# is correct on the SDPA path, just slower. 256 (Qwen3.8 full attn) landed
# with the N4 census fix VIA THE SPLIT-K partition/reduce pair — the
# monolithic one-simdgroup-per-head walk under-occupies at decode widths
# (routed and measured both here and on main; ~15% slower end-to-end).
# VLLM_QC_PA256=0 drops back to the SDPA gather (null-check / kill switch —
# the kernel's online softmax rounds differently than the torch-SDPA path,
# so shas roll when it turns on).
_PAGED_HEAD_SIZES = (64, 128)
if os.environ.get("VLLM_QC_PA256", "1") != "0":
    _PAGED_HEAD_SIZES = (64, 128, 256)

# Pure-prefill batches read the (row-exact there) CPU bound in the causal
# SDPA loop instead of materializing seq_lens via a queue-draining D2H.
# =0 restores the unconditional exact-lens pull.
_SDPA_PREFILL_BOUND = os.environ.get("VLLM_QC_SDPA_PREFILL_BOUND", "1") != "0"
# One-dispatch paged KV insert (kv_cache_scatter kernel) instead of the
# .to + index math + two advanced-indexing scatters per attention layer.
# Pure copies, identical destinations. =0 restores the torch path.
_KV_SCATTER = os.environ.get("VLLM_QC_KV_SCATTER", "1") != "0"
_KV_SCATTER_SYMBOL: bool | None = None


def _kv_scatter_symbol() -> bool:
    global _KV_SCATTER_SYMBOL
    if _KV_SCATTER_SYMBOL is None:
        try:
            import vllm._quixicore_C as qc

            _KV_SCATTER_SYMBOL = hasattr(qc, "qc_kv_cache_scatter")
        except ImportError:
            _KV_SCATTER_SYMBOL = False
    return _KV_SCATTER_SYMBOL


# Uniform non-causal block decode (DFlash draft-block attention) routed to
# the expanded paged kernel instead of the per-request SDPA python loop.
# REJECTED as default (UPDATE 47): kernel-vs-SDPA numerics shift the
# 16-way candidate/path selection enough to cost acceptance (c1 -7.7%,
# 2500x64 -4.1% + trajectory fork), and the kernel is no faster at draft
# shapes. Kept opt-in (=1) as a diagnostic.
_DRAFT_BLOCK_PA = os.environ.get("VLLM_QC_DRAFT_BLOCK_PA", "0") == "1"


class MetalAttentionBackend(AttentionBackend):
    # forward() writes K/V into the paged cache itself, so vLLM must not also
    # call the separate unified KV-cache update op.
    forward_includes_kv_cache_update: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ]

    @staticmethod
    def get_name() -> str:
        return "METAL_ATTN"

    @staticmethod
    def get_impl_cls() -> type["MetalAttentionImpl"]:
        return MetalAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["MetalAttentionMetadataBuilder"]:
        return MetalAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Dim 0 selects key(0)/value(1) so each half stays contiguous.
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # Physical layout [num_blocks, 2, block_size, H, D]: block b's K and V
        # both live inside physical page b. The hybrid KV manager shares one
        # raw tensor between attention and mamba layers under exactly that
        # page-identity contract; the previous (default) K-first physical
        # layout spread block b over half-pages b/2 and N/2+b/2, so a mamba
        # state write to its own (validly allocated) page s silently
        # overwrote attention K blocks 2s/2s+1.
        #
        # The layered variant is intentionally NOT provided (same tuple is
        # returned), which keeps indexes_kv_by_block_stride() False and the
        # packed/padded-page machinery unchanged.
        return (1, 0, 2, 3, 4)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 112, 128, 192, 256]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class MetalAttentionMetadata:
    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    # CPU copies, used to slice the batch per request without a device sync
    # inside the loop.
    query_start_loc_cpu: torch.Tensor
    # Device-side, for the cache write and the fused decode.
    block_table: torch.Tensor
    seq_lens_gpu: torch.Tensor
    slot_mapping: torch.Tensor
    # Host-side bound on the batch's longest context (>= exact), for kernel
    # routing and split-K sizing without a device sync.
    seq_lens_cpu_max: int = 0
    # Per-request CPU upper bound on seq_lens (>= exact; precise for prefill
    # rows and outside async spec decode). None when unavailable.
    seq_lens_cpu_bound: torch.Tensor | None = None
    causal: bool = True
    # True when every batch row is mid-prefill: the CPU bound then equals the
    # exact lens row-for-row (the async-scheduling mirror only diverges on
    # spec-decode rows), so the causal SDPA loop can read the bound instead
    # of forcing the queue-draining D2H materialization below.
    bound_exact: bool = False
    # Exact host seq_lens, materialized LAZILY: the D2H copy blocks on every
    # queued GPU op, so the paged decode paths and the bound-mode draft SDPA
    # must never touch it. Only the exact (causal/prefill) SDPA loop does.
    _seq_lens_cpu: torch.Tensor | None = None

    @property
    def seq_lens_cpu(self) -> torch.Tensor:
        # Memo is safe only because this builder has NO steady_decode_update:
        # the steady-reuse path in attn_utils rebuilds this metadata object
        # every step. If in-place steady refresh is ever added here, this
        # memo must be invalidated by it or it will freeze step-1 lens.
        if self._seq_lens_cpu is None:
            self._seq_lens_cpu = self.seq_lens_gpu.to("cpu")
        return self._seq_lens_cpu


class MetalAttentionMetadataBuilder(AttentionMetadataBuilder[MetalAttentionMetadata]):
    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = vllm_config.cache_config.block_size
        # With serial (non-async) scheduling the scheduler has consumed the
        # previous step's acceptance before scheduling this one, so the
        # CPU-side seq_lens_cpu_upper_bound (committed + scheduled) is
        # EXACT and the device seq_lens need never be pulled to the host.
        # That pull was the step's hidden pipeline drain: it waited out the
        # whole previous GPU step (~40 ms) inside build(). Under async
        # scheduling the upper bound can overshoot (it assumes full draft
        # acceptance), and an overshot length would attend stale KV rows,
        # so the synced copy stays for that mode.
        self._exact_cpu_seq_lens = not bool(
            vllm_config.scheduler_config.async_scheduling
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MetalAttentionMetadata:
        m = common_attn_metadata
        causal = m.causal if isinstance(m.causal, bool) else True
        # Sync-free host bound when the caller provides one: the eager
        # `seq_lens.to("cpu")` blocks on the whole in-flight GPU queue (60
        # ms/step at the draft build under async scheduling). Exact host lens
        # are materialized lazily by the exact SDPA loop only.
        bound = m.seq_lens_cpu_upper_bound
        exact = None
        if bound is not None and bound.numel() > 0:
            bound = bound.to(dtype=torch.int32)
            seq_lens_cpu_max = int(bound.max())
        else:
            exact = m.seq_lens.to("cpu", dtype=torch.int32)
            seq_lens_cpu_max = int(exact.max()) if exact.numel() else 0
        bound_exact = False
        if _SDPA_PREFILL_BOUND and causal and bound is not None:
            isp = m.is_prefilling
            bound_exact = (
                isp is not None
                and isp.device.type == "cpu"
                and isp.numel() >= m.num_reqs
                and bool(isp[: m.num_reqs].all())
            )
        return MetalAttentionMetadata(
            num_actual_tokens=m.num_actual_tokens,
            num_reqs=m.num_reqs,
            max_query_len=m.max_query_len,
            query_start_loc_cpu=m.query_start_loc_cpu.to(dtype=torch.int32),
            block_table=m.block_table_tensor.to(torch.int32),
            seq_lens_gpu=m.seq_lens.to(torch.int32),
            slot_mapping=m.slot_mapping,
            seq_lens_cpu_max=seq_lens_cpu_max,
            seq_lens_cpu_bound=bound,
            causal=causal,
            bound_exact=bound_exact,
            _seq_lens_cpu=exact,
        )


class MetalAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.use_native_range_gather = quixicore_ops.has("kv_cache_gather_range")

        if sinks is not None:
            raise NotImplementedError("Attention sinks require TurboQuant on Metal.")
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi has no Metal attention path.")
        # Sliding windows ride the paged-attention kernel's `window` argument
        # on the decode path and a banded mask on the SDPA fallback.
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"Metal attention is decoder-only, got {attn_type}."
            )

    # D=256 batch-1 crossover (measured, UPDATE 20): the split-K kernel's
    # per-call encoder/scheduling latency loses to the in-stream SDPA path
    # at short context (c1 1000x256: 10.02 vs 10.74 tok/s) and reaches
    # parity around ctx 2500 (2.93 vs 2.91); batch >= 2 amortizes the fixed
    # cost and the kernel wins big (c4 +10%, c8 +17%). Route batch-1 short
    # context to SDPA, everything else to the kernel.
    _PA256_MIN_CTX_BATCH1 = 2048

    def _pa256_route_ok(self, metadata: MetalAttentionMetadata) -> bool:
        if self.head_size != 256:
            return True
        return (
            metadata.num_reqs >= 2
            or metadata.seq_lens_cpu_max >= self._PA256_MIN_CTX_BATCH1
        )

    def _fused_decode_applies(
        self, metadata: MetalAttentionMetadata, num_tokens: int
    ) -> bool:
        return (
            metadata.max_query_len == 1
            and num_tokens == metadata.num_reqs
            and self.head_size in _PAGED_HEAD_SIZES
            and self._pa256_route_ok(metadata)
        )

    # Largest uniform query length routed through the fused kernel by batch
    # expansion. Sized for speculative decoding (DFlash verify is k+1 = 17
    # queries per request, the draft block 16); long prefill stays on the
    # matmul-based SDPA path, where each key is read once instead of once
    # per query row.
    _EXPAND_MAX_QUERY_LEN = 32

    def _expanded_decode_applies(
        self, metadata: MetalAttentionMetadata, num_tokens: int
    ) -> bool:
        q_len = metadata.max_query_len
        if not (
            metadata.causal
            and 1 < q_len <= self._EXPAND_MAX_QUERY_LEN
            and num_tokens == metadata.num_reqs * q_len
            and self.head_size in _PAGED_HEAD_SIZES
        ):
            return False
        # Cached-prompt A/B at 1734-token context: expansion 16.1 tok/s vs
        # SDPA gather 14.55 -- expansion wins at least through ~2k. The
        # knob exists for larger-context research (the per-row KV re-read
        # grows linearly and must cross over somewhere).
        import os

        ctx_max = int(os.environ.get("VLLM_EXPAND_CTX_MAX", "100000"))
        return int(metadata.seq_lens_gpu.max().item()) <= ctx_max

    def _expanded_block_decode_applies(
        self, metadata: MetalAttentionMetadata, num_tokens: int
    ) -> bool:
        """Non-causal twin: DFlash draft-block attention. Every query token
        of a request sees the same key range, so the expansion uses a flat
        seq_len per pseudo-request (no causal step offsets)."""
        q_len = metadata.max_query_len
        return (
            _DRAFT_BLOCK_PA
            and not metadata.causal
            and 1 < q_len <= self._EXPAND_MAX_QUERY_LEN
            and num_tokens == metadata.num_reqs * q_len
            and self.head_size in _PAGED_HEAD_SIZES
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Out-of-forward K/V insert (DFlash context precompute). Same
        per-token advanced-indexing write as forward(); PAD_SLOT_ID rows are
        clamped onto block 0, the null block, whose contents are never read
        (no host sync on a data-dependent mask)."""
        if self.kv_sharing_target_layer_name is not None:
            return
        _, num_blocks, block_size, _, _ = kv_cache.shape
        slot = slot_mapping.to(torch.long).clamp_min(0)
        block_idx = slot // block_size
        block_off = slot % block_size
        num_tokens = slot.shape[0]
        dense = kv_cache.transpose(0, 1)
        if dense.is_contiguous():
            dense_kv = dense.reshape(
                2 * num_blocks, block_size, self.num_kv_heads, self.head_size
            )
            dense_kv[block_idx * 2, block_off] = key[:num_tokens]
            dense_kv[block_idx * 2 + 1, block_off] = value[:num_tokens]
        else:
            kv_cache[0][block_idx, block_off] = key[:num_tokens]
            kv_cache[1][block_idx, block_off] = value[:num_tokens]

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The profile and warmup runs pass no metadata.
        if attn_metadata is None:
            return output
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Fused output quantization has no Metal path.")

        num_tokens = attn_metadata.num_actual_tokens
        num_heads = self.num_heads
        head_size = self.head_size

        _, num_blocks, block_size, _, _ = kv_cache.shape
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        # Page-local physical layout [num_blocks, 2, block_size, H, D]
        # (get_kv_cache_stride_order): block b's K and V are dense blocks 2b
        # and 2b+1 of the raw region. Every hot-path read and write goes
        # through this contiguous dense view with transformed block indices —
        # advanced indexing over the strided per-K/V views falls off the MPS
        # fast path — and the fused kernel gets it with a doubled block table
        # and a one-block-shifted alias for V. No repacking, all views.
        dense = kv_cache.transpose(0, 1)
        page_local = dense.is_contiguous()
        dense_kv = (
            dense.reshape(2 * num_blocks, block_size, self.num_kv_heads, head_size)
            if page_local
            else None
        )

        # Write the new K/V into the persistent paged cache.
        #
        # Per-token advanced indexing, deliberately: index_copy_ over the
        # flattened multi-million-row view is O(cache) on MPS and stalls,
        # while this is O(num_tokens).
        if self.kv_sharing_target_layer_name is None:
            k_src = key[:num_tokens]
            v_src = value[:num_tokens]
            scattered = False
            if _KV_SCATTER and k_src.is_contiguous() and v_src.is_contiguous():
                from vllm.quixicore import quixicore_ops

                if quixicore_ops.is_available() and _kv_scatter_symbol():
                    # One dispatch replaces the .to + index math + two
                    # advanced-indexing scatters. Pure copies, identical
                    # destination rows (slot<0 never occurs on this path).
                    slot = attn_metadata.slot_mapping[:num_tokens]
                    if slot.dtype != torch.long:
                        slot = slot.to(torch.long)
                    slot = slot.contiguous()
                    if dense_kv is not None:
                        quixicore_ops.qc_kv_cache_scatter(
                            k_src,
                            v_src,
                            slot,
                            dense_kv,
                            dense_kv[1:],
                            self.num_kv_heads,
                            head_size,
                            block_size,
                            2,
                        )
                    else:
                        quixicore_ops.qc_kv_cache_scatter(
                            k_src,
                            v_src,
                            slot,
                            key_cache,
                            value_cache,
                            self.num_kv_heads,
                            head_size,
                            block_size,
                            1,
                        )
                    scattered = True
            if not scattered:
                slot = attn_metadata.slot_mapping[:num_tokens].to(torch.long)
                block_idx = slot // block_size
                block_off = slot % block_size
                if dense_kv is not None:
                    dense_kv[block_idx * 2, block_off] = key[:num_tokens]
                    dense_kv[block_idx * 2 + 1, block_off] = value[:num_tokens]
                else:
                    key_cache[block_idx, block_off] = key[:num_tokens]
                    value_cache[block_idx, block_off] = value[:num_tokens]

        out = output.view(num_tokens, num_heads, head_size)

        if page_local:
            kc_kernel = dense_kv
            vc_kernel = dense_kv[1:]
            kernel_block_table = attn_metadata.block_table * 2
        else:
            kc_kernel = key_cache
            vc_kernel = value_cache
            kernel_block_table = attn_metadata.block_table

        if self._fused_decode_applies(attn_metadata, num_tokens):
            from vllm.quixicore import quixicore_ops

            if quixicore_ops.is_available():
                out.copy_(
                    quixicore_ops.paged_attention(
                        query[:num_tokens].contiguous(),
                        kc_kernel,
                        vc_kernel,
                        kernel_block_table,
                        attn_metadata.seq_lens_gpu,
                        self.scale,
                        self.sliding_window or 0,
                        # host-side batch bound, sizes the D=256 split-K
                        # partitions without a device sync
                        attn_metadata.seq_lens_cpu_max,
                    )
                )
                return output

        if self._expanded_decode_applies(attn_metadata, num_tokens):
            from vllm.quixicore import quixicore_ops

            if quixicore_ops.is_available():
                # Uniform multi-query causal decode (speculative verify and
                # DFlash draft blocks): expand each request into q_len
                # pseudo-requests sharing its block table. Query token t of a
                # request sees seq_len - q_len + t + 1 positions, so causal
                # semantics -- and the sliding window, which the kernel clamps
                # per pseudo-request -- are exact. One dispatch replaces the
                # per-request SDPA gather loop.
                q_len = attn_metadata.max_query_len
                seq_lens = attn_metadata.seq_lens_gpu
                steps = torch.arange(
                    1 - q_len, 1, device=seq_lens.device, dtype=seq_lens.dtype
                )
                expanded_seq_lens = (
                    seq_lens.unsqueeze(1) + steps.unsqueeze(0)
                ).flatten()
                expanded_block_table = kernel_block_table.repeat_interleave(
                    q_len, dim=0
                )
                # Global (unwindowed) layers at length: the multi-query
                # kernel shares each K/V read across the m rows (measured
                # 3.9x at 9.9k ctx; parity exact). Windowed layers keep the
                # expansion path (scan capped by the window -- a wash).
                use_mq = (
                    self.sliding_window is None
                    and attn_metadata.num_reqs == 1
                    # The verify kernel does not walk strided caches yet;
                    # head_dim 256 implies the hybrid pool's strided views,
                    # so it stays on the expansion path (which does).
                    and self.head_size != 256
                    and int(seq_lens.max().item()) > 1024
                )
                if use_mq:
                    if not getattr(self, "_mq_logged", False):
                        self._mq_logged = True
                        logger.info(
                            "multi-query verify attention engaged (ctx=%d)",
                            int(seq_lens.max().item()),
                        )
                    # Page-local caches: key_cache/value_cache are strided
                    # views the verify kernel's contiguity contract rejects;
                    # the kernel-facing pair (dense_kv and its offset view)
                    # is contiguous under the doubled block table.
                    out.copy_(
                        quixicore_ops.paged_attention_verify(
                            query[:num_tokens].contiguous(),
                            kc_kernel,
                            vc_kernel,
                            kernel_block_table,
                            seq_lens,
                            self.scale,
                            0,
                        )
                    )
                    return output
                out.copy_(
                    quixicore_ops.paged_attention(
                        query[:num_tokens].contiguous(),
                        kc_kernel,
                        vc_kernel,
                        expanded_block_table,
                        expanded_seq_lens,
                        self.scale,
                        self.sliding_window or 0,
                        attn_metadata.seq_lens_cpu_max,
                    )
                )
                return output

        if self._expanded_block_decode_applies(attn_metadata, num_tokens):
            from vllm.quixicore import quixicore_ops

            if quixicore_ops.is_available():
                # Uniform non-causal block decode (DFlash draft blocks):
                # every query token of a request sees the same key range
                # [kv_start, seq_len), so each pseudo-request carries the
                # flat seq_len. The SDPA loop anchors the sliding window at
                # the BLOCK START (kv_start = seq_len - q_len - W + 1) while
                # the kernel clamps per pseudo-request to
                # [context_len - window, context_len) — widening the window
                # by q_len - 1 makes the ranges identical. One dispatch
                # replaces the per-request SDPA gather loop; reads exact
                # seq_lens_gpu, so it is also sync-free where the SDPA
                # bound-mode needed the GPU validity mask.
                q_len = attn_metadata.max_query_len
                seq_lens = attn_metadata.seq_lens_gpu
                # repeat_interleave with a scalar count: static output shape
                # and a contiguous result (reshape of an expand view is not
                # contiguous on MPS, and the host op requires it).
                expanded_seq_lens = seq_lens.repeat_interleave(q_len)
                expanded_block_table = kernel_block_table.repeat_interleave(
                    q_len, dim=0
                )
                window = self.sliding_window + q_len - 1 if self.sliding_window else 0
                out.copy_(
                    quixicore_ops.paged_attention(
                        query[:num_tokens].contiguous(),
                        kc_kernel,
                        vc_kernel,
                        expanded_block_table,
                        expanded_seq_lens,
                        self.scale,
                        window,
                        attn_metadata.seq_lens_cpu_max,
                    )
                )
                return output

        self._sdpa_forward(
            query,
            out,
            attn_metadata,
            key_cache,
            value_cache,
            num_blocks,
            block_size,
            dense_kv=dense_kv,
        )
        return output

    def _sdpa_forward(
        self,
        query: torch.Tensor,
        out: torch.Tensor,
        metadata: MetalAttentionMetadata,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_blocks: int,
        block_size: int,
        dense_kv: torch.Tensor | None = None,
    ) -> None:
        """Per-request attention over keys gathered from the paged cache."""
        starts = metadata.query_start_loc_cpu
        # Bound mode (sync-free, non-causal draft groups only): slice by the
        # CPU upper bound and enforce the exact key range with a GPU mask
        # built from seq_lens_gpu — the exact host lens would require a D2H
        # that drains the whole in-flight queue. Causal groups (target
        # prefill SDPA) keep the exact path: their bound rows are precise, and
        # the loop below is byte-identical to the pre-bound-mode code there.
        bound_mode = not metadata.causal and metadata.seq_lens_cpu_bound is not None
        if bound_mode:
            seq_lens = metadata.seq_lens_cpu_bound
        elif metadata.bound_exact and metadata.seq_lens_cpu_bound is not None:
            # Pure-prefill batch: the bound equals the exact lens row-for-row
            # (see bound_exact), so skip the D2H that blocks on every queued
            # prefill chunk (~hundreds of ms per prefill event). Geometry
            # below stays on the exact path — only the lens source changes.
            seq_lens = metadata.seq_lens_cpu_bound
        else:
            seq_lens = metadata.seq_lens_cpu
        num_kv_heads = self.num_kv_heads
        head_size = self.head_size

        for req in range(metadata.num_reqs):
            begin = int(starts[req])
            end = int(starts[req + 1])
            if end == begin:
                continue
            seq_len = int(seq_lens[req])

            num_req_blocks = (seq_len + block_size - 1) // block_size
            query_len_req = end - begin
            # Sliding window: no query row in this request sees a key before
            # kv_start, so blocks before it are never gathered. With the
            # hybrid KV manager those blocks may already be freed; reading
            # them would be stale, not just wasteful.
            kv_start = 0
            if self.sliding_window is not None:
                kv_start = max(0, seq_len - query_len_req - self.sliding_window + 1)
            seq_len = min(seq_len, num_req_blocks * block_size)

            if self.use_native_range_gather and not bound_mode:
                # MPS index_select uses signed 32-bit element offsets for this
                # strided source. A hybrid cache page beyond 2^31 elements is
                # therefore read from the wrong address. The native gather
                # carries the physical block stride and all cache arithmetic
                # in 64 bits; it also avoids materializing unused rows in the
                # first/last page of a sliding-window request. Bound mode
                # keeps the block-window fallback below: its kv_start may
                # exceed the exact window, which the fallback absorbs with
                # the one-block backoff + GPU validity mask.
                keys, values = quixicore_ops.kv_cache_gather_range(
                    key_cache,
                    value_cache,
                    metadata.block_table[req],
                    kv_start,
                    seq_len - kv_start,
                )
            else:
                first_block = kv_start // block_size
                if bound_mode and first_block > 0:
                    # The bound may exceed the exact seq_len by up to the
                    # draft length, which can push the window start past the
                    # exact window's first block. Back off one block and
                    # start at row 0 — the GPU validity mask enforces the
                    # exact range.
                    first_block -= 1
                blocks = (
                    metadata.block_table[req, first_block:num_req_blocks]
                    .to(torch.long)
                    # The profile run allocates a small dummy cache whose
                    # block table can point past it. That output is
                    # discarded; a real run never clamps.
                    .clamp_(0, num_blocks - 1)
                )
                row_start = 0 if bound_mode else kv_start - first_block * block_size
                row_end = seq_len - first_block * block_size

                if dense_kv is not None:
                    keys = dense_kv.index_select(0, blocks * 2)
                    values = dense_kv.index_select(0, blocks * 2 + 1)
                else:
                    keys = key_cache.index_select(0, blocks)
                    values = value_cache.index_select(0, blocks)
                keys = keys.reshape(-1, num_kv_heads, head_size)[row_start:row_end]
                values = values.reshape(-1, num_kv_heads, head_size)[row_start:row_end]

            if self.num_queries_per_kv > 1:
                keys = keys.repeat_interleave(self.num_queries_per_kv, dim=1)
                values = values.repeat_interleave(self.num_queries_per_kv, dim=1)

            query_len = end - begin
            mask = None
            if bound_mode:
                # Exact key visibility from GPU seq_lens (no host sync):
                # keys at [max(0, sl - q - window + 1), sl) are attendable,
                # matching the exact-mode gather range bit-for-bit in
                # semantics. True = attend.
                sl = metadata.seq_lens_gpu[req].to(torch.long)
                key_pos = torch.arange(
                    first_block * block_size + row_start,
                    first_block * block_size + row_end,
                    device=query.device,
                )
                valid = key_pos < sl
                if self.sliding_window is not None:
                    lo = (sl - (query_len_req + self.sliding_window - 1)).clamp_min(0)
                    valid &= key_pos >= lo
                mask = valid[None, None, :]
            if metadata.causal and query_len > 1:
                # Query j sits at absolute position seq_len - query_len + j and
                # may attend every key at or before it, back at most
                # `sliding_window - 1` positions.
                query_pos = torch.arange(
                    seq_len - query_len, seq_len, device=query.device
                )
                key_pos = torch.arange(kv_start, seq_len, device=query.device)
                mask = key_pos[None, :] <= query_pos[:, None]
                if self.sliding_window is not None:
                    mask &= key_pos[None, :] > (
                        query_pos[:, None] - self.sliding_window
                    )
                mask = mask[None]

            attended = F.scaled_dot_product_attention(
                query[begin:end].transpose(0, 1),
                keys.transpose(0, 1),
                values.transpose(0, 1),
                attn_mask=mask,
                scale=self.scale,
            )
            out[begin:end] = attended.transpose(0, 1)
