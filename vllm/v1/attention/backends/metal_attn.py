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
# is correct on the SDPA path, just slower.
_PAGED_HEAD_SIZES = (64, 128)


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
    seq_lens_cpu: torch.Tensor
    # Device-side, for the cache write and the fused decode.
    block_table: torch.Tensor
    seq_lens_gpu: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MetalAttentionMetadata:
        m = common_attn_metadata
        causal = m.causal if isinstance(m.causal, bool) else True
        return MetalAttentionMetadata(
            num_actual_tokens=m.num_actual_tokens,
            num_reqs=m.num_reqs,
            max_query_len=m.max_query_len,
            query_start_loc_cpu=m.query_start_loc_cpu.to(dtype=torch.int32),
            seq_lens_cpu=m.seq_lens.to("cpu", dtype=torch.int32),
            block_table=m.block_table_tensor.to(torch.int32),
            seq_lens_gpu=m.seq_lens.to(torch.int32),
            slot_mapping=m.slot_mapping,
            causal=causal,
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

    def _fused_decode_applies(
        self, metadata: MetalAttentionMetadata, num_tokens: int
    ) -> bool:
        return (
            metadata.max_query_len == 1
            and num_tokens == metadata.num_reqs
            and self.head_size in _PAGED_HEAD_SIZES
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

        # Write the new K/V into the persistent paged cache.
        #
        # Per-token advanced indexing, deliberately: index_copy_ over the
        # flattened multi-million-row view is O(cache) on MPS and stalls,
        # while this is O(num_tokens).
        if self.kv_sharing_target_layer_name is None:
            slot = attn_metadata.slot_mapping[:num_tokens].to(torch.long)
            block_idx = slot // block_size
            block_off = slot % block_size
            key_cache[block_idx, block_off] = key[:num_tokens]
            value_cache[block_idx, block_off] = value[:num_tokens]

        out = output.view(num_tokens, num_heads, head_size)

        if self._fused_decode_applies(attn_metadata, num_tokens):
            from vllm.quixicore import quixicore_ops

            if quixicore_ops.is_available():
                out.copy_(
                    quixicore_ops.paged_attention(
                        query[:num_tokens].contiguous(),
                        key_cache,
                        value_cache,
                        attn_metadata.block_table,
                        attn_metadata.seq_lens_gpu,
                        self.scale,
                        self.sliding_window or 0,
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
                expanded_block_table = attn_metadata.block_table.repeat_interleave(
                    q_len, dim=0
                )
                # Global (unwindowed) layers at length: the multi-query
                # kernel shares each K/V read across the m rows (measured
                # 3.9x at 9.9k ctx; parity exact). Windowed layers keep the
                # expansion path (scan capped by the window -- a wash).
                use_mq = (
                    self.sliding_window is None
                    and attn_metadata.num_reqs == 1
                    and int(seq_lens.max().item()) > 1024
                )
                if use_mq:
                    if not getattr(self, "_mq_logged", False):
                        self._mq_logged = True
                        logger.info(
                            "multi-query verify attention engaged (ctx=%d)",
                            int(seq_lens.max().item()),
                        )
                    out.copy_(
                        quixicore_ops.paged_attention_verify(
                            query[:num_tokens].contiguous(),
                            key_cache,
                            value_cache,
                            attn_metadata.block_table,
                            seq_lens,
                            self.scale,
                            0,
                        )
                    )
                    return output
                out.copy_(
                    quixicore_ops.paged_attention(
                        query[:num_tokens].contiguous(),
                        key_cache,
                        value_cache,
                        expanded_block_table,
                        expanded_seq_lens,
                        self.scale,
                        self.sliding_window or 0,
                    )
                )
                return output

        self._sdpa_forward(
            query, out, attn_metadata, key_cache, value_cache, num_blocks, block_size
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
    ) -> None:
        """Per-request attention over keys gathered from the paged cache."""
        starts = metadata.query_start_loc_cpu
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

            if self.use_native_range_gather:
                # MPS index_select uses signed 32-bit element offsets for this
                # strided source. A hybrid cache page beyond 2^31 elements is
                # therefore read from the wrong address. The native gather
                # carries the physical block stride and all cache arithmetic
                # in 64 bits; it also avoids materializing unused rows in the
                # first/last page of a sliding-window request.
                keys, values = quixicore_ops.kv_cache_gather_range(
                    key_cache,
                    value_cache,
                    metadata.block_table[req],
                    kv_start,
                    seq_len - kv_start,
                )
            else:
                first_block = kv_start // block_size
                blocks = (
                    metadata.block_table[req, first_block:num_req_blocks]
                    .to(torch.long)
                    # The profile run allocates a small dummy cache whose block
                    # table can point past it. That output is discarded; a real
                    # run never clamps.
                    .clamp_(0, num_blocks - 1)
                )
                row_start = kv_start - first_block * block_size
                row_end = seq_len - first_block * block_size
                page_elems = block_size * num_kv_heads * head_size
                if key_cache.stride(0) == 2 * page_elems:
                    pages_view = key_cache.as_strided(
                        (num_blocks, 2, block_size, num_kv_heads, head_size),
                        (2 * page_elems, page_elems, *key_cache.stride()[1:]),
                        key_cache.storage_offset(),
                    )
                    pages = pages_view.index_select(0, blocks)
                    keys = pages[:, 0]
                    values = pages[:, 1]
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
