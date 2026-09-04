# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

from typing import ClassVar, cast

import torch

import vllm.envs as envs
from torch import nn

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm._custom_ops import reshape_and_cache_flash
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionType,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "turboquant_k8v4",
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str.startswith("turboquant"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
            return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)
        return FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(AttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel.

    Standalone impl: FlashAttention contributes no kernels here. Prefill and
    decode both run ``forward_qsa``'s Triton sparse kernel (which dequantizes
    an FP8 cache in-register on any SM), and cache writes go through the
    generic ``reshape_and_cache_flash`` CUDA op.
    """

    supports_dcp: bool = False
    supports_pcp: bool = False

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
        tq_config=None,
    ) -> None:
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if kv_cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
            "turboquant_k8v4",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8, or TQ-k8v4 main KV cache"
            )
        if kv_cache_dtype == "turboquant_k8v4" and tq_config is None:
            raise ValueError("TQ main KV requires a TurboQuantConfig")
        self.tq_config = tq_config
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = (
            (-1, -1) if sliding_window is None else (sliding_window - 1, 0)
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.sinks = sinks
        self.supports_quant_query_input = False

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError(
            "Qwen4Exp QSA attention is dispatched through forward_qsa, "
            "never the generic forward path"
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.tq_config is not None:
            # TQ slab is [blocks, block_size, H, slot]: fused fp8-key cast +
            # 4-bit value quantization, one program per (token, head).
            from vllm.v1.attention.ops.triton_turboquant_store import (
                triton_turboquant_store,
            )

            empty = torch.empty(0, dtype=torch.float32, device=key.device)
            triton_turboquant_store(
                key,
                value,
                kv_cache,
                slot_mapping,
                empty,  # PiT: unused on the fp8-key path
                empty,  # midpoints: unused on the fp8-key path
                mse_bits=self.tq_config.mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.value_quant_bits,
                key_fp8=True,
            )
            return
        # Scatter-write into the paged cache; (B, H, N, 2*D) ->
        # ((B, N, H, D), (B, N, H, D)). reshape_and_cache_flash converts to
        # the cache dtype (including fp8, with the layer's unit scales) and
        # sizes itself from slot_mapping, so padded key/value are fine.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]

        from .ops.qsa import qsa_sparse_paged_attention

        if self.tq_config is not None:
            # TQ slab arrives already in [blocks, block_size, H, slot] order;
            # the kernel decodes keys and values in-register.
            if query.dtype != torch.bfloat16 or kv_cache.dtype != torch.uint8:
                raise NotImplementedError(
                    "Qwen4Exp QSA TQ requires BF16 Q and a uint8 slab"
                )
            qsa_sparse_paged_attention(
                query[:num_tokens],
                kv_cache,
                kv_cache,
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                output[:num_tokens],
                tq_slot_size=self.tq_config.slot_size_aligned,
            )
            return output

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if query.dtype != torch.bfloat16 or key_cache.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q and BF16/FP8 KV")

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8 (e4m3) main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.tq_config = None
        # The profile selects FP8 main KV through the engine's own
        # kv_cache_dtype ("fp8" / "fp8_e4m3"); VLLM_QWEN4_EXP_FP8_MAIN_KV is
        # the layer-scoped diagnostic switch that does the same without
        # touching cache_config.
        fp8_main_kv = envs.VLLM_QWEN4_EXP_FP8_MAIN_KV or cache_config.cache_dtype in (
            "fp8",
            "fp8_e4m3",
        )
        if envs.VLLM_QWEN4_EXP_TQ_MAIN_KV:
            # Model-scoped TurboQuant k8v4 for the main QSA KV only: fp8
            # keys (e4b15 on Ampere) plus 4-bit uniform values with per-slot
            # fp16 scale/zero, ~2.64x smaller than bf16. Stored as a uint8
            # slab; the sparse-attention kernel dequantizes in-register.
            if fp8_main_kv:
                raise ValueError(
                    "VLLM_QWEN4_EXP_TQ_MAIN_KV is mutually exclusive with an FP8"
                    " main KV (kv_cache_dtype fp8 or VLLM_QWEN4_EXP_FP8_MAIN_KV)"
                )
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            self.kv_cache_dtype = "turboquant_k8v4"
            self.kv_cache_torch_dtype = torch.uint8
            self.tq_config = TurboQuantConfig.from_cache_dtype(
                "turboquant_k8v4", int(config.head_dim)
            )
        elif fp8_main_kv:
            # Model-scoped FP8 for the main QSA KV only: the indexer's
            # compressed cache, the raw-key ring, and the GDN state keep
            # their own dtypes. The sparse-attention kernel dequantizes
            # in-register (unit scale), halving the dominant KV-slab term.
            # Storage is allocated as e4m3fn directly; the generic string
            # table maps "fp8" to uint8, which is the view dtype the kernel
            # wrapper applies, not the allocation dtype.
            self.kv_cache_dtype = "fp8"
            self.kv_cache_torch_dtype = torch.float8_e4m3fn
        else:
            self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
                self.kv_cache_dtype, model_config
            )
        if self.kv_cache_torch_dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.uint8,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16, FP8, or TQ-k8v4 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
            tq_config=self.tq_config,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if self.tq_config is not None:
            from vllm.v1.kv_cache_interface import TQFullAttentionSpec

            return TQFullAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                head_size_v=self.head_dim,
                dtype=self.kv_cache_torch_dtype,
                tq_slot_size=self.tq_config.slot_size_aligned,
                tq_cache_dtype="turboquant_k8v4",
            )
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def qwen4_exp_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )


def qwen4_exp_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, layer_name


direct_register_custom_op(
    op_name="qwen4_exp_qsa_with_output",
    op_func=qwen4_exp_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen4_exp_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "qwen4_exp_qsa_with_output",
]
