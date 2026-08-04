# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Absorbed MLA decode on gfx942, in HIP.

AITER's MLA decode is hand-written assembly shipped as pre-linked code objects
with the query head count baked into the binary, so it runs only multiples and
divisors of 16 -- Kimi K3 at TP8 gives 12 per rank and had nothing to run on.
This backend calls a HIP kernel that takes the head count as a grid dimension
instead, so every tensor-parallel size that divides K3's 96 heads works.
"""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)

logger = init_logger(__name__)


def _hip_mla_available() -> bool:
    try:
        import vllm._quixicore_C as qc
    except ImportError:
        return False
    return hasattr(qc, "mla_decode_fwd")


class HipMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


class HipMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # The kernel reads the latent cache in its stored dtype; fp8 rows would
    # need dequant-on-read, which it does not do yet.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "HIP_MLA"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The kernel pages with a plain divide/modulo, so any block size works
        # -- including the inflated page a hybrid model like K3 ends up with.
        return [MultipleOf(1)]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # wave64 and gfx942-tuned; do not claim RDNA.
        return _hip_mla_available()

    @staticmethod
    def get_impl_cls() -> type["HipMLAImpl"]:
        return HipMLAImpl

    @staticmethod
    def get_builder_cls() -> type["HipMLAMetadataBuilder"]:
        return HipMLAMetadataBuilder


class HipMLAImpl(MLACommonImpl[MLACommonMetadata]):
    # The kernel normalises inside the split-K reduce and does not hand back a
    # log-sum-exp, so it cannot feed decode context parallelism.
    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "HipMLAImpl does not support alibi_slopes, sliding_window or "
                "logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("HipMLAImpl supports decoder self-attention only")
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("HipMLAImpl does not support fp8 KV cache")

        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)

        o = torch.empty(
            q.shape[0], q.shape[1], self.kv_lora_rank, dtype=q.dtype, device=q.device
        )

        import vllm._quixicore_C as qc

        qc.mla_decode_fwd(
            q.contiguous(),
            kv_c_and_k_pe_cache,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            o,
            self.scale,
            attn_metadata.max_seq_len,
        )
        return o, None
