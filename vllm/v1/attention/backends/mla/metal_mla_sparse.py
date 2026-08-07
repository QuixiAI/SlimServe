# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA (DSA) backend for Apple Metal.

The Apple counterpart of `quixicore_mla_sparse` (Ampere) and
`rocm_aiter_mla_sparse` (gfx942). Like both of those it has no dense-MHA
prefill path, so serving requires
`--attention-config '{"sparse_mla_force_mqa": true}'`.

Status: **not servable yet, and the gap is kernels rather than plumbing.**
This class exists so the platform's backend selection resolves and fails with
an accurate message instead of an ImportError, and so the eventual
implementation has the surface already agreed.

What is missing, precisely. The CUDA path calls four ops that QuixiCore-Metal
does not have compiled:

    mla_decode_fp8_sparse_glm         packed 576-wide fp8 page, one k_scale
    mla_decode_fp8_sparse_glm_splitq  reads q from its two source buffers
    mla_decode_bf16_sparse_glm        bf16 latent cache
    sparse_topk_tlen                  effective top-k length per row

Metal has the *generic* `mla_decode_fp8_sparse`, which is not a drop-in: it
takes separate `data_cache` and `scale_cache` tensors, while vLLM stores the
latent as a single `(num_blocks, block_size, 576[+scale])` page. Closing this
needs one of:

1. A cache-layout adapter that presents the packed page to the kernel as two
   views. Free if the packing allows a stride trick, a per-step copy if not --
   and a per-step copy over the whole latent would cost more than it saves.
2. GLM-shaped Metal kernels mirroring the `_glm` CUDA variants, which is the
   route the CUDA side took for exactly this reason.

`sparse_topk_tlen` is the easy one: it is a last-valid-index reduction and a
torch fallback is fine until it shows up in a profile.
"""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    MLAAttentionImpl,
    MultipleOf,
)

logger = init_logger(__name__)

_UNIMPLEMENTED = (
    "Sparse MLA is not implemented on Apple Metal yet. QuixiCore-Metal has the "
    "generic mla_decode_fp8_sparse kernel, but not the GLM-shaped variants the "
    "serving path calls (mla_decode_fp8_sparse_glm{,_splitq}, "
    "mla_decode_bf16_sparse_glm), and the generic kernel takes separate "
    "data/scale caches rather than vLLM's single packed latent page. See "
    "vllm/v1/attention/backends/mla/metal_mla_sparse.py for the two ways to "
    "close that gap. GLM-5.2-Vision and DeepSeek-V4-Flash therefore do not "
    "serve on Metal; the dense/GQA path (METAL_ATTN) is unaffected."
)


class MetalMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "METAL_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["MetalMLASparseImpl"]:
        return MetalMLASparseImpl

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError(_UNIMPLEMENTED)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class MetalMLASparseImpl(MLAAttentionImpl):
    is_sparse = True

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(_UNIMPLEMENTED)

    def forward_mqa(self, *args, **kwargs):
        raise NotImplementedError(_UNIMPLEMENTED)
