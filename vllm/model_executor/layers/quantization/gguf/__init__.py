# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-tree GGUF quantization.

Moved here from the out-of-tree `vllm_gguf_plugin` so the kernels and the
Python that drives them live together and can be specialised for the one model
this repo serves. The diffusion and Triton-fallback halves of the plugin were
dropped: this targets gfx942 with the HIP kernels in
`csrc/libtorch_stable/quantization/gguf/`.
"""

from .config import GGUFConfig
from .fused_moe import GGUFMoEMethod, fused_moe_gguf
from .linear import GGUFLinearMethod, fused_mul_mat_gguf
from .params import (
    GGUFUninitializedParameter,
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    GGUFWeightParameter,
    GGUFWeightTypeParameter,
)
from .utils import (
    DEQUANT_TYPES,
    IMATRIX_QUANT_TYPES,
    KQUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    STANDARD_QUANT_TYPES,
    UNQUANTIZED_TYPES,
    is_layer_skipped_gguf,
)
from .vocal_embeds import GGUFEmbeddingMethod, apply_gguf_embedding

__all__ = [
    "DEQUANT_TYPES",
    "GGUFConfig",
    "GGUFEmbeddingMethod",
    "GGUFLinearMethod",
    "GGUFMoEMethod",
    "GGUFUninitializedParameter",
    "GGUFUninitializedWeightParameter",
    "GGUFUninitializedWeightTypeParameter",
    "GGUFWeightParameter",
    "GGUFWeightTypeParameter",
    "IMATRIX_QUANT_TYPES",
    "KQUANT_TYPES",
    "MMQ_QUANT_TYPES",
    "MMVQ_QUANT_TYPES",
    "STANDARD_QUANT_TYPES",
    "UNQUANTIZED_TYPES",
    "apply_gguf_embedding",
    "fused_moe_gguf",
    "fused_mul_mat_gguf",
    "is_layer_skipped_gguf",
]
