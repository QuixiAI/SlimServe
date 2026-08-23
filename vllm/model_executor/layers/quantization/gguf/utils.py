# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from types import MappingProxyType

from gguf import GGMLQuantizationType as WeightType

from vllm.logger import init_logger

logger = init_logger(__name__)


def is_layer_skipped_gguf(
    prefix: str,
    unquantized_modules: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
):
    proj_name = prefix.split(".")[-1]
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = any(
                shard_prefix in module_name for module_name in unquantized_modules
            )

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_skipped = any(module_name in prefix for module_name in unquantized_modules)

    assert is_skipped is not None
    return is_skipped


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
# Q8_1 is deliberately absent: it is the activation format the q8 dot-product
# kernels quantize *into*, never a weight storage type, and no dequant, MMVQ or
# MMQ kernel has a case for it. Listing it here promised a path that would have
# dereferenced a null dequant function.
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# OCP MXFP4 (GGML_TYPE_MXFP4 = 39), the format DeepSeek-V4-Flash stores its
# routed experts in. Dequant, the q8_1 vector paths and the MMQ tile paths are
# all implemented.
MXFP4_QUANT_TYPES = {WeightType.MXFP4}

DEQUANT_TYPES = (
    STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES | MXFP4_QUANT_TYPES
)
MMVQ_QUANT_TYPES = (
    STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES | MXFP4_QUANT_TYPES
)
# IQ2_XXS is the only imatrix quant with a CUDA/HIP tile kernel. The rest of
# IMATRIX_QUANT_TYPES stays vector-only there, so it is listed on its own
# rather than folding the whole set in.
MMQ_IMATRIX_QUANT_TYPES = {WeightType.IQ2_XXS}
MMQ_QUANT_TYPES = (
    STANDARD_QUANT_TYPES | KQUANT_TYPES | MXFP4_QUANT_TYPES | MMQ_IMATRIX_QUANT_TYPES
)
# The Metal tile GEMM (qgemm.metal) decodes the full GGUF imatrix set
# (dequant.metal tile decoders + qgemm/qgemm_frag instantiations).
METAL_MMQ_QUANT_TYPES = (
    STANDARD_QUANT_TYPES
    | KQUANT_TYPES
    | MXFP4_QUANT_TYPES
    | {
        WeightType.IQ1_S,
        WeightType.IQ1_M,
        WeightType.IQ2_XXS,
        WeightType.IQ2_XS,
        WeightType.IQ2_S,
        WeightType.IQ3_XXS,
        WeightType.IQ3_S,
        WeightType.IQ4_XS,
        WeightType.IQ4_NL,
    }
)
