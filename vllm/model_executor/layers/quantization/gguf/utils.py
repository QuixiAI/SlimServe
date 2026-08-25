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
# Tile-kernel coverage differs between the DENSE and MoE entry points, so the
# two get separate lists. Conflating them is what broke Qwen3.8 GGUF prefill:
# IQ2_XXS has a tile kernel only in moe.cuh, but it was listed in the single
# shared MMQ set, so dense layers were routed to `ggml_mul_mat_a8`, whose
# switch has no case for it. With no `default:` there, the call returned its
# output buffer untouched -- zeros on ROCm, uninitialised on CUDA -- silently
# corrupting every prefill through an IQ2_XXS layer.
#
# Dense (mmq.cuh): standard quants, k-quants and MXFP4 only. Imatrix quants
# stay vector-only; past the mmvq batch limit they fall through to
# DEQUANT_TYPES, which is exact.
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | MXFP4_QUANT_TYPES
# MoE (moe.cuh): the same set plus IQ2_XXS, which is the DSV4 gate/up format
# and does have a real MoE tile kernel (`ggml_moe_iq2_xxs_q8_1_cuda`).
MOE_MMQ_IMATRIX_QUANT_TYPES = {WeightType.IQ2_XXS}
MOE_MMQ_QUANT_TYPES = MMQ_QUANT_TYPES | MOE_MMQ_IMATRIX_QUANT_TYPES
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
