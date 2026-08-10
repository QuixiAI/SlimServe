# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrappers over the in-tree GGML kernels.

The plugin version of this file dispatched between a bundled ``_C_gguf``
extension and a ~6.6k-line Triton fallback. Both are gone: the kernels now live
in ``csrc/libtorch_stable/quantization/gguf/`` and register under the ``_C``
namespace. A missing op means the build is broken and should say so rather than
silently running something 20x slower.

Apple Metal is the exception. ``_C_stable_libtorch`` is a CUDA/HIP target and
is not built there, so ``torch.ops._C.ggml_*`` does not exist; the same ops
come from the QuixiCore Metal kernels through ``vllm._quixicore_C`` instead.

The branch is cached on first call rather than evaluated at import: touching
``current_platform`` during module import resolves the platform earlier than
vLLM intends and can pin the wrong one.
"""

from functools import cache

import torch


@cache
def _is_metal() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_metal()


@cache
def _load_stable_libtorch() -> None:
    """Importing the stable-ABI extension puts ggml_* into torch.ops._C."""
    import vllm._C_stable_libtorch  # noqa: F401


def ggml_dequantize(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError(
            "quixicore(metal): no generic GGUF dequant kernel yet. The Metal "
            "path reads quantized blocks directly through qgemv/qgemm, so the "
            "dequant-then-dense route is unavailable; the embedding and "
            "kv_b_proj callers still need one."
        )
    _load_stable_libtorch()
    return torch.ops._C.ggml_dequantize(W, quant_type, m, n, dtype)


def ggml_dequantize_into(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    out: torch.Tensor,
) -> None:
    if _is_metal():
        raise NotImplementedError(
            "quixicore(metal): no generic GGUF dequant kernel yet."
        )
    _load_stable_libtorch()
    torch.ops._C.ggml_dequantize_into(W, quant_type, m, n, out)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    if _is_metal():
        from vllm.quixicore import quixicore_ops

        return quixicore_ops.ggml_mul_mat_vec_a8(W, X, quant_type, row)
    _load_stable_libtorch()
    return torch.ops._C.ggml_mul_mat_vec_a8(W, X, quant_type, row)


def ggml_quantize_q8_1(X: torch.Tensor) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("GGUF Q8_1 activation packing is CUDA/HIP-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_quantize_q8_1(X)


def ggml_dsv4_rms_norm_q8_1(
    X: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_metal():
        raise NotImplementedError("DSV4 fused RMSNorm/Q8_1 is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_rms_norm_q8_1(X, weight, epsilon)


def ggml_mul_mat_vec_prequant_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("prequantized GGUF GEMV is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_mul_mat_vec_prequant_a8(
        W, X, quant_X, quant_type, row
    )


def ggml_dsv4_repack_q8_0_aligned(W: torch.Tensor) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 aligned Q8_0 is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_repack_q8_0_aligned(W)


def ggml_dsv4_mul_mat_vec_aligned_q8_0(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_input: torch.Tensor | None,
    row: int,
    rows_per_cta: int,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 aligned Q8_0 is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_mul_mat_vec_aligned_q8_0(
        W, X, quant_input, row, rows_per_cta
    )


def ggml_mul_mat_a8(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    if _is_metal():
        from vllm.quixicore import quixicore_ops

        return quixicore_ops.ggml_mul_mat_a8(W, X, quant_type, row)
    _load_stable_libtorch()
    return torch.ops._C.ggml_mul_mat_a8(W, X, quant_type, row)


def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
    mxfp4_repacked: bool = False,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError(
            "quixicore(metal): no grouped GEMM over GGUF-quantized experts; "
            "the MoE layer takes the per-expert loop instead."
        )
    _load_stable_libtorch()
    return torch.ops._C.ggml_moe_a8(
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
        mxfp4_repacked,
    )


def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    expert_parallel: bool = False,
) -> torch.Tensor:
    if _is_metal():
        if expert_parallel:
            raise NotImplementedError(
                "Metal exposes one device and does not support expert parallelism."
            )
        from vllm.quixicore import quixicore_ops

        return quixicore_ops.ggml_moe_a8_vec(
            X,
            W,
            topk_ids,
            top_k,
            quant_type,
            row,
            tokens,
        )
    _load_stable_libtorch()
    op = torch.ops._C.ggml_moe_a8_vec
    schema = str(op.default._schema)
    if "expert_parallel" in schema:
        return op(X, W, topk_ids, top_k, quant_type, row, tokens, expert_parallel)
    if expert_parallel:
        raise NotImplementedError(
            "loaded ggml_moe_a8_vec extension does not support expert_parallel"
        )
    return op(X, W, topk_ids, top_k, quant_type, row, tokens)


def ggml_dsv4_moe_a8(
    X: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    w1_expert_ids: torch.Tensor,
    w2_expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    intermediate: int,
    out_row: int,
    top_k: int,
    tokens: int,
    swiglu_limit: float,
    w1_repacked: bool = False,
    w2_repacked: bool = False,
    quant_input: torch.Tensor | None = None,
    defer_down: bool = False,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError(
            "quixicore(metal): DSV4 fused native MoE is implemented in the "
            "platform backend, not _C_stable_libtorch."
        )
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_moe_a8(
        X,
        W1,
        W2,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        w1_expert_ids,
        w2_expert_ids,
        num_tokens_post_padded,
        intermediate,
        out_row,
        top_k,
        tokens,
        swiglu_limit,
        w1_repacked,
        w2_repacked,
        quant_input,
        defer_down,
    )


def ggml_dsv4_moe_w1_a8(
    X: torch.Tensor,
    W1: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    w1_expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    intermediate: int,
    top_k: int,
    tokens: int,
    swiglu_limit: float,
    w1_repacked: bool = False,
    quant_input: torch.Tensor | None = None,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 output-owned Ampere W1 is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_moe_w1_a8(
        X,
        W1,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        w1_expert_ids,
        num_tokens_post_padded,
        intermediate,
        top_k,
        tokens,
        swiglu_limit,
        w1_repacked,
        quant_input,
    )


def ggml_dsv4_moe_down_output_owned(
    W2: torch.Tensor,
    quant_mid: torch.Tensor,
    topk_ids: torch.Tensor,
    tokens: int,
    top_k: int,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 output-owned Ampere W2 is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_moe_down_output_owned(
        W2, quant_mid, topk_ids, tokens, top_k
    )


def ggml_dsv4_repack_q2_k(
    W2: torch.Tensor, intermediate: int
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 Q2_K Ampere repack is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_repack_q2_k(W2, intermediate)


def ggml_dsv4_repack_iq2_xxs(W1: torch.Tensor, hidden: int) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 IQ2_XXS Ampere repack is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_repack_iq2_xxs(W1, hidden)


def ggml_dsv4_repack_mxfp4(W: torch.Tensor, values_per_row: int) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 MXFP4 Ampere repack is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_repack_mxfp4(W, values_per_row)


def ggml_dsv4_moe_a8_mxfp4(
    X: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    intermediate: int,
    out_row: int,
    top_k: int,
    tokens: int,
    swiglu_limit: float,
    w1_repacked: bool = False,
    w2_repacked: bool = False,
    quant_input: torch.Tensor | None = None,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 MXFP4 fused MoE is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_moe_a8_mxfp4(
        X,
        W1,
        W2,
        topk_weights,
        topk_ids,
        intermediate,
        out_row,
        top_k,
        tokens,
        swiglu_limit,
        w1_repacked,
        w2_repacked,
        quant_input,
    )


def ggml_dsv4_moe_a8_mxfp4_seg(
    X: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    intermediate: int,
    out_row: int,
    top_k: int,
    tokens: int,
    swiglu_limit: float,
    w1_repacked: bool = False,
    w2_repacked: bool = False,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 MXFP4 segmented MoE is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_moe_a8_mxfp4_seg(
        X,
        W1,
        W2,
        topk_weights,
        topk_ids,
        intermediate,
        out_row,
        top_k,
        tokens,
        swiglu_limit,
        w1_repacked,
        w2_repacked,
    )


def ggml_dsv4_moe_a8_iq2_seg(
    X: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    intermediate: int,
    out_row: int,
    top_k: int,
    tokens: int,
    swiglu_limit: float,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 hybrid segmented MoE is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_moe_a8_iq2_seg(
        X,
        W1,
        W2,
        topk_weights,
        topk_ids,
        intermediate,
        out_row,
        top_k,
        tokens,
        swiglu_limit,
    )


def ggml_dsv4_shared_gate_up_swiglu(
    W: torch.Tensor,
    X: torch.Tensor,
    swiglu_limit: float,
    quant_input: torch.Tensor | None = None,
) -> torch.Tensor:
    if _is_metal():
        raise NotImplementedError("DSV4 shared-expert Ampere fusion is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_shared_gate_up_swiglu(
        W, X, swiglu_limit, quant_input
    )


def ggml_dsv4_shared_gate_up_swiglu_q8_1(
    W: torch.Tensor,
    X: torch.Tensor,
    swiglu_limit: float,
    quant_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_metal():
        raise NotImplementedError("DSV4 shared-expert Ampere fusion is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_shared_gate_up_swiglu_q8_1(
        W, X, swiglu_limit, quant_input
    )


def ggml_dsv4_o_proj_q8_0(
    W: torch.Tensor,
    O: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    local_groups: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_metal():
        raise NotImplementedError("DSV4 native Q8 o_proj is CUDA-only")
    _load_stable_libtorch()
    return torch.ops._C.ggml_dsv4_o_proj_q8_0(
        W, O, positions, cos_sin_cache, local_groups, rope_dim
    )


def ggml_moe_get_block_size(quant_type: int) -> int:
    _load_stable_libtorch()
    return torch.ops._C.ggml_moe_get_block_size(quant_type)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._moe_C.moe_sum(input, output)
