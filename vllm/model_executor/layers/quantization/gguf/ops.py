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
        raise NotImplementedError(
            "quixicore(metal): no grouped GEMV over GGUF-quantized experts; "
            "the MoE layer takes the per-expert loop instead."
        )
    _load_stable_libtorch()
    return torch.ops._C.ggml_moe_a8_vec(
        X, W, topk_ids, top_k, quant_type, row, tokens, expert_parallel
    )


def ggml_moe_get_block_size(quant_type: int) -> int:
    _load_stable_libtorch()
    return torch.ops._C.ggml_moe_get_block_size(quant_type)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._moe_C.moe_sum(input, output)
