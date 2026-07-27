# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrappers over the in-tree GGML kernels.

The plugin version of this file dispatched between a bundled ``_C_gguf``
extension and a ~6.6k-line Triton fallback. Both are gone: the kernels now live
in ``csrc/libtorch_stable/quantization/gguf/`` and register under the ``_C``
namespace, and this repo targets exactly one GPU (gfx942), so there is nothing
to fall back to. A missing op means the build is broken and should say so
rather than silently running something 20x slower.
"""

import torch

# Importing the stable-ABI extension is what puts ggml_* into torch.ops._C.
import vllm._C_stable_libtorch  # noqa: F401


def ggml_dequantize(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.ops._C.ggml_dequantize(W, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_vec_a8(W, X, quant_type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
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
) -> torch.Tensor:
    return torch.ops._C.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return torch.ops._C.ggml_moe_get_block_size(quant_type)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._moe_C.moe_sum(input, output)
