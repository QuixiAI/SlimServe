# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GGUF vector kernel must write every output element.

ROCm used to pre-fill the output of `ggml_mul_mat_vec_a8` with zeros, which cost
about a fifth of every bs=1 decode matmul -- there are hundreds per step. The
fill is gone because the kernels were audited to cover their whole output, so
that coverage is now load-bearing: a quant type that skips elements would leak
whatever the caching allocator last left in the buffer, which is far worse than
slow. Zeroed weights make every correct result exactly 0.0, so any survivor of a
poisoned buffer is an unwritten element.

The tile kernel is deliberately not asserted here: the same audit found IQ2_XXS
leaving gaps there, which is why that path keeps its fill.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.gguf.ops import (
    ggml_moe_a8_vec,
    ggml_mul_mat_vec_a8,
)
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the pre-fill this guards was ROCm-only", allow_module_level=True)

# (quant type id, bytes per block, elements per block)
QUANTS = [
    pytest.param(10, 84, 256, id="Q2_K"),
    pytest.param(11, 110, 256, id="Q3_K"),
    pytest.param(12, 144, 256, id="Q4_K"),
    pytest.param(13, 176, 256, id="Q5_K"),
    pytest.param(14, 210, 256, id="Q6_K"),
    pytest.param(8, 34, 32, id="Q8_0"),
    pytest.param(16, 66, 256, id="IQ2_XXS"),
]


@pytest.mark.parametrize(("qtype", "block_bytes", "block_elems"), QUANTS)
@pytest.mark.parametrize(
    ("cols", "rows"),
    # A row count that divides the tile, and one that does not.
    [(512, 3072), (256, 129)],
)
def test_every_output_element_is_written(qtype, block_bytes, block_elems, cols, rows):
    if cols % block_elems:
        pytest.skip(f"{cols} is not a whole number of blocks")
    weight = torch.zeros(
        (rows, cols // block_elems * block_bytes), dtype=torch.uint8, device="cuda"
    )
    x = torch.randn(1, cols, device="cuda", dtype=torch.bfloat16)

    # Poison the allocator's free list so an unwritten element reads back NaN
    # rather than a plausible zero.
    poison = torch.full((4, rows), float("nan"), device="cuda", dtype=torch.bfloat16)
    del poison

    out = ggml_mul_mat_vec_a8(weight, x, qtype, rows)
    assert not torch.isnan(out).any(), (
        f"{int(torch.isnan(out).sum())} of {out.numel()} outputs left unwritten"
    )


@pytest.mark.parametrize(("qtype", "block_bytes", "block_elems"), QUANTS)
@pytest.mark.parametrize("tokens", [1, 4])
def test_moe_vector_kernel_writes_every_row(qtype, block_bytes, block_elems, tokens):
    """Same contract on the expert path, which lost its pre-fill for the same reason.

    Every (token, top_k) output row must be written, including when the router
    sends several tokens to the same expert and when it sends none to some.
    """
    experts, top_k, cols, rows = 32, 8, 512, 3072
    weight = torch.zeros(
        (experts, rows, cols // block_elems * block_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    x = torch.randn(tokens, cols, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(
        0, experts, (tokens, top_k), device="cuda", dtype=torch.int32
    )

    poison = torch.full(
        (8, tokens * top_k, rows), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    del poison

    out = ggml_moe_a8_vec(x, weight, topk_ids, top_k, qtype, rows, tokens)
    assert not torch.isnan(out).any(), (
        f"{int(torch.isnan(out).sum())} of {out.numel()} expert outputs unwritten"
    )
