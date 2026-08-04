# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GGUF vector decode matmul must be right in every output element.

ROCm used to pre-fill this kernel's output with zeros, which cost about a fifth
of every bs=1 decode matmul. The fill is gone because the kernel was audited to
write its whole output, so that coverage is now load-bearing: an element the
kernel skips leaks whatever the allocator last left there.

The check is against a dequantize-then-matmul reference rather than a poisoned
buffer. A poison test that allocates a NaN tensor and frees it -- hoping the
caching allocator hands the same block back -- cannot fail reliably, because
torch may serve a different block or a fresh (zeroed) allocation. A reference
comparison catches an unwritten element as a wrong value regardless of what
happened to be in memory, and catches arithmetic errors too.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.gguf import ops
from vllm.model_executor.layers.quantization.gguf.utils import DEQUANT_TYPES

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="these are device kernels"
)

# (name, quant type id, bytes per block, elements per block)
QUANTS = [
    ("Q2_K", 10, 84, 256),
    ("Q3_K", 11, 110, 256),
    ("Q4_K", 12, 144, 256),
    ("Q5_K", 13, 176, 256),
    ("Q6_K", 14, 210, 256),
    ("Q8_0", 8, 34, 32),
    ("IQ2_XXS", 16, 66, 256),
]
SUPPORTED = {int(t) for t in DEQUANT_TYPES}


@pytest.mark.parametrize(("name", "qtype", "blk_bytes", "blk_elems"), QUANTS)
@pytest.mark.parametrize(
    ("cols", "rows"),
    # A row count that divides the tile, and one that does not.
    [(512, 3072), (256, 129)],
)
def test_matches_dequantized_reference(name, qtype, blk_bytes, blk_elems, cols, rows):
    if qtype not in SUPPORTED:
        pytest.skip(f"{name} has no dequant kernel to build a reference from")
    if cols % blk_elems:
        pytest.skip(f"{cols} is not a whole number of {name} blocks")

    generator = torch.Generator(device="cpu").manual_seed(0)
    weight = torch.randint(
        0,
        256,
        (rows, cols // blk_elems * blk_bytes),
        generator=generator,
        dtype=torch.uint8,
    ).cuda()
    x = torch.randn(1, cols, generator=generator, dtype=torch.bfloat16).cuda()

    got = ops.ggml_mul_mat_vec_a8(weight, x, qtype, rows)
    deq = ops.ggml_dequantize(weight, qtype, rows, cols, torch.bfloat16)
    want = (deq.to(torch.float32) @ x.to(torch.float32).T).T

    finite = torch.isfinite(want)
    assert finite.any(), "reference produced no finite values; pick a different seed"
    # The kernel quantizes the activation to q8_1, so it is deliberately lossy;
    # this tolerance catches an unwritten or grossly wrong element, not rounding.
    torch.testing.assert_close(
        got.to(torch.float32)[finite], want[finite], atol=2e-1, rtol=2e-1
    )
