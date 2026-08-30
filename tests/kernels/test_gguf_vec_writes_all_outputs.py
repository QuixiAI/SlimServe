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

# Byte offsets of the fp16 block scales. Arbitrary bytes are not valid test
# blocks: they frequently encode NaN/Inf scales, for which the direct dot and
# dequantized reference need not preserve the same finite subset.
SCALE_OFFSETS = {
    10: (80, 82),  # Q2_K: d, dmin
    11: (108,),  # Q3_K: d
    12: (0, 2),  # Q4_K: d, dmin
    13: (0, 2),  # Q5_K: d, dmin
    14: (208,),  # Q6_K: d
    8: (0,),  # Q8_0: d
    16: (0,),  # IQ2_XXS: d
}


def _make_finite_blocks(
    rows: int,
    cols: int,
    blk_bytes: int,
    blk_elems: int,
    qtype: int,
    generator: torch.Generator,
) -> torch.Tensor:
    weight = torch.randint(
        0,
        256,
        (rows, cols // blk_elems * blk_bytes),
        generator=generator,
        dtype=torch.uint8,
    )
    blocks = weight.view(rows, -1, blk_bytes)
    one = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
    for offset in SCALE_OFFSETS[qtype]:
        blocks[:, :, offset : offset + 2] = one
    return weight.cuda()


def _dequantize_q8_1(x: torch.Tensor) -> torch.Tensor:
    """Recover the activation actually consumed by the vector dot kernel."""
    blocks = ops.ggml_quantize_q8_1(x).view(torch.uint8).reshape(x.shape[0], -1, 36)
    blocks = blocks[:, : x.shape[1] // 32]
    scales = blocks[:, :, :2].contiguous().view(torch.float16).squeeze(-1).float()
    values = blocks[:, :, 4:].view(torch.int8).float()
    return (values * scales.unsqueeze(-1)).reshape_as(x).float()


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
    weight = _make_finite_blocks(rows, cols, blk_bytes, blk_elems, qtype, generator)
    x = torch.randn(1, cols, generator=generator, dtype=torch.bfloat16).cuda()

    got = ops.ggml_mul_mat_vec_a8(weight, x, qtype, rows)
    deq = ops.ggml_dequantize(weight, qtype, rows, cols, torch.float32)
    want = (deq @ _dequantize_q8_1(x).T).T

    assert torch.isfinite(want).all()
    # The direct kernel accumulates decoded values in a different order and
    # returns bf16. This tolerance covers that rounding while still rejecting
    # an unwritten or grossly wrong output element.
    torch.testing.assert_close(got.float(), want, atol=8, rtol=5e-2)


def test_iq2_xxs_moe_ep_matches_route_major_with_nonlocal_experts():
    qtype = 16
    if qtype not in SUPPORTED:
        pytest.skip("IQ2_XXS has no dequant kernel in this build")

    generator = torch.Generator(device="cpu").manual_seed(1)
    tokens, top_k, experts, rows, cols = 2, 4, 3, 129, 256
    weight = torch.randint(
        0,
        256,
        (experts, rows, 66),
        generator=generator,
        dtype=torch.uint8,
    ).cuda()
    x = torch.randn(tokens, cols, generator=generator, dtype=torch.bfloat16).cuda()
    topk_ids = torch.tensor(
        [[0, -1, 1, -1], [-1, 2, -1, 0]], dtype=torch.int32, device="cuda"
    )

    want = ops.ggml_moe_a8_vec(x, weight, topk_ids, top_k, qtype, rows, tokens)
    got = ops.ggml_moe_a8_vec(
        x,
        weight,
        topk_ids,
        top_k,
        qtype,
        rows,
        tokens,
        expert_parallel=True,
    )

    torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)
    assert torch.count_nonzero(got[topk_ids.flatten() < 0]) == 0
