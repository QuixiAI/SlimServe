# SPDX-License-Identifier: Apache-2.0
"""The pooled indexer's packed [wk | compress gate | weights_proj] projection
matches the three separate projections it replaces."""

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.glm5_next_indexer import (
    apply_packed_indexer_projection,
    packed_indexer_projection,
)

HIDDEN, HEAD_DIM, N_HEADS = 4096, 128, 32


class _Plain(nn.Module):
    def __init__(self, n: int, dtype=torch.bfloat16, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(n, HIDDEN, dtype=dtype, device=device) * 0.02
        )
        self.quant_method = None


def test_packing_order_and_refusals():
    wk, wp = _Plain(HEAD_DIM), _Plain(N_HEADS)
    gate = torch.randn(HEAD_DIM, HIDDEN)  # fp32 in the checkpoint
    packed = packed_indexer_projection(wk, gate, wp)
    assert packed is not None and packed.shape == (2 * HEAD_DIM + N_HEADS, HIDDEN)
    assert packed.dtype == torch.bfloat16 and packed.is_contiguous()
    assert torch.equal(packed[:HEAD_DIM], wk.weight.data)
    assert torch.equal(packed[HEAD_DIM : 2 * HEAD_DIM], gate.to(torch.bfloat16))
    assert torch.equal(packed[2 * HEAD_DIM :], wp.weight.data)
    assert packed_indexer_projection(_Plain(HEAD_DIM, torch.float16), gate, wp) is None
    assert packed_indexer_projection(wk, torch.randn(HEAD_DIM, 512), wp) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("tokens", [1, 4, 8, 9, 64])
def test_packed_projection_matches_the_separate_projections(tokens):
    torch.manual_seed(0)
    device = torch.device("cuda")
    wk, wp = _Plain(HEAD_DIM, device=device), _Plain(N_HEADS, device=device)
    gate = torch.randn(HEAD_DIM, HIDDEN, device=device)
    x = torch.randn(tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    packed = packed_indexer_projection(wk, gate, wp)
    k, g, w = apply_packed_indexer_projection(x, packed, HEAD_DIM, N_HEADS)
    ref_k = x.float() @ wk.weight.float().T
    # The gate is bf16 either way; compare against the exact product of
    # the bf16-rounded gate rows within a bf16 ulp.
    ref_g = x.float() @ gate.to(torch.bfloat16).float().T
    ref_w = x.float() @ wp.weight.float().T
    assert (
        k.dtype == torch.float32
        and g.dtype == torch.bfloat16
        and w.dtype == torch.float32
    )
    torch.testing.assert_close(k, ref_k, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(g.float(), ref_g, atol=2e-2, rtol=2**-7)
    torch.testing.assert_close(w, ref_w, atol=2e-2, rtol=2e-2)
