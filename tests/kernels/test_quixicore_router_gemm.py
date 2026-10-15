# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QuixiCore bf16 -> fp32 router GEMV against the fp32 reference, at
DSV4's 256 experts and GLM-5.3-Flash's 288, for every token count it serves."""

import pytest
import torch

from vllm.quixicore import quixicore_ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not quixicore_ops.is_available(),
    reason="needs a CUDA device with the QuixiCore extension",
)


@pytest.mark.parametrize("experts", [256, 288])
@pytest.mark.parametrize("tokens", [1, 2, 4, 7, 8])
def test_router_gemv_matches_fp32_reference(experts: int, tokens: int):
    g = torch.Generator(device="cuda").manual_seed(experts * 16 + tokens)
    x = torch.randn(tokens, 4096, dtype=torch.bfloat16, device="cuda", generator=g)
    w = torch.randn(experts, 4096, dtype=torch.bfloat16, device="cuda", generator=g)
    w *= 0.02
    out = quixicore_ops.dsv4_router_gemm(x, w)
    ref = x.float() @ w.float().T
    assert out.shape == (tokens, experts) and out.dtype == torch.float32
    # fp32 accumulation of exact bf16 products in another order.
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (
        f"max |diff| {(out - ref).abs().max().item():.3g}"
    )


def test_router_gemv_rejects_more_than_eight_tokens():
    x = torch.randn(9, 4096, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(288, 4096, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError):
        quixicore_ops.dsv4_router_gemm(x, w)
