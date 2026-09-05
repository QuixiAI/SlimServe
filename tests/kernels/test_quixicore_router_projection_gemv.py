# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router-gate use of the QuixiCore decode projection GEMV (dsv4_projection_gemv:
bf16 in, fp32 out, H=4096, M<=8) at GLM-5.3-Flash's expert count (E=288,
not the DSV4 kernel's E=256) against a float32 torch reference, and against
the bf16 F.linear + fp32 cast path it replaces in GateLinear tier 1b."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("vllm._quixicore_C")
from vllm.quixicore import quixicore_ops as qc  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = "cuda"
HIDDEN = 4096


@pytest.mark.parametrize("experts", [288, 256, 32])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8])
def test_router_projection_gemv_matches_fp32_reference(experts, tokens):
    torch.manual_seed(0)
    w = (torch.randn(experts, HIDDEN, device=DEV) * 0.02).to(torch.bfloat16)
    x = torch.randn(tokens, HIDDEN, device=DEV).to(torch.bfloat16)
    ref = x.float() @ w.float().t()
    out = qc.dsv4_projection_gemv(x, w, False)
    assert out.dtype == torch.float32 and out.shape == (tokens, experts)
    # fp32 accumulation of bf16 products: only summation order differs.
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
    # The path it replaces rounds the logits through bf16 first; the new
    # tier must be at least as close to the reference as that.
    old = F.linear(x, w).to(torch.float32)
    assert (out - ref).abs().max() <= (old - ref).abs().max() + 1e-6


def test_router_projection_gemv_rejects_wrong_shapes():
    w = torch.zeros(288, HIDDEN, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        qc.dsv4_projection_gemv(
            torch.zeros(9, HIDDEN, device=DEV, dtype=torch.bfloat16), w, False
        )
    with pytest.raises(RuntimeError):
        qc.dsv4_projection_gemv(
            torch.zeros(1, 2048, device=DEV, dtype=torch.bfloat16), w[:, :2048], False
        )
