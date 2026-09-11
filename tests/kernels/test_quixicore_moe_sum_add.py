# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE combine (quixicore moe_sum_add): out = shared + sum_k x[:, k] with
fp32 accumulation and one bf16 rounding, against the same fold in torch."""

import pytest
import torch

pytest.importorskip("vllm._quixicore_C")
from vllm.quixicore.ops import quixicore_ops as qc  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not qc.has_moe_sum_add():
    pytest.skip("QuixiCore build without moe_sum_add", allow_module_level=True)

DEV = "cuda"


def _reference(x: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    # Same association as the kernel: shared first, then k in order, fp32.
    acc = shared.float()
    for k in range(x.shape[1]):
        acc = acc + x[:, k].float()
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 8, 16, 37, 300])
@pytest.mark.parametrize("topk", [8, 6])
@pytest.mark.parametrize("hidden", [4096, 2048, 8])
def test_moe_sum_add_matches_fp32_fold(tokens, topk, hidden):
    torch.manual_seed(tokens * 31 + topk * 7 + hidden)
    x = torch.randn(tokens, topk, hidden, device=DEV, dtype=torch.bfloat16) * 3
    shared = torch.randn(tokens, hidden, device=DEV, dtype=torch.bfloat16) * 5
    out = torch.empty(tokens, hidden, device=DEV, dtype=torch.bfloat16)
    qc.moe_sum_add(x, shared, out)
    torch.testing.assert_close(out, _reference(x, shared), rtol=0, atol=0)
    # And within two bf16 ulps (of the largest magnitude) of the unfused
    # two-step path, which rounds twice.
    two_step = (shared + x.sum(dim=1)).float()
    ulp = 2.0 ** (torch.floor(torch.log2(two_step.abs().max())) - 7)
    torch.testing.assert_close(out.float(), two_step, rtol=0, atol=2 * ulp.item())


def test_moe_sum_add_is_total_on_non_finite_inputs():
    x = torch.full((4, 8, 4096), float("nan"), device=DEV, dtype=torch.bfloat16)
    shared = torch.randn(4, 4096, device=DEV, dtype=torch.bfloat16)
    out = torch.zeros(4, 4096, device=DEV, dtype=torch.bfloat16)
    qc.moe_sum_add(x, shared, out)
    torch.cuda.synchronize()
    assert out.isnan().all()


def test_moe_sum_add_rejects_bad_shapes():
    x = torch.randn(4, 8, 4096, device=DEV, dtype=torch.bfloat16)
    shared = torch.randn(4, 4096, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        qc.moe_sum_add(x, shared[:2], torch.empty_like(shared))
    with pytest.raises(RuntimeError):
        qc.moe_sum_add(
            x[:, :, :4092], shared[:, :4092], torch.empty_like(shared[:, :4092])
        )
    with pytest.raises(RuntimeError):
        qc.moe_sum_add(x.float(), shared, torch.empty_like(shared))
