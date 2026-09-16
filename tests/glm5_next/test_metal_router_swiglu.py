# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash MoE routing and SwiGLU clamp on the Metal path.

Routing: ``DeepseekV2MoE`` builds ``FusedMoE(use_grouped_topk=True)`` with
GLM's config (sigmoid scoring, ``noaux_tc`` => e_score_correction_bias,
top-8 of 288, n_group 1 / topk_group 1, norm_topk_prob, routed_scaling
2.5). Off CUDA the ``GroupedTopk`` CustomOp dispatches ``forward_mps`` ->
``forward_native`` -> the torch ``grouped_topk`` in
``fused_moe/router/grouped_topk_router.py`` (the ``is_cuda`` gate at :93
keeps the fused CUDA kernel off). This pins that torch path against an
independent float64 reference and checks it is total over NaN logits.

SwiGLU: ``swiglu_limit`` 10 reaches ``SiluAndMulWithClamp`` (dense and
shared experts) and ``apply_moe_activation(clamp_limit=10)`` (routed
experts through the GGUF MoE path), which on Metal hands the clamp to
``qc_swiglu`` (fused_moe/activation.py) when the shapes qualify.
"""

from __future__ import annotations

import pytest
import torch

DEVICES = ["cpu"]
MPS_ONLY: list[str] = []
if torch.backends.mps.is_available():
    DEVICES.append("mps")
    MPS_ONLY.append("mps")

E, TOPK, SCALE_F = 288, 8, 2.5


def _ref_route(logits: torch.Tensor, bias: torch.Tensor):
    """Independent float64 GLM router: sigmoid scores, +bias only for the
    choice, weights from the unbiased scores, renormalize, x2.5."""
    s = torch.sigmoid(logits.double())
    ids = torch.topk(s + bias.double()[None, :], TOPK, dim=-1).indices
    w = s.gather(1, ids)
    w = w / w.sum(dim=-1, keepdim=True) * SCALE_F
    return w, ids


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_grouped_topk_torch_matches_reference(device):
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk,
    )

    g = torch.Generator().manual_seed(0)
    T = 37
    # fp32 router logits, as GateLinear emits for the DeepSeek/GLM router
    # (the torch path sigmoids in the logits dtype).
    logits = torch.randn(T, E, generator=g) * 2
    bias = torch.randn(E, generator=g) * 0.5
    hidden = torch.randn(T, 16, generator=g)
    w, ids = grouped_topk(
        hidden.to(device), logits.to(device), TOPK, True,
        num_expert_group=1, topk_group=1, scoring_func="sigmoid",
        routed_scaling_factor=SCALE_F, e_score_correction_bias=bias.to(device),
    )
    w_r, ids_r = _ref_route(logits, bias)
    biased = torch.sigmoid(logits.double()) + bias.double()[None, :]
    assert w.dtype == torch.float32 and ids.dtype == torch.int32
    for t in range(T):
        got = set(ids[t].tolist())
        assert len(got) == TOPK
        if got != set(ids_r[t].tolist()):
            # fp32-vs-fp64 boundary: still a valid top-8 of the biased scores
            chosen_min = min(biased[t, e].item() for e in got)
            other_max = max(
                biased[t, e].item() for e in range(E) if e not in got
            )
            assert chosen_min >= other_max - 1e-5, t
            continue
        order = {e: i for i, e in enumerate(ids[t].tolist())}
        w_sorted = torch.tensor([w[t, order[e]].item() for e in ids_r[t].tolist()])
        torch.testing.assert_close(w_sorted.double(), w_r[t], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(w.sum(dim=-1).cpu(), torch.full((T,), SCALE_F),
                               atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_grouped_topk_total_over_nan(device):
    """A NaN logit row must not raise and must not disturb other rows."""
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk,
    )

    g = torch.Generator().manual_seed(1)
    logits = (torch.randn(4, E, generator=g) * 2).float()
    bias = torch.randn(E, generator=g) * 0.5
    hidden = torch.randn(4, 16, generator=g)
    clean_w, clean_ids = grouped_topk(
        hidden.to(device), logits.to(device), TOPK, True, 1, 1, "sigmoid",
        SCALE_F, bias.to(device),
    )
    logits[2, :] = float("nan")
    w, ids = grouped_topk(
        hidden.to(device), logits.to(device), TOPK, True, 1, 1, "sigmoid",
        SCALE_F, bias.to(device),
    )
    assert w.shape == (4, TOPK) and ids.shape == (4, TOPK)
    for t in (0, 1, 3):
        assert torch.equal(ids[t].cpu(), clean_ids[t].cpu())
        torch.testing.assert_close(w[t].cpu(), clean_w[t].cpu())
    assert (ids[2] >= 0).all() and (ids[2] < E).all()


def test_grouped_topk_custom_op_dispatch_off_cuda():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        GroupedTopk,
    )
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        pytest.skip("CUDA dispatch differs")
    with set_current_vllm_config(VllmConfig()):
        op = GroupedTopk(TOPK, True, 1, 1, "sigmoid", SCALE_F)
    fwd = op._forward_method
    name = getattr(fwd, "__name__", "")
    assert name in ("forward_mps", "forward_native", "forward_cpu"), name


def _ref_swiglu(x: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = x.double().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


# NOTE: on the Metal platform SiluAndMulWithClamp dispatches forward_mps
# (qc_swiglu) for every tensor, so CPU tensors raise there; the serving
# path only ever hands it MPS tensors.
@pytest.mark.parametrize("device", MPS_ONLY)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.no_grad()
def test_silu_and_mul_with_clamp(device, dtype):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    with set_current_vllm_config(VllmConfig()):
        act = SiluAndMulWithClamp(10.0)
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(65, 2 * 256, generator=g) * 8).to(dtype)  # exercises the clamp
    out = act(x.to(device))
    ref = _ref_swiglu(x, 10.0)
    assert (x.double().abs() > 10).any()
    torch.testing.assert_close(out.cpu().double(), ref, atol=0.25, rtol=2e-2)


@pytest.mark.parametrize("device", MPS_ONLY)
@torch.no_grad()
def test_apply_moe_activation_clamp(device):
    """The routed-expert activation with clamp_limit=10 (GGUF MoE path):
    on Metal this is qc_swiglu with the clamp, else the torch fallback."""
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        apply_moe_activation,
    )

    g = torch.Generator().manual_seed(4)
    x = (torch.randn(48, 2 * 512, generator=g) * 8).to(torch.float16).to(device)
    out = torch.empty((48, 512), dtype=torch.float16, device=device)
    apply_moe_activation(MoEActivation.SILU, out, x, clamp_limit=10.0)
    torch.testing.assert_close(out.cpu().double(), _ref_swiglu(x.cpu(), 10.0),
                               atol=0.25, rtol=2e-2)
    # without a limit the values beyond +-10 must NOT be clamped
    out2 = torch.empty_like(out)
    apply_moe_activation(MoEActivation.SILU, out2, x)
    assert not torch.allclose(out.float(), out2.float())


def test_glm_config_reaches_clamped_activation():
    """DeepseekV2MLP / DeepseekV2MoE (as built by glm5_next.py) construct the
    clamped activation from config.swiglu_limit."""
    import inspect

    from vllm.model_executor.models import deepseek_v2, glm5_next

    src = inspect.getsource(deepseek_v2.DeepseekV2MLP.__init__)
    assert "SiluAndMulWithClamp" in src and "swiglu_limit" in src
    moe_src = inspect.getsource(deepseek_v2.DeepseekV2MoE.__init__)
    assert moe_src.count("swiglu_limit") >= 2  # shared experts + routed
    layer_src = inspect.getsource(glm5_next.Glm5NextDecoderLayer.__init__)
    assert "swiglu_limit" in layer_src


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
