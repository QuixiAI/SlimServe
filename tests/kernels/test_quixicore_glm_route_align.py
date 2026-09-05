# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused small-M routing (quixicore glm_route_align): sigmoid scores, bias-only
top-k, renormalize + scaling, Marlin block alignment in one launch, against
the router's grouped_topk and moe_align_block_size, for GLM-5.3-Flash's
routing (E=288, top-8, one expert group) at M = 1..16."""

import pytest
import torch
import triton

pytest.importorskip("vllm._quixicore_C")
from vllm.quixicore.ops import quixicore_ops as qc  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not qc.has_glm_route_align():
    pytest.skip("QuixiCore build without glm_route_align", allow_module_level=True)

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.router import glm_route_align  # noqa: E402
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (  # noqa: E402
    grouped_topk,
)

DEV = "cuda"
E, K, H = 288, 8, 4096
SCALE = 2.5


def _blocks(sorted_ids, expert_ids, post_pad, block_size, numel):
    n = int(post_pad.item())
    layout: dict[int, list[int]] = {}
    for b in range(n // block_size):
        e = int(expert_ids[b].item())
        toks = [
            int(t) for t in sorted_ids[b * block_size : (b + 1) * block_size].tolist()
        ]
        layout.setdefault(e, []).extend(t for t in toks if t < numel)
    return {e: sorted(v) for e, v in layout.items()}, n


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 13, 16])
def test_glm_route_align_matches_router_and_alignment(tokens):
    torch.manual_seed(0)
    logits = (torch.randn(tokens, E, device=DEV) * 2).float()
    bias = (torch.randn(E, device=DEV) * 0.5).float()
    hidden = torch.randn(tokens, H, device=DEV, dtype=torch.bfloat16)
    block_size = glm_route_align.marlin_block_size_m(tokens, K, E)
    max_padded, max_blocks = glm_route_align.alignment_geometry(
        tokens, K, E, block_size
    )

    w, ids, sorted_ids, expert_ids, post_pad = torch.ops.vllm.glm_route_align(
        logits,
        bias,
        K,
        glm_route_align.SCORING["sigmoid"],
        True,
        SCALE,
        block_size,
        max_padded,
        max_blocks,
    )
    ref_w, ref_ids = grouped_topk(
        hidden_states=hidden,
        gating_output=logits,
        topk=K,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=SCALE,
        e_score_correction_bias=bias,
    )
    assert torch.equal(
        ids.sort(dim=1).values, ref_ids.to(torch.int32).sort(dim=1).values
    )
    torch.testing.assert_close(
        w.sort(dim=1).values, ref_w.sort(dim=1).values, atol=1e-6, rtol=1e-6
    )

    ref_sorted, ref_experts, ref_pad = moe_align_block_size(
        ref_ids.to(torch.int32), block_size, E, None, ignore_invalid_experts=True
    )
    assert int(post_pad.item()) == int(ref_pad.item())
    assert _blocks(sorted_ids, expert_ids, post_pad, block_size, tokens * K) == _blocks(
        ref_sorted, ref_experts, ref_pad, block_size, tokens * K
    )
    assert max_blocks == triton.cdiv(max_padded, block_size)


def test_glm_route_align_rejects_unsupported_shapes():
    logits = torch.zeros(17, E, device=DEV)
    bias = torch.zeros(E, device=DEV)
    with pytest.raises(RuntimeError):
        torch.ops.vllm.glm_route_align(logits, bias, K, 0, True, SCALE, 8, 64, 8)
