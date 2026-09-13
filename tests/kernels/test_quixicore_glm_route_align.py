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

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (  # noqa: E402
    marlin_moe_block_size_m,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
    moe_align_block_size_geometry,
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
    block_size = marlin_moe_block_size_m(tokens, K, E, None)
    max_padded, max_blocks = moe_align_block_size_geometry(tokens * K, E, block_size)

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
    ordered, permutation = ids.sort(dim=1)
    ref_ordered, ref_permutation = ref_ids.to(torch.int32).sort(dim=1)
    assert torch.equal(ordered, ref_ordered)
    torch.testing.assert_close(
        w.gather(1, permutation), ref_w.gather(1, ref_permutation), atol=1e-6, rtol=1e-6
    )

    ref_sorted, ref_experts, ref_pad = moe_align_block_size(
        ref_ids.to(torch.int32), block_size, E, None, ignore_invalid_experts=True
    )
    assert int(post_pad.item()) == int(ref_pad.item())
    assert _blocks(sorted_ids, expert_ids, post_pad, block_size, tokens * K) == _blocks(
        ref_sorted, ref_experts, ref_pad, block_size, tokens * K
    )
    assert max_blocks == triton.cdiv(max_padded, block_size)


def _non_finite_cases(tokens):
    torch.manual_seed(1)
    base = (torch.randn(tokens, E, device=DEV) * 2).float()
    cases = {"all_nan": torch.full_like(base, float("nan"))}
    half = base.clone()
    half[tokens // 2 :] = float("nan")
    cases["half_nan"] = half
    row = base.clone()
    row[0] = float("inf")
    cases["inf_row"] = row
    row = base.clone()
    row[-1] = float("-inf")
    cases["neg_inf_row"] = row
    mixed = base.clone()
    mixed[:, ::3] = float("nan")
    cases["nan_columns"] = mixed
    return cases


@pytest.mark.parametrize("tokens", [1, 8, 16])
def test_glm_route_align_is_total_on_non_finite_logits(tokens):
    """vLLM's dummy runs (graph-capture warmups, the sampler warmup) carry NaN
    activations. Whatever the logits, every expert id must stay in [0, E) and
    the alignment must be one Marlin can consume: post_pad a multiple of the
    block size within the buffer, every block its valid entries first and then
    padding, the tails filled, every assignment placed exactly once in a block
    of its own expert. The ids of a NaN token are meaningless but bounded."""
    bias = (torch.randn(E, device=DEV) * 0.5).float()
    block_size = marlin_moe_block_size_m(tokens, K, E, None)
    max_padded, max_blocks = moe_align_block_size_geometry(tokens * K, E, block_size)
    numel = tokens * K
    for name, logits in _non_finite_cases(tokens).items():
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
        n = int(post_pad.item())
        assert 0 < n <= max_padded and n % block_size == 0, name
        assert int(ids.min()) >= 0 and int(ids.max()) < E, name
        assert ids.shape == (tokens, K) and w.shape == (tokens, K), name
        blocks = sorted_ids[:n].view(-1, block_size)
        valid = blocks < numel
        padded_before = (~valid).int().cumsum(dim=1) > 0
        assert not bool((padded_before & valid).any()), f"{name}: pad before valid"
        assert bool((sorted_ids[n:] == numel).all()), f"{name}: sorted tail"
        used = expert_ids[: n // block_size]
        assert int(used.min()) >= 0 and int(used.max()) < E, name
        assert bool((expert_ids[n // block_size :] == -1).all()), f"{name}: expert tail"
        assignments = blocks[valid]
        assert torch.equal(
            assignments.sort().values,
            torch.arange(numel, device=DEV, dtype=assignments.dtype),
        ), f"{name}: every assignment once"
        owner = ids.view(-1)[blocks.clamp(max=numel - 1)]
        assert not bool(((owner != used[:, None]) & valid).any()), (
            f"{name}: assignment outside its expert's block"
        )


def test_glm_route_align_rejects_unsupported_shapes():
    logits = torch.zeros(17, E, device=DEV)
    bias = torch.zeros(E, device=DEV)
    max_padded, max_blocks = moe_align_block_size_geometry(17 * K, E, 8)
    with pytest.raises(RuntimeError, match=r"handles 1\.\.16 tokens"):
        torch.ops.vllm.glm_route_align(
            logits, bias, K, 0, True, SCALE, 8, max_padded, max_blocks
        )
