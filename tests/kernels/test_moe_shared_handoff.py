# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two decode hand-offs between the MoE router, runner and Marlin experts:
the fused routing's alignment (router -> fused_marlin_moe) and the shared-expert
output (runner -> Marlin moe_sum). Both match on the identity of the batch's
topk_ids tensor and clear after one consumer, so a stale entry can never be
applied to another batch."""

import torch

from vllm.model_executor.layers.fused_moe import combine_shared
from vllm.model_executor.layers.fused_moe.router import glm_route_align


def test_alignment_is_consumed_once_and_only_by_its_batch():
    ids = torch.zeros(2, 8, dtype=torch.int32)
    other = torch.zeros(2, 8, dtype=torch.int32)
    alignment = glm_route_align.RoutingAlignment(
        ids, torch.empty(0), torch.empty(0), torch.empty(0), block_size=8
    )
    glm_route_align.publish(alignment)
    assert glm_route_align.consume(other) is None
    assert glm_route_align.consume(ids) is None  # cleared by the miss
    glm_route_align.publish(alignment)
    assert glm_route_align.consume(ids) is alignment
    assert glm_route_align.consume(ids) is None


def test_shared_output_entry_matches_its_batch_and_reports_the_fold():
    ids = torch.zeros(1, 8, dtype=torch.int32)
    entry = combine_shared.SharedOutput(ids, torch.zeros(1, 8), stream=None)
    combine_shared.publish(entry)
    assert combine_shared.consume(torch.zeros(1, 8, dtype=torch.int32)) is None
    taken = combine_shared.consume(ids)
    assert taken is entry and not taken.folded
    taken.folded = True
    assert entry.folded
    combine_shared.clear()
    assert combine_shared.consume(ids) is None


def test_marlin_block_size_and_geometry_mirror_the_reference():
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    for tokens in (1, 4, 16):
        block = glm_route_align.marlin_block_size_m(tokens, 8, 288)
        assert block == 8
        max_padded, max_blocks = glm_route_align.alignment_geometry(
            tokens, 8, 288, block
        )
        if torch.cuda.is_available():
            ids = torch.randint(0, 288, (tokens, 8), dtype=torch.int32, device="cuda")
            sorted_ids, expert_ids, _ = moe_align_block_size(ids, block, 288)
            assert sorted_ids.numel() == max_padded
            assert expert_ids.numel() == max_blocks
