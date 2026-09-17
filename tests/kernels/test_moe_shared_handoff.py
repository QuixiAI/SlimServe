# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two decode hand-offs between the MoE router, runner and Marlin experts:
the fused routing's alignment (router -> fused_marlin_moe) and the shared-expert
output (runner -> Marlin moe_sum). Both match on the identity of the batch's
topk_ids tensor and clear after one consumer, so a stale entry can never be
applied to another batch."""

from types import SimpleNamespace

import pytest
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


def test_block_size_and_geometry_match_moe_align_block_size():
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        marlin_moe_block_size_m,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
        moe_align_block_size_geometry,
    )

    for tokens in (1, 4, 16):
        block = marlin_moe_block_size_m(tokens, 8, 288, None)
        assert block == 8
        max_padded, max_blocks = moe_align_block_size_geometry(tokens * 8, 288, block)
        if torch.cuda.is_available():
            ids = torch.randint(0, 288, (tokens, 8), dtype=torch.int32, device="cuda")
            sorted_ids, expert_ids, _ = moe_align_block_size(ids, block, 288)
            assert sorted_ids.numel() == max_padded
            assert expert_ids.numel() == max_blocks


def test_shared_experts_discard_drops_an_unconsumed_output_and_tolerates_none():
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

    # Only the output slots are exercised; the constructor wants a live MoE
    # config, so build the module shell and set what discard() touches.
    se = SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(se)
    se.enable_dbo = False
    se._output = [torch.ones(1), None]
    se.discard()
    assert se._output == [None, None]
    se.discard()
    assert se._output == [None, None]


def _runner_shell():
    """A MoERunner with only the attributes the tests below touch; the
    constructor wants a live vLLM config and quantized experts."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    return runner


def test_set_moe_config_reselects_the_forward_entry():
    # Elastic EP reconfigures a live runner; the entry depends on the config
    # (use_ep and the padded hidden dim decide the fold), so it is re-chosen.
    runner = _runner_shell()
    seen: list[object] = []
    runner.routed_experts = SimpleNamespace(_set_moe_config=seen.append)
    runner._shared_experts = SimpleNamespace(_set_moe_config=seen.append)
    runner._forward_entry = "stale"
    runner._select_forward = lambda: ("fresh", runner.moe_config)
    cfg = object()
    runner._set_moe_config(cfg)
    assert runner.moe_config is cfg
    assert seen == [cfg, cfg]
    assert runner._forward_entry == ("fresh", cfg)


def test_a_router_failure_after_the_deferred_shared_run_discards_its_output():
    # forward_deferred_join fills the shared slot before expert selection; a
    # select_experts that raises must still drop it (and the published entry),
    # or the next batch asserts on the stale slot.
    runner = _runner_shell()
    calls: list[str] = []
    x = torch.zeros(2, 4)
    runner._shared_experts = SimpleNamespace(
        forward_deferred_join=lambda inp, pq: (torch.ones(2, 4), None),
        discard=lambda: calls.append("discard"),
    )
    runner._shared_fold_eligible = lambda *args: True
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_monolithic=False, topk_indices_dtype=None),
        forward_modular=lambda **kw: calls.append("forward_modular"),
    )

    def boom(**kwargs):
        raise RuntimeError("router failed")

    runner.router = SimpleNamespace(select_experts=boom)
    runner.__dict__["_expert_stats_obj"] = None  # what the _expert_stats property reads
    stale_ids = torch.zeros(2, 8, dtype=torch.int32)
    combine_shared.publish(combine_shared.SharedOutput(stale_ids, x, None))
    with pytest.raises(RuntimeError, match="router failed"):
        runner._apply_quant_method(x, torch.zeros(2, 8), x, fold_shared=True)
    assert calls == ["discard"]
    assert combine_shared.consume(stale_ids) is None


def test_a_monolithic_forward_that_raises_discards_the_shared_output():
    # Without a fold the NO_OVERLAP order fills the shared slot before the
    # routed forward; a monolithic kernel that raises must drop it too.
    runner = _runner_shell()
    calls: list[str] = []

    class Shared:
        def __call__(self, inp, order, pq):
            calls.append(f"shared:{order.name}")

        def discard(self):
            calls.append("discard")

    runner._shared_experts = Shared()

    def boom(**kwargs):
        raise RuntimeError("monolithic failed")

    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_monolithic=True), forward_monolithic=boom
    )
    x = torch.zeros(2, 4)
    with pytest.raises(RuntimeError, match="monolithic failed"):
        runner._apply_quant_method(x, torch.zeros(2, 8), x)
    assert calls == ["shared:NO_OVERLAP", "discard"]


def test_deferring_the_shared_expert_add_reselects_the_entry_and_excludes_the_fold():
    # DeepSeek V4 sets defer_shared_expert_add after construction to get
    # (shared_output, fused_output) back separately; the folded op returns
    # one tensor, so the flag must re-select the entry and veto the fold.
    from vllm.platforms import current_platform

    runner = _runner_shell()
    runner._shared_experts = SimpleNamespace(enable_dbo=False)
    runner.routed_scaling_factor = 1.0
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.moe_config = SimpleNamespace(
        is_sequence_parallel=False,
        moe_parallel_config=SimpleNamespace(use_ep=False),
        hidden_dim_unpadded=4096,
        hidden_dim=4096,
    )
    runner.defer_shared_expert_add = False  # before the entry exists: no reselect
    assert "_forward_entry" not in runner.__dict__
    if current_platform.is_cuda():
        assert runner._fold_shared_static()
    selections: list[str] = []
    runner._select_forward = lambda: selections.append("select") or "entry"
    runner._forward_entry = "stale"
    runner.defer_shared_expert_add = True
    assert not runner._fold_shared_static()
    assert selections == ["select"]
    assert runner._forward_entry == "entry"
