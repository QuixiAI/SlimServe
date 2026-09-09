# SPDX-License-Identifier: Apache-2.0
"""Real import-time native policy with no diagnostic flags or mock kernels."""

import os
from types import SimpleNamespace

import pytest
import torch

from slimserve import canonical_indexer, glm53_ordering
from tests.kernels.test_glm53_indexer_ties import matrix, oracle
from tests.kernels.test_glm53_stable_route import check_result
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts import marlin_moe
from vllm.model_executor.layers.fused_moe.router import glm_route_align as route
from vllm.quixicore.ops import quixicore_ops as qc

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or os.getenv(glm53_ordering.FLAG) != "1",
    reason="requires CUDA and explicit native-order policy at process start",
)


def test_real_import_policy_without_observers():
    assert glm53_ordering.enabled()
    assert all(os.getenv(k, "0") == "0" for k in glm53_ordering.LEGACY_FLAGS)
    assert os.getenv("CUDA_LAUNCH_BLOCKING", "0") == "0"
    for suffix in ("MODEL_JOURNAL", "MOE_JOURNAL", "SCORE_JOURNAL", "INDEX_JOURNAL"):
        assert not os.getenv("SLIMSERVE_GLM53_" + suffix)
    assert route._STABLE_ALIGNMENT_DIAGNOSTIC
    assert marlin_moe._NATIVE_ORDER
    assert marlin_moe._CANONICAL_MOE_DIAGNOSTIC
    assert marlin_moe._STABLE_ALIGN_DIAGNOSTIC
    assert "glm53_native_order_v1=1" in qc.graph_factors()


@pytest.mark.parametrize("tokens", [1, 16])
def test_real_router_publishes_stable_alignment(tokens):
    logits = torch.zeros(tokens, 288, device="cuda")
    bias = torch.zeros(288, device="cuda")
    router = SimpleNamespace(
        e_score_correction_bias=bias,
        top_k=8,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    assert route.eligible(router, logits, torch.int32)
    capacity, blocks = route.alignment_geometry(tokens, 8, 288, 8)
    for phase in range(3):
        generator = torch.Generator().manual_seed(53900 + phase)
        if phase:
            logits.copy_(torch.randn(tokens, 288, generator=generator))
        else:
            logits.zero_()
        weights, ids = route.route(router, logits)
        alignment = route.consume(ids)
        assert alignment is not None and alignment.canonical_assignment_order
        assert route.consume(ids) is None
        control = qc.glm_route_align(
            logits, bias, 8, 0, True, 2.5, 8, capacity, blocks
        )
        check_result(
            [weights, ids, alignment.sorted_token_ids, alignment.expert_ids,
             alignment.num_tokens_post_padded],
            control, tokens, 8,
        )


@pytest.mark.parametrize("columns", [513, 1904, 8193, 262144])
@pytest.mark.parametrize("family", ["random", "adjacent", "cutoff"])
def test_real_selector_wrapper_eager_and_changed_input_graph(columns, family):
    # Pass the ordinary selector into the same factory used by the model.
    function = canonical_indexer.maybe_ordered_topk(ops.top_k_per_row_prefill)
    logits = torch.zeros(8, columns, device="cuda")
    starts = torch.zeros(8, dtype=torch.int32, device="cuda")
    ends = torch.zeros_like(starts)
    ids = torch.empty(8, 512, dtype=torch.int32, device="cuda")

    def call():
        function(logits, starts, ends, ids, 8, columns, 1, 512)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for phase in range(3):
        values, visible = matrix(8, columns, family, phase)
        expected = oracle(values, visible)
        values[torch.arange(columns)[None, :] >= visible[:, None]] = float("nan")
        logits.copy_(values)
        ends.copy_(visible)
        for invoke in (call, graph.replay):
            ids.fill_(-777)
            invoke()
            assert torch.equal(ids.cpu(), expected)
    torch.cuda.synchronize()
