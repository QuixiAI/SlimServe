# SPDX-License-Identifier: Apache-2.0
"""Grouped compact pooled scoring (one pool read per request for the k+1
speculative verify rows) must match the per-row scorer."""

import pytest
import torch

from vllm.model_executor.layers.glm5_next_pool_cache import (
    cached_pool_logits,
    cached_pool_logits_grouped,
    rows_form_request_groups,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")


def _case(num_reqs, group, bs=64, num_blocks=96, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    rows = num_reqs * group
    cache = (torch.randn(num_blocks, bs, 64, device=dev) * 0.5).to(torch.bfloat16)
    pools_per_page = bs // 4
    stride = num_blocks // num_reqs
    bt = torch.zeros(num_reqs, stride, dtype=torch.int32, device=dev)
    for r in range(num_reqs):
        bt[r] = torch.randperm(num_blocks, device=dev)[:stride].to(torch.int32)
    # Each request sees a different number of tokens; rows within a request
    # advance by one token (speculative positions).
    base_vis = torch.randint(8, stride * pools_per_page * 4 - group - 1, (num_reqs,), device=dev)
    visible = (base_vis.repeat_interleave(group) + torch.arange(rows, device=dev) % group).to(torch.int32)
    row_req = (torch.arange(rows, device=dev) // group).to(torch.int32)
    q = (torch.randn(rows, 32, 128, device=dev) * 0.3).to(torch.bfloat16)
    weights = torch.rand(rows, 32, device=dev, dtype=torch.float32)
    max_pools = stride * pools_per_page
    return q, weights, cache, bt, row_req, visible, max_pools


@pytest.mark.parametrize("num_reqs,group", [(1, 2), (3, 4), (5, 4), (2, 3), (2, 5), (4, 8)])
def test_grouped_matches_per_row(num_reqs, group):
    q, w, cache, bt, row_req, visible, max_pools = _case(num_reqs, group)
    ref = torch.full((q.shape[0], max_pools), float("nan"), device="cuda", dtype=torch.float32)
    out = torch.full_like(ref, float("nan"))
    cached_pool_logits(q, w, cache, bt, row_req, visible, ref, 64)
    cached_pool_logits_grouped(q, w, cache, bt, row_req, visible, out, group, 64)
    # Only [0, count_r) is consumed downstream (top_k_per_row_prefill reads
    # n_pools = visible // 4 per row); neither kernel defines the cells past
    # its last tile. The grouped kernel is bit-exact on the defined range.
    for r in range(q.shape[0]):
        n = int(visible[r]) // 4
        a, b = ref[r, :n], out[r, :n]
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(b, a, rtol=0, atol=0)


def test_group_layout_check():
    rr = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device="cuda")
    assert rows_form_request_groups(rr, 3)
    assert not rows_form_request_groups(rr, 2)
    assert not rows_form_request_groups(rr, 4)
