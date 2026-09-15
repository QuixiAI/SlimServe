# SPDX-License-Identifier: Apache-2.0
"""Metal single-group MoE router (``moe_router_topk``) vs the torch
``grouped_topk`` chain: same expert sets, same weights."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)


def _qc():
    from vllm.quixicore.ops import quixicore_ops

    if not quixicore_ops.has("moe_router_topk"):
        pytest.skip("moe_router_topk not built")
    return quixicore_ops


@pytest.mark.parametrize("scoring", ["sigmoid", "softmax"])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("renorm", [True, False])
@pytest.mark.parametrize(
    "T,E,K", [(1, 288, 8), (7, 288, 8), (64, 256, 6), (3, 1000, 32)]
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_router_matches_grouped_topk(scoring, with_bias, renorm, T, E, K, dtype):
    # fp32 logits only (the serving case: GLM routes in fp32). The torch
    # chain on bf16 logits rounds every op to bf16 and picks different
    # experts at near-ties, so the Metal route is gated to fp32 and bf16 is
    # not compared here.
    tol = 2e-6
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk,
    )

    qc = _qc()
    torch.manual_seed(0)
    dev = "mps"
    logits = (torch.randn(T, E, device=dev) * 2).to(dtype)
    bias = (torch.randn(E, device=dev) * 0.1) if with_bias else None
    hidden = torch.zeros(T, 8, device=dev)
    ref_w, ref_ids = grouped_topk(
        hidden, logits, K, renorm, 1, 1, scoring, 2.5, bias
    )
    w, ids = qc.moe_router_topk(logits, bias, K, renorm, scoring == "softmax", 2.5)
    torch.mps.synchronize()
    assert ids.dtype == torch.int32 and w.dtype == torch.float32
    for t in range(T):
        rs = sorted(ref_ids[t].tolist())
        ks = sorted(ids[t].tolist())
        assert rs == ks, (t, rs, ks)
        rw = dict(zip(ref_ids[t].tolist(), ref_w[t].tolist()))
        kw = dict(zip(ids[t].tolist(), w[t].tolist()))
        for e in rs:
            assert abs(rw[e] - kw[e]) <= tol * max(1.0, abs(rw[e])), (
                t, e, rw[e], kw[e],
            )
