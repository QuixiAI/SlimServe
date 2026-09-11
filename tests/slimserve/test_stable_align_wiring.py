# SPDX-License-Identifier: Apache-2.0
"""Large alignment dispatch, fail-closed scope, and canonical provenance."""

from types import SimpleNamespace

import pytest
import torch

from slimserve import canonical_moe
from vllm.model_executor.layers.fused_moe.experts import marlin_moe
from vllm.model_executor.layers.fused_moe.router import glm_route_align as route
from vllm.model_executor.layers.fused_moe.router import glm_stable_align as stable
from vllm.quixicore import ops
from vllm.scalar_type import scalar_types


def test_stable_align_requires_canonical_diagnostic(monkeypatch):
    monkeypatch.delenv("SLIMSERVE_GLM53_STABLE_ALIGN", raising=False)
    assert not canonical_moe.stable_align_enabled()
    for value in ("yes", "true", "-1", "2"):
        monkeypatch.setenv("SLIMSERVE_GLM53_STABLE_ALIGN", value)
        with pytest.raises(ValueError, match="0 or 1"):
            canonical_moe.stable_align_enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_STABLE_ALIGN", "1")
    monkeypatch.delenv("SLIMSERVE_GLM53_CANONICAL_MOE", raising=False)
    with pytest.raises(ValueError, match="requires canonical MoE"):
        canonical_moe.stable_align_enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_MOE", "1")
    monkeypatch.delenv("SLIMSERVE_GLM53_MODEL_JOURNAL", raising=False)
    with pytest.raises(ValueError, match="bounded model journal"):
        canonical_moe.stable_align_enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    assert canonical_moe.stable_align_enabled()


def test_missing_native_fails_closed(monkeypatch):
    monkeypatch.setattr(ops, "_qc", lambda: SimpleNamespace())
    with pytest.raises(RuntimeError, match="requires rebuilt native"):
        ops.quixicore_ops.glm_stable_align(None, None, None, None, None, 8)


@pytest.mark.parametrize("tokens", [17, 32, 33, 640, 8192])
@pytest.mark.parametrize("block", [8, 16, 32, 48, 64])
def test_fake_geometry(tokens, block):
    ids = torch.empty((tokens, 8), device="meta", dtype=torch.int32)
    outputs = stable._glm_stable_align_fake(ids, block)
    capacity, blocks = route.alignment_geometry(tokens, 8, 288, block)
    assert [tuple(t.shape) for t in outputs] == [(capacity,), (blocks,), (1,)]
    assert all(t.device.type == "meta" and t.dtype == torch.int32 for t in outputs)


def invoke(monkeypatch, tokens, *, active=True, invalid=None, reused=False):
    monkeypatch.setattr(marlin_moe, "_STABLE_ALIGN_DIAGNOSTIC", active)
    monkeypatch.setattr(marlin_moe, "_CANONICAL_MOE_DIAGNOSTIC", True)
    monkeypatch.setattr(route, "_published", None)
    hidden = 2048 if invalid == "hidden" else 4096
    experts = 287 if invalid == "experts" else 288
    topk = 7 if invalid == "topk" else 8
    dtype = torch.float16 if invalid == "dtype" else torch.bfloat16
    quant = scalar_types.uint4b8 if invalid == "quant" else scalar_types.float4_e2m1f
    ids = torch.empty(tokens, topk, dtype=torch.int32, device="meta")
    result = [torch.empty(1, device="meta", dtype=torch.int32) for _ in range(3)]
    calls = []

    def align(*args, **kwargs):
        calls.append("atomic")
        assert args[0] is ids
        return result

    def canonical(ids_arg, block):
        calls.append("native")
        assert ids_arg is ids and block == route.marlin_block_size_m(tokens, 8, 288)
        return result

    def gemm(**kwargs):
        assert kwargs["sorted_token_ids"] is result[0]
        return torch.empty(tokens * topk, hidden, device="meta", dtype=dtype)

    monkeypatch.setattr(marlin_moe, "moe_align_block_size", align)
    monkeypatch.setattr(stable, "align", canonical)
    monkeypatch.setattr(
        canonical_moe, "canonicalize", lambda *a, **k: calls.append("sort")
    )
    monkeypatch.setattr(marlin_moe, "_fused_marlin_moe", gemm)
    if reused:
        route.publish(
            route.RoutingAlignment(ids, *result, 8, canonical_assignment_order=True)
        )
    args = (
        torch.empty(tokens, hidden, device="meta", dtype=dtype),
        torch.empty(experts, hidden // 16, 1, device="meta", dtype=torch.int32),
        torch.empty(experts, 1, hidden * 2, device="meta", dtype=torch.int32),
        None,
        None,
        torch.empty(0),
        torch.empty(0),
        torch.empty(tokens, topk),
        ids,
        quant.id,
    )
    kwargs = dict(
        expert_map=torch.arange(experts) if invalid == "ep" else None,
        global_num_experts=576 if invalid == "global" else -1,
    )
    if invalid:
        with pytest.raises(ValueError, match="requires GLM53 NVFP4 TP"):
            marlin_moe.fused_marlin_moe(*args, **kwargs)
        assert calls == []
    else:
        assert marlin_moe.fused_marlin_moe(*args, **kwargs).shape == (tokens, hidden)
    return calls


@pytest.mark.parametrize("tokens", [1, 16, 17, 32, 33, 640, 8192, 8193])
@pytest.mark.parametrize("active", [False, True])
def test_dispatch_only_skips_sort_for_actual_native_result(monkeypatch, tokens, active):
    expected = ["native"] if active and 17 <= tokens <= 8192 else ["atomic", "sort"]
    assert invoke(monkeypatch, tokens, active=active) == expected


def test_small_fused_alignment_still_reused(monkeypatch):
    assert invoke(monkeypatch, 16, reused=True) == []


@pytest.mark.parametrize(
    "invalid", ["hidden", "experts", "topk", "dtype", "quant", "ep", "global"]
)
def test_invalid_model_scope_fails_before_alignment(monkeypatch, invalid):
    invoke(monkeypatch, 640, invalid=invalid)
