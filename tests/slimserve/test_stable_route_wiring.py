# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch/provenance checks; no real GEMM or GPU allocation."""

from types import SimpleNamespace

import pytest
import torch

from slimserve import canonical_moe
from vllm.model_executor.layers.fused_moe.experts import marlin_moe
from vllm.model_executor.layers.fused_moe.router import glm_route_align as route
from vllm.scalar_type import scalar_types


def test_benchmark_source_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    from benchmarks.kernels.benchmark_glm53_stable_route import ROOT, source_paths

    native = SimpleNamespace(__file__=str(tmp_path / "native.so"))
    probe = SimpleNamespace(__file__=str(tmp_path / "probe.so"))
    before = source_paths(native, probe)
    monkeypatch.chdir(tmp_path)
    assert source_paths(native, probe) == before
    assert all(path.is_absolute() for path in before)
    assert ROOT / "csrc/quixicore/serving/glm_moe_routing.cuh" in before
    assert before[-1] == tmp_path / "probe.so"


def test_stale_native_fails_only_when_stable_requested(monkeypatch):
    from vllm.quixicore import ops

    sentinel = object()
    monkeypatch.setattr(
        ops, "_qc", lambda: SimpleNamespace(glm_route_align=lambda *args: sentinel)
    )
    args = (None, None, 8, 0, True, 2.5, 8, 64, 8)
    assert ops.quixicore_ops.glm_route_align(*args) is sentinel
    with pytest.raises(RuntimeError, match="requires rebuilt native"):
        ops.quixicore_ops.glm_route_align(*args, stable=True)


def test_stable_routing_flag_requires_canonical_diagnostic(monkeypatch):
    key = "SLIMSERVE_GLM53_STABLE_ROUTE"
    monkeypatch.delenv(key, raising=False)
    assert not canonical_moe.stable_route_enabled()
    for bad in ("yes", "true", "-1", "2"):
        monkeypatch.setenv(key, bad)
        with pytest.raises(ValueError, match="0 or 1"):
            canonical_moe.stable_route_enabled()
    monkeypatch.setenv(key, "1")
    monkeypatch.delenv("SLIMSERVE_GLM53_CANONICAL_MOE", raising=False)
    with pytest.raises(ValueError, match="requires canonical MoE"):
        canonical_moe.stable_route_enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_MOE", "1")
    monkeypatch.delenv("SLIMSERVE_GLM53_MODEL_JOURNAL", raising=False)
    with pytest.raises(ValueError, match="bounded model journal"):
        canonical_moe.stable_route_enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    assert canonical_moe.stable_route_enabled()


@pytest.mark.parametrize("stable", (False, True))
def test_router_dispatch_and_published_provenance(monkeypatch, stable):
    calls = []
    result = [
        torch.empty(8, 8),
        torch.empty(8, 8, dtype=torch.int32),
        torch.empty(512, dtype=torch.int32),
        torch.empty(64, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
    ]

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(route, "_STABLE_ALIGNMENT_DIAGNOSTIC", stable)
    monkeypatch.setattr(route.quixicore_ops, "glm_route_align", native)
    monkeypatch.setattr(torch.ops.vllm, "glm_route_align", route._glm_route_align_impl)
    monkeypatch.setattr(route, "_published", None)
    router = SimpleNamespace(
        top_k=8,
        e_score_correction_bias=torch.zeros(288),
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    logits = torch.empty(8, 288)
    weights, ids = route.route(router, logits)
    assert weights is result[0] and ids is result[1]
    assert len(calls) == 1 and calls[0][0][0] is logits
    assert calls[0][1] == ({"stable": True} if stable else {})
    aligned = route.consume(ids)
    assert aligned.canonical_assignment_order is stable
    assert aligned.sorted_token_ids is result[2]
    assert route.consume(ids) is None


@pytest.mark.parametrize(
    "kind,diagnostic,sort_calls,align_calls",
    (
        ("stable", True, 0, 0),
        ("atomic", True, 1, 0),
        ("wrong-block", True, 1, 1),
        ("wrong-ids", True, 1, 1),
        ("missing", True, 1, 1),
        ("stable", False, 0, 0),
        ("expert-map", False, 0, 1),
    ),
)
def test_sort_skip_depends_on_actual_reused_alignment(
    monkeypatch,
    kind,
    diagnostic,
    sort_calls,
    align_calls,
):
    monkeypatch.setattr(route, "_published", None)
    monkeypatch.setattr(marlin_moe, "_CANONICAL_MOE_DIAGNOSTIC", diagnostic)
    counts = {"sort": 0, "align": 0}
    ids = torch.zeros(8, 8, dtype=torch.int32)
    original = [
        torch.zeros(512, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    ]
    fallback = [t.clone() for t in original]
    if kind != "missing":
        route.publish(
            route.RoutingAlignment(
                ids.clone() if kind == "wrong-ids" else ids,
                *original,
                16 if kind == "wrong-block" else 8,
                canonical_assignment_order=kind != "atomic",
            )
        )

    def align(*args, **kwargs):
        counts["align"] += 1
        return fallback

    def sort(*args, **kwargs):
        counts["sort"] += 1
        assert args[0] is (fallback if align_calls else original)[0]

    def gemm(**kwargs):
        assert kwargs["sorted_token_ids"] is (fallback if align_calls else original)[0]
        assert kwargs["block_size_m"] == 8
        return torch.zeros(64, 4096, dtype=torch.bfloat16)

    monkeypatch.setattr(marlin_moe, "moe_align_block_size", align)
    monkeypatch.setattr(canonical_moe, "canonicalize", sort)
    monkeypatch.setattr(marlin_moe, "_fused_marlin_moe", gemm)
    monkeypatch.setattr(marlin_moe.ops, "moe_sum", lambda x, out, *args: out.zero_())
    out = marlin_moe.fused_marlin_moe(
        torch.zeros(8, 4096, dtype=torch.bfloat16),
        torch.empty(288, 256, 1, device="meta", dtype=torch.int32),
        torch.empty(288, 1, 8192, device="meta", dtype=torch.int32),
        None,
        None,
        torch.empty(0),
        torch.empty(0),
        torch.empty(8, 8),
        ids,
        scalar_types.float4_e2m1f.id,
        expert_map=torch.arange(288) if kind == "expert-map" else None,
    )
    assert out.shape == (8, 4096)
    assert counts == {"sort": sort_calls, "align": align_calls}
    assert route.consume(ids) is None
