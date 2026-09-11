# SPDX-License-Identifier: Apache-2.0
"""Focused CPU regressions for serving integration review findings."""

import builtins
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_geometry_case_dispatch_does_not_import_benchmarks(monkeypatch):
    from slimserve import glm53_serving_diagnostic, rmsnorm_geometry

    original_import = builtins.__import__

    def without_benchmarks(name, *args, **kwargs):
        if name == "benchmarks" or name.startswith("benchmarks."):
            raise ImportError("benchmarks are not packaged")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_benchmarks)
    assert glm53_serving_diagnostic.cases(
        {"serving_schema": rmsnorm_geometry.SERVING_SCHEMA}
    ) == (
        ("control", "control"),
        ("geometry", "geometry"),
        ("return-control", "control"),
    )


@pytest.mark.parametrize("symbol", [False, True])
def test_router_projection_requires_its_native_symbol(monkeypatch, symbol):
    from vllm.model_executor.layers.fused_moe.router import gate_linear
    from vllm.quixicore import ops

    native = (
        SimpleNamespace(dsv4_projection_gemv=object()) if symbol else SimpleNamespace()
    )
    monkeypatch.setattr(ops, "_qc", lambda: native)
    assert gate_linear._quixicore_projection_available() is symbol


@pytest.mark.parametrize("generators,fp64", [(True, False), (False, True)])
def test_cuda_sampler_fallback_preserves_small_k_metadata(generators, fp64):
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

    expected = object()
    native = Mock(return_value=expected)
    owner = SimpleNamespace(forward_native=native, use_fp64_gumbel=fp64)
    logits, k = torch.empty(1, 16), torch.tensor([4])
    per_request = {0: object()} if generators else {}
    assert (
        TopKTopPSampler.forward_cuda(owner, logits, per_request, k, None, max_top_k=4)
        is expected
    )
    native.assert_called_once_with(logits, per_request, k, None, max_top_k=4)


@pytest.mark.parametrize("aborted", [False, True])
def test_routing_journal_drops_complete_aborted_frame(aborted):
    from vllm.v1.core.sched.scheduler import Scheduler

    journal = Mock()
    requests = {
        "second": SimpleNamespace(num_computed_tokens=1002, num_prompt_tokens=1000),
        "first": SimpleNamespace(num_computed_tokens=42, num_prompt_tokens=40),
    }
    if aborted:
        del requests["second"]
    owner = SimpleNamespace(_slimserve_routing_journal=journal, requests=requests)
    routed = SimpleNamespace(routing_data=object(), slot_mapping=object())
    req_ids, scheduled = ["second", "first"], {"first": 1, "second": 1}
    Scheduler._record_routing_journal(owner, routed, req_ids, scheduled)
    if aborted:
        journal.record.assert_not_called()
    else:
        journal.record.assert_called_once_with(
            routed.routing_data,
            routed.slot_mapping,
            req_ids,
            scheduled,
            {"second": 1002, "first": 42},
            {"second": 1000, "first": 40},
        )
