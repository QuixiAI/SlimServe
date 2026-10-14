# SPDX-License-Identifier: Apache-2.0
"""CPU ownership gate for pending external-router overlap integration."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.glm5_next_mhc_project import (
    prepare_router_projection,
    router_projection_enabled,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def make_moe():
    moe = nn.Module()
    moe.is_sequence_parallel = False
    moe.gate = nn.Linear(4096, 288, bias=False, dtype=torch.bfloat16)
    moe.gate.allow_cublas_router_gemm = True
    moe.gate.quant_method = UnquantizedLinearMethod()
    runner = MoERunner.__new__(MoERunner)
    nn.Module.__init__(runner)
    runner.gate = moe.gate
    runner.routed_input_transform = None
    runner.shared_expert_gate = None
    runner._fse_fuse_gate = runner.enable_dbo = False
    moe.experts = runner
    return moe


def test_router_config_is_separate_strict_opt_in():
    assert not router_projection_enabled({})
    base = {"glm5_next_mhc_projection_overlap": True}
    assert not router_projection_enabled(base)
    flag = "glm5_next_mhc_router_projection_overlap"
    assert router_projection_enabled(base | {flag: True})
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="boolean"):
            router_projection_enabled(base | {flag: invalid})
    with pytest.raises(ValueError, match="requires"):
        router_projection_enabled({flag: True})


def test_externalization_preserves_weights_and_canonical_loader_names():
    moe = make_moe()
    before = dict(moe.named_parameters())
    assert moe.experts.is_internal_router
    assert prepare_router_projection(moe)
    assert not moe.experts.is_internal_router
    after = dict(moe.named_parameters())
    assert after.keys() == before.keys() == {"gate.weight"}
    assert after["gate.weight"] is before["gate.weight"]
    assert prepare_router_projection(moe)  # idempotent


@pytest.mark.parametrize(
    "incompatible",
    ["sp", "transform", "shared_gate", "fse", "dbo", "different_gate", "dtype", "bias"],
)
def test_incompatible_runner_is_unchanged(incompatible):
    moe = make_moe()
    runner = moe.experts
    if incompatible == "sp":
        moe.is_sequence_parallel = True
    elif incompatible == "transform":
        runner.routed_input_transform = nn.Identity()
    elif incompatible == "shared_gate":
        runner.shared_expert_gate = nn.Identity()
    elif incompatible == "fse":
        runner._fse_fuse_gate = True
    elif incompatible == "dbo":
        runner.enable_dbo = True
    elif incompatible == "different_gate":
        runner.gate = nn.Identity()
    elif incompatible == "dtype":
        moe.gate.weight = nn.Parameter(moe.gate.weight.float())
    elif incompatible == "bias":
        moe.gate.bias = nn.Parameter(torch.zeros(288))
    before = runner.gate
    assert not prepare_router_projection(moe)
    assert runner.gate is before


def test_runner_consumes_external_logits_without_skipping_shared_sync():
    moe = make_moe()
    runner = moe.experts
    x = torch.randn(2, 4096, dtype=torch.bfloat16)
    logits = torch.randn(2, 288)
    gate_call = Mock(return_value=(logits, None))
    moe.gate.forward = gate_call
    runner.moe_config = SimpleNamespace(num_experts=288)
    runner.routed_experts = SimpleNamespace(_ensure_moe_quant_config_init=Mock())
    runner._maybe_sync_shared_experts_stream = Mock()
    runner._sequence_parallel_context = nullcontext
    runner._maybe_dispatch = lambda h, r: (h, r)
    runner._apply_quant_method = Mock(return_value=(None, x))
    runner._maybe_combine = lambda shared, hidden: hidden
    runner._forward_impl(x, x, x)
    assert gate_call.call_count == 1
    assert runner._apply_quant_method.call_args.kwargs["router_logits"] is logits
    assert prepare_router_projection(moe)
    supplied = torch.randn_like(logits)
    runner._forward_impl(x, supplied, x)
    assert gate_call.call_count == 1  # not projected a second time
    assert runner._maybe_sync_shared_experts_stream.call_count == 2
    assert runner._apply_quant_method.call_args.kwargs["router_logits"] is supplied
