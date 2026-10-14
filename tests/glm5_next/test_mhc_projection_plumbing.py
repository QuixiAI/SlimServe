# SPDX-License-Identifier: Apache-2.0
"""Guard and caller contracts for the optional mHC consumer projection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.glm5_next_mhc_project import (
    plain_bf16_projection,
    projection_enabled,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE


def test_config_opt_in_and_strict_boolean():
    args = dict(
        sm80=True, hidden_size=4096, hc_mult=4, dtype=torch.bfloat16, lora=False
    )
    assert not projection_enabled({}, **args)
    assert projection_enabled({"glm5_next_mhc_projection_overlap": True}, **args)
    for value in ("true", "false", 1, None):
        with pytest.raises(ValueError, match="boolean"):
            projection_enabled({"glm5_next_mhc_projection_overlap": value}, **args)
    for name, value in (
        ("sm80", False),
        ("hidden_size", 2048),
        ("hc_mult", 8),
        ("dtype", torch.float16),
        ("lora", True),
    ):
        assert not projection_enabled(
            {"glm5_next_mhc_projection_overlap": True}, **(args | {name: value})
        )


@pytest.mark.parametrize("bad", [None, "quant", "bias", "gather", "dtype", "stride"])
def test_projection_does_not_bypass_linear_semantics(bad):
    layer = SimpleNamespace(
        weight=torch.empty(32, 4096, dtype=torch.bfloat16),
        quant_method=UnquantizedLinearMethod(),
        bias=None,
        gather_output=False,
    )
    if bad == "quant":
        layer.quant_method = object()
    elif bad == "bias":
        layer.bias = torch.empty(32)
    elif bad == "gather":
        layer.gather_output = True
    elif bad == "dtype":
        layer.weight = layer.weight.float()
    elif bad == "stride":
        layer.weight = torch.empty(4096, 32, dtype=torch.bfloat16).T
    assert plain_bf16_projection(layer) == (bad is None)


@pytest.mark.parametrize("supplied", [False, True])
def test_precomputed_router_preserves_expert_call_without_double_projection(supplied):
    x = torch.randn(8, 4096)
    logits = torch.randn(8, 288)
    experts = Mock(return_value=x)
    experts.is_internal_router = False
    gate = Mock(return_value=(logits, None))
    gate.out_dtype = torch.float32
    layer = SimpleNamespace(
        is_sequence_parallel=False, n_routed_experts=288, gate=gate, experts=experts
    )
    actual = DeepseekV2MoE.forward(layer, x, router_logits=logits if supplied else None)
    assert torch.equal(actual, x)
    assert gate.call_count == (0 if supplied else 1)
    # Existing forward reshapes through a zero-copy view.
    assert experts.call_args.kwargs["hidden_states"].data_ptr() == x.data_ptr()
    assert torch.equal(experts.call_args.kwargs["hidden_states"], x)
    assert experts.call_args.kwargs["router_logits"] is logits


@pytest.mark.parametrize("sp,internal", [(True, False), (False, True)])
def test_precomputed_router_rejects_incompatible_semantics(sp, internal):
    layer = SimpleNamespace(
        is_sequence_parallel=sp, experts=SimpleNamespace(is_internal_router=internal)
    )
    with pytest.raises(AssertionError):
        DeepseekV2MoE.forward(
            layer, torch.empty(8, 4096), router_logits=torch.empty(8, 288)
        )
