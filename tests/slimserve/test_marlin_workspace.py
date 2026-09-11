# SPDX-License-Identifier: Apache-2.0
"""Marlin lock reuse follows layer/ubatch ownership, never a global device."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts import marlin_moe as marlin


@pytest.fixture
def owners(monkeypatch):
    state = SimpleNamespace(ubatch=0, allocations=[])
    config = SimpleNamespace(
        additional_config={}, parallel_config=SimpleNamespace(use_ubatching=True)
    )
    monkeypatch.setattr(marlin, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(marlin, "get_marlin_input_dtype", lambda: None)
    monkeypatch.setattr(
        marlin, "current_platform", SimpleNamespace(is_cuda=lambda: False)
    )
    monkeypatch.setattr(marlin, "dbo_current_ubatch_id", lambda: state.ubatch)

    def allocate(device, blocks):
        assert blocks == 4
        workspace = torch.zeros(16, dtype=torch.int32)
        state.allocations.append((device, workspace))
        return workspace

    monkeypatch.setattr(marlin, "marlin_make_workspace_new", allocate)

    def make(batched=False):
        quant = int4_w4a16_moe_quant_config(torch.ones(1), torch.ones(1))
        cls = marlin.BatchedMarlinExperts if batched else marlin.MarlinExperts
        kwargs = dict(max_num_tokens=8, num_dispatchers=1) if batched else {}
        return cls(SimpleNamespace(), quant, **kwargs)

    return state, config, make


def test_workspace_is_owned_by_layer_device_and_ubatch(owners):
    state, _, make = owners
    first, second = make(), make()
    device = torch.device("cuda:0")
    workspace = first._marlin_workspace(device)
    assert first._marlin_workspace(device) is workspace
    assert second._marlin_workspace(device) is not workspace
    assert first._marlin_workspace(torch.device("cuda:1")) is not workspace
    state.ubatch = 1
    other_ubatch = first._marlin_workspace(device)
    assert other_ubatch is not workspace
    assert first._marlin_workspace(device) is other_ubatch
    state.ubatch = 0
    assert first._marlin_workspace(device) is workspace
    assert len(state.allocations) == 4


@pytest.mark.parametrize("config_present", [False, True])
def test_non_ubatched_owner_never_queries_thread_context(
    owners, monkeypatch, config_present
):
    _, config, make = owners
    config.parallel_config.use_ubatching = False
    if not config_present:
        monkeypatch.setattr(marlin, "get_current_vllm_config_or_none", lambda: None)

    def unexpected_context_query():
        pytest.fail("non-ubatched serving must not consult DBO thread state")

    monkeypatch.setattr(marlin, "dbo_current_ubatch_id", unexpected_context_query)
    owner = make()
    device = torch.device("cuda:0")
    assert owner._marlin_workspace(device) is owner._marlin_workspace(device)


@pytest.mark.parametrize("path", ["standard", "lora", "batched"])
def test_every_expert_path_passes_owned_workspace(owners, monkeypatch, path):
    state, _, make = owners
    owner = make(batched=path == "batched")
    if path == "lora":
        owner.set_lora_context(object())
    observed = []

    def run(**kwargs):
        observed.append(kwargs["workspace"])
        return kwargs["output"]

    monkeypatch.setattr(marlin, "fused_marlin_moe", run)
    monkeypatch.setattr(marlin, "batched_fused_marlin_moe", run)
    tensor = torch.empty(2, 32)
    for state.ubatch in (0, 1, 0):
        owner.apply(
            output=tensor,
            hidden_states=tensor,
            w1=tensor,
            w2=tensor,
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            activation=marlin.MoEActivation.SILU,
            global_num_experts=1,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=tensor,
            workspace2=tensor,
            expert_tokens_meta=SimpleNamespace(expert_num_tokens=torch.tensor([2])),
            apply_router_weight_on_input=False,
        )
    assert observed[0] is observed[2]
    assert observed[0] is not observed[1]
    assert len(state.allocations) == 2


@pytest.mark.parametrize("explicit", [False, True])
def test_standalone_call_uses_fresh_or_explicit_workspace(
    owners, monkeypatch, explicit
):
    state, _, _ = owners
    observed = []

    def gemm(
        a,
        output,
        b,
        bias,
        scales,
        a_scales,
        global_scale,
        zeros,
        g_idx,
        sort_indices,
        workspace,
        *args,
        **kwargs,
    ):
        observed.append(workspace)
        return output

    monkeypatch.setattr(marlin.ops, "moe_wna16_marlin_gemm", gemm)
    monkeypatch.setattr(marlin, "marlin_moe_intermediate_size", lambda *_: 16)
    workspace = torch.zeros(16, dtype=torch.int32) if explicit else None
    tensor = torch.empty(2, 32)
    for _ in range(2):
        marlin._fused_marlin_moe(
            hidden_states=tensor,
            w1=tensor,
            w2=tensor,
            bias1=None,
            bias2=None,
            w1_scale=tensor,
            w2_scale=tensor,
            topk_weights=torch.ones(2, 1),
            num_topk=1,
            quant_type=marlin.scalar_types.uint4b8,
            apply_router_weight_on_input=False,
            expert_map=None,
            block_size_m=8,
            sorted_token_ids=torch.arange(2, dtype=torch.int32),
            expert_ids=torch.zeros(2, dtype=torch.int32),
            num_tokens_post_padded=torch.tensor([2], dtype=torch.int32),
            activation_func=lambda *args, **kwargs: None,
            workspace=workspace,
        )
    assert observed[0] is observed[1]
    assert observed[2] is observed[3]
    if explicit:
        assert all(item is workspace for item in observed)
        assert not state.allocations
    else:
        assert observed[0] is not observed[2]
        assert len(state.allocations) == 2
