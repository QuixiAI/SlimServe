# SPDX-License-Identifier: Apache-2.0
"""CPU metadata proof; actual Marlin/graph parity is a separate GPU gate."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.singleton_alignment import (
    SingletonAlignment,
    make_singleton_alignment,
)


@pytest.mark.parametrize("topk", [1, 2, 8, 16])
@pytest.mark.parametrize("block", [8, 16])
def test_exact_routed_row_mapping_without_sorting(topk, block):
    owner = SingletonAlignment(topk, block, "cpu")
    ids = torch.randperm(288, dtype=torch.int32)[:topk].view(1, -1)
    for _ in range(4):
        rows, experts, padded = owner.get(ids)
        assert padded.item() == topk * block
        assert experts.data_ptr() == ids.data_ptr()
        assert rows is owner.rows
        routed = [
            (int(experts[b]), int(row))
            for b, group in enumerate(rows.reshape(-1, block))
            for row in group
            if row < topk
        ]
        assert sorted(routed) == sorted((int(e), j) for j, e in enumerate(ids[0]))
        assert (rows.reshape(-1, block)[:, 1:] == topk).all()
        ids.copy_(torch.randperm(288, dtype=torch.int32)[:topk].view(1, -1))


@pytest.mark.parametrize(
    "shape,dtype", [((2, 8), torch.int32), ((1, 7), torch.int32), ((1, 8), torch.int64)]
)
def test_rejects_incompatible_metadata(shape, dtype):
    owner = SingletonAlignment(8, 8, "cpu")
    with pytest.raises(ValueError, match="one contiguous int32"):
        owner.get(torch.zeros(shape, dtype=dtype))


def test_rejects_strided_ids():
    owner = SingletonAlignment(8, 8, "cpu")
    with pytest.raises(ValueError, match="one contiguous int32"):
        owner.get(torch.zeros(1, 16, dtype=torch.int32)[:, ::2])


@pytest.mark.parametrize(
    "disabled",
    [None, "flag", "arch", "quant", "input", "hidden", "experts", "ep", "lora"],
)
def test_factory_is_opt_in_and_scoped(disabled):
    moe = SimpleNamespace(
        hidden_dim=4096,
        num_experts=288,
        experts_per_token=8,
        device="cpu",
        is_lora_enabled=disabled == "lora",
        moe_parallel_config=SimpleNamespace(use_ep=disabled == "ep"),
    )
    if disabled == "hidden":
        moe.hidden_dim = 2048
    if disabled == "experts":
        moe.num_experts = 256
    quant = SimpleNamespace(use_nvfp4_w4a16=disabled != "quant")
    extra = {} if disabled == "flag" else {"glm5_next_singleton_marlin_alignment": True}
    owner = make_singleton_alignment(
        moe,
        quant,
        extra,
        torch.int8 if disabled == "input" else None,
        disabled != "arch",
    )
    assert (owner is not None) == (disabled is None)


def test_rejects_non_boolean_flag():
    with pytest.raises(ValueError, match="must be a boolean"):
        make_singleton_alignment(
            None, None, {"glm5_next_singleton_marlin_alignment": "false"}, None, False
        )


@pytest.mark.parametrize(
    "fallback",
    [None, "rows", "dtype", "owner", "ep", "global_experts", "int8", "strided"],
)
def test_public_marlin_dispatch_preserves_general_fallback(monkeypatch, fallback):
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe as module
    from vllm.scalar_type import scalar_types

    rows, topk, k, n, e = (8 if fallback == "rows" else 1), 8, 256, 128, 16
    x = torch.zeros(rows, k, dtype=torch.bfloat16)
    ids = torch.arange(topk, dtype=torch.int32).repeat(rows, 1)
    if fallback == "dtype":
        ids = ids.long()
    if fallback == "strided":
        ids = torch.arange(2 * topk, dtype=torch.int32).view(1, -1)[:, ::2]
    owner = None if fallback == "owner" else SingletonAlignment(topk, 8, "cpu")
    generic_calls = []

    def generic(*args, **kwargs):
        generic_calls.append(True)
        return (
            torch.zeros(rows * topk * 8, dtype=torch.int32),
            torch.zeros(rows * topk, dtype=torch.int32),
            torch.tensor([rows * topk * 8], dtype=torch.int32),
        )

    def gemms(**kwargs):
        if fallback is None:
            assert kwargs["sorted_token_ids"] is owner.rows
            assert kwargs["expert_ids"].data_ptr() == ids.data_ptr()
            assert kwargs["num_tokens_post_padded"] is owner.padded
        return torch.zeros(rows * topk, k, dtype=x.dtype)

    monkeypatch.setattr(module, "moe_align_block_size", generic)
    monkeypatch.setattr(module, "_fused_marlin_moe", gemms)
    result = module.fused_marlin_moe(
        x,
        torch.empty(e, k // 16, 2 * n * 2, dtype=torch.int32),
        torch.empty(e, n // 16, k * 2, dtype=torch.int32),
        None,
        None,
        torch.empty(0),
        torch.empty(0),
        torch.ones(rows, topk),
        ids,
        scalar_types.uint4.id,
        expert_map=torch.arange(e) if fallback == "ep" else None,
        global_num_experts=2 * e if fallback == "global_experts" else e,
        input_dtype=torch.int8 if fallback == "int8" else None,
        singleton_alignment=owner,
        moe_sum=lambda value, output, *_: value.sum(dim=1),
    )
    assert result.shape == (rows, k)
    assert bool(generic_calls) == (fallback is not None)
