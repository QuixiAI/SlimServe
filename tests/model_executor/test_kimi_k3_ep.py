# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization.gguf.fused_moe import GGUFMoEMethod
from vllm.model_executor.layers.quantization.gguf.params import (
    GGUFUninitializedWeightParameter,
)
from vllm.model_executor.model_loader.gguf_adapters.kimi_k3 import (
    KimiK3GGUFAdapter,
)
from vllm.models.kimi_k3.amd import linear as kimi_linear


def test_sequence_parallel_combine_satisfies_nonlinear_transform() -> None:
    runner = SimpleNamespace(
        routed_output_transform=SimpleNamespace(requires_reduced_input=True),
        moe_config=SimpleNamespace(is_sequence_parallel=True),
    )
    shared = torch.randn(2, 4)
    routed = torch.randn(2, 3)

    got_shared, got_routed, is_reduced = MoERunner._reduce_routed_transform_inputs(
        runner, shared, routed
    )

    assert got_shared is shared
    assert got_routed is routed
    assert is_reduced


@pytest.mark.parametrize(
    ("expert_ids", "expected"),
    [
        ((2, 3), torch.tensor([2, 3])),
        ((1, 3, 5), torch.tensor([1, 3, 5])),
    ],
)
def test_kimi_fused_gguf_stack_selects_local_experts(
    expert_ids: tuple[int, ...], expected: torch.Tensor
) -> None:
    adapter = KimiK3GGUFAdapter.__new__(KimiK3GGUFAdapter)
    adapter.local_expert_ids = expert_ids
    weight = torch.arange(6).view(6, 1, 1)

    [(name, local_weight)] = adapter.map_weights(
        [("language_model.layers.1.experts.0.w1.qweight", weight)]
    )

    assert name == f"language_model.layers.1.experts.{expert_ids[0]}.w1.qweight"
    torch.testing.assert_close(local_weight.flatten(), expected)


def test_kimi_fused_gguf_stack_rejects_invalid_expert_id() -> None:
    adapter = KimiK3GGUFAdapter.__new__(KimiK3GGUFAdapter)
    adapter.local_expert_ids = (6,)

    with pytest.raises(ValueError, match="exceeds fused stack"):
        list(
            adapter.map_weights(
                [
                    (
                        "language_model.layers.1.experts.0.w1.qweight",
                        torch.empty(6, 1, 1),
                    )
                ]
            )
        )


class _Gate(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states.sum(dim=-1, keepdim=True), None


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_hidden_states: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        self.last_hidden_states = hidden_states
        assert router_logits.shape[0] == hidden_states.shape[0]
        return hidden_states + 1


@pytest.mark.parametrize("num_tokens", [0, 5])
def test_kimi_sequence_parallel_moe_chunks_and_restores_tokens(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int
) -> None:
    moe = kimi_linear.KimiMoE.__new__(kimi_linear.KimiMoE)
    nn.Module.__init__(moe)
    moe.is_sequence_parallel = True
    moe.gate = _Gate()
    moe.experts = _Experts()

    def chunk(hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        padded = nn.functional.pad(hidden_states, (0, 0, 0, hidden_states.shape[0] % 2))
        return padded[: padded.shape[0] // 2]

    def gather(hidden_states: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0
        return torch.cat((hidden_states, hidden_states), dim=0)

    monkeypatch.setattr(kimi_linear, "sequence_parallel_chunk", chunk)
    monkeypatch.setattr(kimi_linear, "tensor_model_parallel_all_gather", gather)
    hidden_states = torch.arange(num_tokens * 3, dtype=torch.float32).view(
        num_tokens, 3
    )

    output = moe(hidden_states)

    assert output.shape == hidden_states.shape
    assert moe.experts.last_hidden_states is not None
    assert moe.experts.last_hidden_states.shape[0] == (num_tokens + 1) // 2


def test_gguf_moe_indexes_experts_with_the_map_not_the_aiter_mask() -> None:
    """The GGUF kernel indexes its local stack, so it needs the real map.

    `RoutedExperts.expert_map` degrades to AITER's 0/1 residency mask when
    AITER fused MoE is enabled, which would send every routed token to local
    expert 0 or 1.
    """
    expert_map = torch.full((8,), -1, dtype=torch.int32)
    expert_map[4:8] = torch.arange(4, dtype=torch.int32)
    expert_mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.int32)

    layer = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(layer)
    layer.register_buffer("_expert_map", expert_map, persistent=False)
    layer.register_buffer("expert_mask", expert_mask, persistent=False)
    layer.rocm_aiter_fmoe_enabled = True

    torch.testing.assert_close(layer.expert_map, expert_mask)
    torch.testing.assert_close(layer.global_to_local_expert_map, expert_map)

    captured: dict[str, torch.Tensor | None] = {}

    def fake_fused_moe_gguf(*args):
        captured["expert_map"] = args[-1]
        return args[0]

    method = GGUFMoEMethod.__new__(GGUFMoEMethod)
    method.moe = SimpleNamespace(
        activation_situ_beta=None, activation_situ_linear_beta=None
    )
    layer.apply_router_weight_on_input = False
    layer.w13_qweight = layer.w2_qweight = torch.empty(0)
    layer.w13_qweight_type = layer.w2_qweight_type = SimpleNamespace(weight_type=0)
    layer.activation = SimpleNamespace(value="situ")

    with mock.patch.dict(
        sys.modules,
        {
            "vllm.model_executor.layers.quantization.gguf": SimpleNamespace(
                fused_moe_gguf=fake_fused_moe_gguf
            )
        },
    ):
        method.apply(
            layer,
            torch.zeros(1, 2),
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int32),
            None,
            None,
        )

    torch.testing.assert_close(captured["expert_map"], expert_map)


@pytest.mark.parametrize(
    ("tp_rank", "tp_size", "expected"),
    [
        (0, 1, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (1, 2, [[2, 3], [6, 7]]),
    ],
)
def test_gguf_row_loader_uses_parameter_tp_metadata(
    tp_rank: int, tp_size: int, expected: list[list[int]]
) -> None:
    param = GGUFUninitializedWeightParameter(requires_grad=False)
    param.tp_rank = tp_rank
    param.tp_size = tp_size

    param.load_row_parallel_weight(torch.arange(8).reshape(2, 4))

    assert param.tolist() == expected


@pytest.mark.parametrize(
    ("tp_rank", "tp_size", "shard_size", "expected"),
    [
        (0, 1, 4, [[0, 1], [2, 3], [4, 5], [6, 7]]),
        (1, 2, 2, [[4, 5], [6, 7]]),
    ],
)
def test_gguf_merged_column_loader_uses_parameter_tp_metadata(
    tp_rank: int, tp_size: int, shard_size: int, expected: list[list[int]]
) -> None:
    """A `disable_tp=True` shared expert must keep its full-width gate/up rows.

    Kimi's sequence-parallel shared expert runs on a token shard with
    replicated weights, so narrowing it by the global TP rank would drop half
    of gate_up.
    """
    param = GGUFUninitializedWeightParameter(requires_grad=False)
    param.tp_rank = tp_rank
    param.tp_size = tp_size
    param.data_container = []
    param.shard_id = []
    param.shard_id_map = {}

    param.load_merged_column_weight(
        torch.arange(8).reshape(4, 2), shard_id=0, shard_size=shard_size
    )

    assert param.data_container[0].tolist() == expected
