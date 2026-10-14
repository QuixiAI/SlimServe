# SPDX-License-Identifier: Apache-2.0
"""CPU ownership/shape contracts; CUDA graph and serving gates are separate."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.glm5_next_indexer_workspace import (
    Glm5NextIndexerWorkspace,
)


def test_lazy_single_allocation_for_eleven_sequential_layers():
    owner = Glm5NextIndexerWorkspace()
    assert owner._decode_logits is None
    buffers = [owner.get_decode_logits(64, 128, torch.device("cpu")) for _ in range(11)]
    assert all(buffer is buffers[0] for buffer in buffers)
    assert buffers[0].dtype == torch.float32 and buffers[0].is_contiguous()
    buffers[0].fill_(3)
    assert torch.equal(buffers[-1], torch.full((64, 128), 3.0))
    unique_bytes = sum(
        t.numel() * t.element_size() for t in {id(t): t for t in buffers}.values()
    )
    assert unique_bytes == 64 * 128 * 4


def test_separate_model_owners_do_not_alias():
    a, b = Glm5NextIndexerWorkspace(), Glm5NextIndexerWorkspace()
    first = a.get_decode_logits(8, 128, torch.device("cpu"))
    second = b.get_decode_logits(8, 128, torch.device("cpu"))
    assert first.data_ptr() != second.data_ptr()
    first.fill_(1)
    second.fill_(2)
    assert first.sum().item() == first.numel()


@pytest.mark.parametrize(
    "rows,pools,device", [(9, 128, "cpu"), (8, 256, "cpu"), (8, 128, "meta")]
)
def test_owner_rejects_incompatible_reuse_without_reallocation(rows, pools, device):
    owner = Glm5NextIndexerWorkspace()
    original = owner.get_decode_logits(8, 128, torch.device("cpu"))
    with pytest.raises(ValueError, match="cannot change"):
        owner.get_decode_logits(rows, pools, torch.device(device))
    assert owner._decode_logits is original


@pytest.mark.parametrize("rows,pools", [(0, 128), (8, 0), (-1, 128)])
def test_invalid_shape_rejected_before_allocating(rows, pools):
    owner = Glm5NextIndexerWorkspace()
    with pytest.raises(ValueError, match="positive"):
        owner.get_decode_logits(rows, pools, torch.device("cpu"))
    assert owner._decode_logits is None


def test_config_defaults_and_independent_owners():
    assert Glm5NextIndexerWorkspace.from_config({}) is None
    assert (
        Glm5NextIndexerWorkspace.from_config(
            {"glm5_next_shared_indexer_scratch": False}
        )
        is None
    )
    extra = {"glm5_next_shared_indexer_scratch": True}
    first = Glm5NextIndexerWorkspace.from_config(extra)
    second = Glm5NextIndexerWorkspace.from_config(extra)
    assert isinstance(first, Glm5NextIndexerWorkspace) and first is not second
    with pytest.raises(ValueError, match="boolean"):
        Glm5NextIndexerWorkspace.from_config(
            {"glm5_next_shared_indexer_scratch": "false"}
        )


@pytest.mark.parametrize("speculative_tokens", [0, 1, 5])
def test_indexer_constructors_share_only_logits(speculative_tokens):
    from vllm.model_executor.layers.glm5_next_indexer import Glm5NextPooledIndexer

    class StubModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class CPUOwner(Glm5NextIndexerWorkspace):
        def __init__(self):
            super().__init__()
            self.devices = []

        def get_decode_logits(self, rows, pools, device):
            self.devices.append(device)
            return super().get_decode_logits(rows, pools, torch.device("cpu"))

    cfg = SimpleNamespace(
        index_n_heads=32,
        index_head_dim=128,
        index_topk=64,
        index_kpool=4,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        q_lora_rank=32,
        hidden_size=16,
    )
    vc = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(max_model_len=513),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        speculative_config=SimpleNamespace(num_speculative_tokens=speculative_tokens),
    )
    owner = CPUOwner()
    topk = torch.empty(4 * (1 + speculative_tokens), 96, dtype=torch.int32)
    with (
        patch(
            "vllm.model_executor.layers.glm5_next_indexer.ReplicatedLinear", StubModule
        ),
        patch(
            "vllm.model_executor.layers.glm5_next_indexer.Glm5NextIndexerCache",
            StubModule,
        ),
        patch("torch.cuda.current_device", return_value=0),
    ):
        layers = [
            Glm5NextPooledIndexer(
                vc, cfg, None, None, topk, prefix=f"layer{i}", workspace=owner
            )
            for i in range(11)
        ]
    assert len({id(layer.decode_logits) for layer in layers}) == 1
    assert layers[0].decode_logits.shape == (4 * (1 + speculative_tokens), 129)
    assert owner.devices == [torch.device("cuda", 0)] * 11
    assert len({id(layer.k_cache) for layer in layers}) == 11
    assert len({layer.index_kpool_compress_ape.data_ptr() for layer in layers}) == 11
