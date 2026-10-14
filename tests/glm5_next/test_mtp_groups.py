# SPDX-License-Identifier: Apache-2.0
"""Distinct draft page sizes, rejection masks and stable graph slot buffers."""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.spec_decode.glm5_next as module
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs

MAIN = "model.layers.45.self_attn.mla_attn"
INDEX = "model.layers.45.self_attn.indexer.k_cache"


class Group:
    def __init__(self, backend, layer_names, kv_cache_spec, kv_cache_group_id):
        self.backend = backend
        self.layer_names = layer_names
        self.kv_cache_group_id = kv_cache_group_id
        self.spec = kv_cache_spec

    def create_metadata_builders(self, config, device, kernel_block_size=None):
        self.builder = SimpleNamespace(
            kv_cache_spec=self.spec.copy_with_new_block_size(kernel_block_size),
            build_for_drafting=lambda **kwargs: SimpleNamespace(
                common=kwargs["common_attn_metadata"],
                slot_mapping=kwargs["common_attn_metadata"].slot_mapping,
                draft_index=kwargs["draft_index"],
            ),
        )

    def get_metadata_builder(self):
        return self.builder


def fixture(monkeypatch):
    proposer = module.Glm5NextMTPProposer.__new__(module.Glm5NextMTPProposer)
    proposer.vllm_config = None
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_nextn_predict_layers=1)
    )
    proposer._draft_attn_layer_names = {MAIN, INDEX}
    proposer._slot_mapping_buffer = torch.empty(16, dtype=torch.int64)
    proposer._per_group_block_tables = {}
    proposer.device = torch.device("cpu")
    proposer.max_model_len = 192
    proposer.positions = torch.zeros(16, dtype=torch.int64)
    proposer._get_positions = lambda n: proposer.positions[:n]
    layers = {}
    for layer, name in ((MAIN, "QUIXICORE_MLA_SPARSE"), (INDEX, "GLM5_NEXT_INDEXER")):
        backend = SimpleNamespace(get_name=lambda name=name: name)
        layers[layer] = SimpleNamespace(get_attn_backend=lambda b=backend: b)
    monkeypatch.setattr(module, "get_layers_from_vllm_config", lambda *_: layers)
    monkeypatch.setattr(module, "AttentionGroup", Group)
    groups = []
    # Deliberately put the indexer first: it must not become the primary group.
    for layer, bs, head in ((INDEX, 128, 256), (MAIN, 64, 512)):
        spec = MLAAttentionSpec(
            block_size=bs, num_kv_heads=1, head_size=head, dtype=torch.bfloat16
        )
        groups.append(
            SimpleNamespace(
                layer_names=[layer],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=bs, kv_cache_specs={layer: spec}
                ),
            )
        )
    config = SimpleNamespace(kv_cache_groups=groups)
    proposer.initialize_attn_backend(config, kernel_block_sizes=[128, 64])
    return proposer, config


def common(slots, starts=(0, 2, 3)):
    qsl = torch.tensor(starts, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor([65, 130], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=3,
        max_query_len=2,
        max_seq_len=130,
        block_table_tensor=torch.tensor([[3, 5, 7], [11, 13, 17]], dtype=torch.int32),
        slot_mapping=torch.tensor(slots, dtype=torch.int64),
    )


def test_distinct_page_sizes_rejection_and_buffer_reuse(monkeypatch):
    p, _ = fixture(monkeypatch)
    assert p.kv_cache_gid == 1 and p.block_size == 64
    assert p.model_returns_tuple()
    p.set_per_group_block_table(
        0, torch.tensor([[19, 23], [31, 37]], dtype=torch.int32)
    )
    p.positions[:3] = torch.tensor([63, 64, 129])
    cm = common([255, -1, 1089])
    _, metadata = p.build_per_group_and_layer_attn_metadata(cm)
    torch.testing.assert_close(
        metadata[INDEX].slot_mapping, torch.tensor([2495, -1, 4737])
    )
    torch.testing.assert_close(metadata[MAIN].slot_mapping, cm.slot_mapping)
    slots = p._get_slot_mapping(5, cm.slot_mapping)
    assert slots[MAIN].data_ptr() != slots[INDEX].data_ptr()
    assert slots[MAIN][-2:].tolist() == slots[INDEX][-2:].tolist() == [-1, -1]
    pointers = {name: value.data_ptr() for name, value in slots.items()}

    # Rejection compacts token rows; request block-table ownership is unchanged.
    p.positions[:3] = torch.tensor([128, 129, 192])
    cm2 = common([448, 1089, -1], starts=(0, 1, 3))
    _, metadata = p.build_per_group_and_layer_attn_metadata(cm2, draft_index=1)
    torch.testing.assert_close(
        metadata[INDEX].slot_mapping, torch.tensor([2944, 4737, -1])
    )
    assert metadata[INDEX].draft_index == 1
    assert {n: v.data_ptr() for n, v in p._get_slot_mapping(5).items()} == pointers


def test_refuses_missing_group_table(monkeypatch):
    p, _ = fixture(monkeypatch)
    with pytest.raises(AssertionError, match="Missing GLM draft block table"):
        p.build_per_group_and_layer_attn_metadata(common([0, 1, 2]))


def test_refuses_missing_cache_owner(monkeypatch):
    p, config = fixture(monkeypatch)
    config.kv_cache_groups = config.kv_cache_groups[1:]
    with pytest.raises(AssertionError, match="Missing GLM draft caches"):
        p.initialize_attn_backend(config, kernel_block_sizes=[64])
