"""Cache topology of the GLM-5.3-Flash MTP proposer: sparse MLA + indexer owners,
each with its own block table and slot mapping."""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.spec_decode.glm5_next_mtp as glm_proposer
from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.spec_decode.glm5_next_mtp import Glm5NextMTPProposer

MLA_LAYER = "draft.model.layers.45.self_attn.attn"
INDEXER_LAYER = "draft.model.layers.45.self_attn.indexer.k_cache"
BLOCK = 64


class _FakeBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    def full_cls_name(self) -> tuple[str, str]:
        return (__name__, self.name)


class _FakeAttentionGroup:
    def __init__(self, backend, layer_names, kv_cache_spec, kv_cache_group_id):
        self.backend = backend
        self.layer_names = list(layer_names)
        self.kv_cache_spec = kv_cache_spec
        self.kv_cache_group_id = kv_cache_group_id
        self.kernel_block_size = None

    def create_metadata_builders(self, vllm_config, device, kernel_block_size=None):
        self.kernel_block_size = kernel_block_size

    def get_metadata_builder(self):
        # Report what the builder was handed: the group's block-table rows
        # and the slot mapping the cache insert would use.
        return SimpleNamespace(
            build_for_drafting=lambda common_attn_metadata, draft_index: (
                self.layer_names[0],
                common_attn_metadata.block_table_tensor.shape[0],
                common_attn_metadata.slot_mapping.tolist(),
            )
        )


def _specs():
    mla = MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
    )
    indexer = MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    return mla, indexer


def _proposer(monkeypatch: pytest.MonkeyPatch, uniform_groups: bool = True):
    mla, indexer = _specs()
    backends = {
        MLA_LAYER: _FakeBackend("QuixiCore"),
        INDEXER_LAYER: _FakeBackend("Idx"),
    }
    fake_layers = {
        name: SimpleNamespace(get_attn_backend=lambda b=b: b)
        for name, b in backends.items()
    }
    monkeypatch.setattr(
        glm_proposer, "get_layers_from_vllm_config", lambda *a, **k: fake_layers
    )
    monkeypatch.setattr(glm_proposer, "AttentionGroup", _FakeAttentionGroup)
    p = Glm5NextMTPProposer.__new__(Glm5NextMTPProposer)
    p.vllm_config = None
    p.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_nextn_predict_layers=1)
    )
    p.device = torch.device("cpu")
    p._draft_attn_layer_names = {MLA_LAYER, INDEXER_LAYER}
    p.kv_cache_gid = -1
    p.draft_attn_groups = []
    p.block_size = -1
    p._per_group_block_tables = {}
    p._per_group_slot_mappings = {}
    p._per_group_slot_mapping_buffers = {}
    p.max_positions = 64
    p._slot_mapping_buffer = torch.zeros(64, dtype=torch.int64)
    if uniform_groups:
        g0 = UniformTypeKVCacheSpecs(block_size=BLOCK, kv_cache_specs={MLA_LAYER: mla})
        g1 = UniformTypeKVCacheSpecs(
            block_size=BLOCK, kv_cache_specs={INDEXER_LAYER: indexer}
        )
    else:
        g0, g1 = mla, indexer
    # Indexer group listed first on purpose: the primary owner is found by
    # name, not by group order.
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["t.layers.3.self_attn.indexer.k_cache", INDEXER_LAYER],
                kv_cache_spec=g1,
            ),
            SimpleNamespace(
                layer_names=["t.layers.3.self_attn.attn", MLA_LAYER], kv_cache_spec=g0
            ),
        ]
    )
    return p, config


@pytest.mark.parametrize("uniform_groups", [True, False])
def test_primary_owner_is_the_sparse_mla_layer(monkeypatch, uniform_groups):
    p, config = _proposer(monkeypatch, uniform_groups)
    p.initialize_attn_backend(config, kernel_block_sizes=[BLOCK, BLOCK])
    assert p.kv_cache_gid == 1
    assert p.block_size == BLOCK
    assert [g.layer_names[0] for g in p.draft_attn_groups] == [MLA_LAYER, INDEXER_LAYER]
    assert [g.kv_cache_group_id for g in p.draft_attn_groups] == [1, 0]
    assert p.model_returns_tuple()


def test_indexer_group_gets_its_own_block_table_and_slot_mapping(monkeypatch):
    p, config = _proposer(monkeypatch)
    p.initialize_attn_backend(config, kernel_block_sizes=[BLOCK, BLOCK])
    idx_slots = torch.tensor([7, 8, 9], dtype=torch.int64)
    p.set_per_group_attn_metadata(0, torch.zeros(9, 4, dtype=torch.int32), idx_slots)
    mla_slots = torch.tensor([1, 2, 3], dtype=torch.int64)
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=3,
        block_table_tensor=torch.zeros(3, 8, dtype=torch.int32),
        slot_mapping=mla_slots,
    )
    per_group, per_layer = p.build_per_group_and_layer_attn_metadata(common)
    assert len(per_group) == 2
    assert per_layer[MLA_LAYER] == (MLA_LAYER, 3, [1, 2, 3])
    assert per_layer[INDEXER_LAYER] == (INDEXER_LAYER, 3, [7, 8, 9])
    sm = p._get_slot_mapping(3, mla_slots)
    assert sm[MLA_LAYER].tolist() == [1, 2, 3]
    assert sm[INDEXER_LAYER].tolist() == [7, 8, 9]


def test_next_step_recomputes_the_indexer_slots_from_its_own_table(monkeypatch):
    p, config = _proposer(monkeypatch)
    p.initialize_attn_backend(config, kernel_block_sizes=[BLOCK, BLOCK])
    p.uses_mrope = False
    p.max_model_len = 4096
    # Indexer group block table: request 0 -> blocks [10, 11], request 1 -> [20, 21].
    p.set_per_group_attn_metadata(
        0,
        torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int64),
    )
    positions = torch.tensor([BLOCK - 1, BLOCK + 5])  # old positions, one per request
    common = SimpleNamespace(slot_mapping=torch.tensor([100, 200]), max_seq_len=10)
    monkeypatch.setattr(
        Glm5NextMTPProposer.__mro__[2],
        "_update_positions_dependent_metadata",
        lambda self, pos, cm, b, ib, bs: pos + 1,
    )
    new_pos = p._update_positions_dependent_metadata(positions, common, 2, 2, BLOCK)
    assert new_pos.tolist() == [BLOCK, BLOCK + 6]
    # request 0 moved into its second block; request 1 stays in its second block
    assert p._per_group_slot_mappings[0].tolist() == [11 * BLOCK, 21 * BLOCK + 6]
    assert p._per_group_slot_mappings[1].tolist() == [100, 200]


def test_mismatched_group_block_sizes_are_rejected(monkeypatch):
    p, config = _proposer(monkeypatch)
    with pytest.raises(AssertionError, match="share one kernel block size"):
        p.initialize_attn_backend(config, kernel_block_sizes=[128, BLOCK])
