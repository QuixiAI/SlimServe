"""Cache topology of the GLM-5.3-Flash MTP proposer: sparse MLA + indexer owners."""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.spec_decode.glm5_next_mtp as glm_proposer
from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.spec_decode.glm5_next_mtp import Glm5NextMTPProposer

MLA_LAYER = "draft.model.layers.45.self_attn.attn"
INDEXER_LAYER = "draft.model.layers.45.self_attn.indexer.k_cache"
SCHED_BLOCK = 64


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
        return SimpleNamespace(
            build_for_drafting=lambda common_attn_metadata, draft_index: (
                self.layer_names[0],
                common_attn_metadata.block_table_tensor.shape[0],
            )
        )


def _specs():
    mla = MLAAttentionSpec(
        block_size=SCHED_BLOCK, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
    )
    indexer = MLAAttentionSpec(
        block_size=SCHED_BLOCK, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    return mla, indexer


def _proposer(monkeypatch: pytest.MonkeyPatch, uniform_groups: bool):
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
    proposer = Glm5NextMTPProposer.__new__(Glm5NextMTPProposer)
    proposer.vllm_config = None
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_nextn_predict_layers=1)
    )
    proposer.device = torch.device("cpu")
    proposer._draft_attn_layer_names = {MLA_LAYER, INDEXER_LAYER}
    proposer.kv_cache_gid = -1
    proposer.draft_attn_groups = []
    proposer.block_size = -1
    proposer._per_group_block_tables = {}
    if uniform_groups:
        g0 = UniformTypeKVCacheSpecs(
            block_size=SCHED_BLOCK, kv_cache_specs={MLA_LAYER: mla}
        )
        g1 = UniformTypeKVCacheSpecs(
            block_size=SCHED_BLOCK, kv_cache_specs={INDEXER_LAYER: indexer}
        )
    else:
        g0, g1 = mla, indexer
    # The indexer group is listed first on purpose: the main owner must be
    # found by name, not by group order.
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
    return proposer, config


@pytest.mark.parametrize("uniform_groups", [True, False])
def test_main_owner_is_the_sparse_mla_layer(monkeypatch, uniform_groups):
    proposer, config = _proposer(monkeypatch, uniform_groups)
    proposer.initialize_attn_backend(config, kernel_block_sizes=[128, 64])
    assert proposer.kv_cache_gid == 1
    assert proposer.block_size == 64
    assert [g.layer_names[0] for g in proposer.draft_attn_groups] == [
        MLA_LAYER,
        INDEXER_LAYER,
    ]
    assert [g.kv_cache_group_id for g in proposer.draft_attn_groups] == [1, 0]
    assert [g.kernel_block_size for g in proposer.draft_attn_groups] == [64, 128]
    assert proposer.model_returns_tuple()


def test_indexer_metadata_uses_its_own_groups_block_table(monkeypatch):
    proposer, config = _proposer(monkeypatch, True)
    proposer.initialize_attn_backend(config, kernel_block_sizes=[128, 64])
    proposer.set_per_group_block_table(0, torch.zeros(9, 4, dtype=torch.int32))
    common = SimpleNamespace(
        num_reqs=3, block_table_tensor=torch.zeros(3, 8, dtype=torch.int32)
    )
    per_group, per_layer = proposer.build_per_group_and_layer_attn_metadata(common)
    assert per_layer[MLA_LAYER] == (MLA_LAYER, 3)  # main group: the proposer's table
    assert per_layer[INDEXER_LAYER] == (
        INDEXER_LAYER,
        3,
    )  # staged table, sliced to num_reqs
    assert len(per_group) == 2


def test_missing_staged_block_table_is_an_assertion(monkeypatch):
    proposer, config = _proposer(monkeypatch, True)
    proposer.initialize_attn_backend(config, kernel_block_sizes=[128, 64])
    common = SimpleNamespace(
        num_reqs=1, block_table_tensor=torch.zeros(1, 8, dtype=torch.int32)
    )
    with pytest.raises(AssertionError, match="group 0"):
        proposer.build_per_group_and_layer_attn_metadata(common)
