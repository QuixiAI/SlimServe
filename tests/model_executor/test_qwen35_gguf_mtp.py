# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused contracts for the Qwen3.5 GGUF fused GDN projections."""

from types import SimpleNamespace
from unittest import mock

from vllm.model_executor.model_loader.gguf_adapters.qwen35 import (
    Qwen35GGUFAdapter,
)
from vllm.model_executor.models.qwen3_5 import (
    _qwen35_mlp_kind,
)
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig
from vllm.transformers_utils.gguf_config_parser import GGUFConfigParser


def test_qwen35_gguf_mtp_maps_only_the_embedded_nextn_block():
    text_config = SimpleNamespace(num_hidden_layers=64)
    hf_config = SimpleNamespace(
        model_type="qwen3_5_mtp",
        architectures=["Qwen3_5MTP"],
        get_text_config=lambda: text_config,
    )
    adapter = Qwen35GGUFAdapter(hf_config)
    name_map = adapter.build_name_map(SimpleNamespace(hf_config=hf_config))

    # The draft shares the target's embedding and lm_head; it never loads
    # (or dequantizes) its own copies.
    assert "token_embd.weight" not in name_map
    assert "output.weight" not in name_map
    assert "output_norm.weight" not in name_map
    assert name_map["blk.64.nextn.eh_proj.weight"] == "mtp.fc.weight"
    assert name_map["blk.64.nextn.enorm.weight"] == ("mtp.pre_fc_norm_embedding.weight")
    assert name_map["blk.64.nextn.hnorm.weight"] == "mtp.pre_fc_norm_hidden.weight"
    assert name_map["blk.64.nextn.shared_head_norm.weight"] == "mtp.norm.weight"
    assert name_map["blk.64.attn_q.weight"] == ("mtp.layers.0.self_attn.q_proj.weight")
    assert name_map["blk.64.ffn_down.weight"] == ("mtp.layers.0.mlp.down_proj.weight")
    assert not any(name.startswith("blk.63.") for name in name_map)


def test_qwen35_mtp_model_type_preserves_dense_and_moe_mlp_dispatch():
    dense = Qwen3_5TextConfig(num_hidden_layers=1)
    dense.model_type = "qwen3_5_mtp"
    dense.architectures = ["Qwen3_5MTP"]
    assert _qwen35_mlp_kind(dense) == "dense"

    moe = Qwen3_5MoeTextConfig(num_hidden_layers=1)
    moe.model_type = "qwen3_5_mtp"
    moe.architectures = ["Qwen3_5MoeMTP"]
    assert _qwen35_mlp_kind(moe) == "moe"


def test_gguf_parser_detaches_cached_config_for_embedded_mtp_override():
    cached = Qwen3_5TextConfig(num_hidden_layers=64)
    cached.architectures = ["Qwen3_5ForCausalLM"]
    parser = GGUFConfigParser()

    with (
        mock.patch(
            "vllm.transformers_utils.gguf_config_parser.gguf_architecture",
            return_value="qwen35",
        ),
        mock.patch(
            "vllm.transformers_utils.gguf_qwen35.build_qwen35_config_from_gguf",
            return_value=cached,
        ),
    ):
        _, target = parser.parse("model.gguf", trust_remote_code=False)
        _, draft = parser.parse("model.gguf", trust_remote_code=False)

    draft.model_type = "qwen3_5_mtp"
    draft.architectures = ["Qwen3_5MTP"]

    assert target is not draft
    assert target is not cached
    assert target.model_type == "qwen3_5_text"
    assert target.architectures == ["Qwen3_5ForCausalLM"]


def test_qwen35_gguf_mtp_draft_runs_on_the_target_embedding_and_lm_head():
    """The GGUF draft never loads token_embd/output, so it must share.

    Qwen3_5MTP declares no has_own_embed_tokens / has_own_lm_head, which is
    what makes the V2 draft loader hand it the target's modules.
    """
    import torch.nn as nn

    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

    draft = Qwen3_5MTP.__new__(Qwen3_5MTP)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.embed_tokens = nn.Embedding(4, 2)
    draft.model.layers = nn.ModuleList([nn.Module()])
    draft.lm_head = nn.Linear(2, 4)
    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = nn.Embedding(4, 2)
    target.lm_head = nn.Linear(2, 4)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=object(), kv_cache_dtype=None
        )
    )

    with (
        mock.patch.object(eagle_utils, "get_model", return_value=draft),
        mock.patch.object(
            eagle_utils,
            "get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
    ):
        loaded = eagle_utils.load_eagle_model(target, config)

    assert loaded.model.embed_tokens is target.model.embed_tokens
    assert loaded.lm_head is target.lm_head


def _mtp_name_map(pp_size: int) -> dict[str, str]:
    from vllm.config import VllmConfig, set_current_vllm_config

    text_config = SimpleNamespace(num_hidden_layers=64)
    hf_config = SimpleNamespace(
        model_type="qwen3_5_mtp",
        architectures=["Qwen3_5MTP"],
        get_text_config=lambda: text_config,
    )
    vllm_config = VllmConfig()
    object.__setattr__(vllm_config.parallel_config, "pipeline_parallel_size", pp_size)
    adapter = Qwen35GGUFAdapter(hf_config)
    with set_current_vllm_config(vllm_config):
        return adapter.build_name_map(SimpleNamespace(hf_config=hf_config))


def test_qwen35_gguf_mtp_loads_its_own_embedding_only_under_pp():
    """PP=1: the runners share the target embedding, so token_embd stays
    unmapped. PP>1: neither runner shares it, so the draft must load it
    or embed through an uninitialised table. lm_head is shared either way
    (the target's lives on the draft's last rank)."""
    single = _mtp_name_map(1)
    assert "token_embd.weight" not in single
    assert "output.weight" not in single

    pipelined = _mtp_name_map(2)
    assert pipelined["token_embd.weight"] == "model.embed_tokens.weight"
    assert "output.weight" not in pipelined
    assert pipelined["blk.64.nextn.eh_proj.weight"] == "mtp.fc.weight"


def _toy_mtp_draft_and_target():
    import torch.nn as nn

    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    draft = Qwen3_5MTP.__new__(Qwen3_5MTP)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.embed_tokens = nn.Embedding(4, 2)
    draft.model.layers = nn.ModuleList([nn.Module()])
    draft.lm_head = nn.Linear(2, 4)
    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = nn.Embedding(4, 2)
    target.lm_head = nn.Linear(2, 4)
    return draft, target


def test_qwen35_gguf_mtp_v1_proposer_shares_target_embedding_and_lm_head():
    """V1 mirror of the V2 load_eagle_model sharing contract."""
    from vllm.v1.spec_decode import llm_base_proposer

    draft, target = _toy_mtp_draft_and_target()
    proposer = llm_base_proposer.SpecDecodeBaseProposer.__new__(
        llm_base_proposer.SpecDecodeBaseProposer
    )
    proposer.model = draft
    proposer.vllm_config = SimpleNamespace(speculative_config=None)
    proposer.use_local_argmax_reduction = False
    with mock.patch.object(
        llm_base_proposer,
        "get_pp_group",
        return_value=SimpleNamespace(world_size=1),
    ):
        proposer._maybe_share_embeddings(target)
        proposer._maybe_share_lm_head(target)

    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head


def test_qwen35_gguf_mtp_v1_proposer_keeps_draft_embedding_under_pp():
    """Under PP the V1 runner keeps the draft's own (loaded) embedding."""
    from vllm.v1.spec_decode import llm_base_proposer

    draft, target = _toy_mtp_draft_and_target()
    own = draft.model.embed_tokens
    proposer = llm_base_proposer.SpecDecodeBaseProposer.__new__(
        llm_base_proposer.SpecDecodeBaseProposer
    )
    proposer.model = draft
    proposer.vllm_config = SimpleNamespace(speculative_config=None)
    proposer.use_local_argmax_reduction = False
    with mock.patch.object(
        llm_base_proposer,
        "get_pp_group",
        return_value=SimpleNamespace(world_size=2),
    ):
        proposer._maybe_share_embeddings(target)
        proposer._maybe_share_lm_head(target)

    assert draft.model.embed_tokens is own
    assert draft.lm_head is target.lm_head
