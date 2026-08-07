# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from gguf import GGUFWriter

from vllm.model_executor.layers import vocab_parallel_embedding
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.gguf_adapters import kimi_k3_dspark
from vllm.model_executor.model_loader.gguf_adapters.kimi_k3_dspark import (
    KimiK3DSparkGGUFAdapter,
)
from vllm.model_executor.models import qwen3_dflash
from vllm.transformers_utils import gguf_config_parser
from vllm.transformers_utils.gguf_config_parser import GGUFConfigParser
from vllm.transformers_utils.gguf_kimi_k3_dspark import (
    build_kimi_k3_dspark_config_from_gguf,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    _draft_validation_parallel_config,
)


def test_replicated_vocab_layers_skip_tensor_parallel_collectives(monkeypatch):
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        lambda _: pytest.fail("a replicated embedding must not be reduced"),
    )
    embedding = SimpleNamespace(
        tp_size=1,
        quant_method=SimpleNamespace(embedding=lambda _layer, inputs: inputs.float()),
    )
    token_ids = torch.tensor([1, 2, 3])

    output = VocabParallelEmbedding.forward(embedding, token_ids)

    assert torch.equal(output, token_ids.float())

    full_logits = torch.arange(6).reshape(2, 3)
    processor = SimpleNamespace(
        org_vocab_size=3,
        _apply_head=lambda *_: full_logits,
        _gather_logits=lambda _: pytest.fail("a replicated head must not be gathered"),
    )
    head = SimpleNamespace(tp_size=1)

    logits = LogitsProcessor._get_logits(processor, torch.empty(0), head, None)

    assert torch.equal(logits, full_logits)


def test_replicated_draft_validates_as_tp1_without_mutating_target_parallelism():
    target_parallel = SimpleNamespace(tensor_parallel_size=6)
    vllm_config = SimpleNamespace(
        parallel_config=target_parallel,
        speculative_config=SimpleNamespace(replicate_draft_backbone=True),
    )

    validation_parallel = _draft_validation_parallel_config(vllm_config)

    assert validation_parallel.tensor_parallel_size == 1
    assert target_parallel.tensor_parallel_size == 6


def test_gguf_config_parser_rejects_unregistered_architectures(monkeypatch):
    monkeypatch.setattr(gguf_config_parser, "gguf_architecture", lambda _: "llama")

    with pytest.raises(ValueError, match="Unsupported GGUF architecture: llama"):
        GGUFConfigParser().parse("other.gguf", trust_remote_code=False)


def test_kimi_dspark_config_comes_from_the_embedded_source_config(tmp_path):
    path = tmp_path / "Kimi-K3-DSpark-Q8_0.gguf"
    source = {
        "hidden_size": 7168,
        "intermediate_size": 14336,
        "num_hidden_layers": 5,
        "num_attention_heads": 64,
        "num_key_value_heads": 16,
        "vocab_size": 163840,
        "max_position_embeddings": 1048576,
        "dflash_config": {"target_layer_ids": [7, 23, 51, 67, 83]},
    }
    writer = GGUFWriter(path, "dflash-draft")
    writer.add_name("Kimi-K3-DSpark-Q8_0")
    writer.add_string("dflash-draft.dflash.target.repository", "moonshotai/Kimi-K3")
    writer.add_string("dflash-draft.source.config_json", json.dumps(source))
    writer.add_uint32("dflash-draft.dflash.block_size", 7)
    writer.add_uint32("dflash-draft.vocab_size", 163840)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    config = build_kimi_k3_dspark_config_from_gguf(str(path))

    assert config.architectures == ["Qwen3DSparkModel"]
    assert config.dspark_block_size == 7
    assert config.draft_vocab_size == 163840
    assert config.dflash_config["target_layer_ids"] == [7, 23, 51, 67, 83]


def test_kimi_dspark_adapter_maps_only_its_draft_tensor_layout(monkeypatch):
    tensors = [
        SimpleNamespace(name="dflash.dspark.markov.w1"),
        SimpleNamespace(name="dflash.fc.weight"),
        SimpleNamespace(name="output_norm.weight"),
        SimpleNamespace(name="blk.3.attn_q.weight"),
        SimpleNamespace(name="blk.3.ffn_down.weight"),
        SimpleNamespace(name="dflash.dspark.confidence.weight"),
    ]
    monkeypatch.setattr(
        kimi_k3_dspark, "gguf_reader", lambda _: SimpleNamespace(tensors=tensors)
    )
    adapter = KimiK3DSparkGGUFAdapter(SimpleNamespace())

    names = adapter.build_name_map(SimpleNamespace(model="draft.gguf"))

    assert names == {
        "dflash.dspark.markov.w1": "markov_head.markov_w1.weight",
        "dflash.fc.weight": "fc.weight",
        "output_norm.weight": "norm.weight",
        "blk.3.attn_q.weight": "layers.3.self_attn.q_proj.weight",
        "blk.3.ffn_down.weight": "layers.3.mlp.down_proj.weight",
    }


def test_dflash_context_projection_supports_packed_qkv(monkeypatch):
    class PackedQKV:
        qweight = object()

        def __init__(self, values):
            self.values = torch.tensor(values, dtype=torch.float32)

        def __call__(self, inputs):
            return self.values.expand(inputs.shape[0], -1), None

    layers = [
        SimpleNamespace(
            q_size=1,
            qkv_proj=PackedQKV([0, 1, 2]),
            k_norm=SimpleNamespace(weight=SimpleNamespace(data=torch.ones(1))),
        ),
        SimpleNamespace(
            q_size=1,
            qkv_proj=PackedQKV([0, 3, 4]),
            k_norm=SimpleNamespace(weight=SimpleNamespace(data=torch.ones(1))),
        ),
    ]
    model = SimpleNamespace(
        hidden_norm=SimpleNamespace(weight=SimpleNamespace(data=torch.ones(1))),
        _rms_norm_eps=1e-6,
    )
    monkeypatch.setattr(
        qwen3_dflash.ops,
        "rms_norm",
        lambda output, inputs, weight, epsilon: output.copy_(inputs),
    )

    qwen3_dflash.DFlashQwen3Model._build_context_kv_buffers(model, layers, False)
    keys, values = qwen3_dflash.DFlashQwen3Model._project_context_kv(
        model,
        torch.ones(2, 1),
        num_ctx=2,
        num_layers=2,
        num_kv_heads=1,
        head_dim=1,
    )

    assert model._fused_kv_weight is None
    assert model._context_qkv_projections == layers
    assert keys[:, :, 0, 0].tolist() == [[1, 1], [3, 3]]
    assert values[:, :, 0, 0].tolist() == [[2, 2], [4, 4]]


def test_dflash_skips_disabled_draft_cudagraph_capture():
    speculator = SimpleNamespace(
        speculative_config=SimpleNamespace(disable_draft_cudagraphs=True),
        _speculator_name="DSpark",
    )

    DFlashSpeculator.capture(speculator)


def test_kimi_k3_kda_cache_reserves_speculative_conv_history():
    conv_shape, recurrent_shape = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=8,
        num_heads=64,
        head_dim=128,
        conv_kernel_size=4,
        num_spec=7,
    )

    assert conv_shape == MambaStateShapeCalculator._orient_conv_shape(3072, 10)
    assert recurrent_shape == (8, 128, 128)
