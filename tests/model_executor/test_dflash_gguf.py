# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the standardized DeepSeek-V4 0731 DSpark GGUF."""

from collections import Counter
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from vllm.config import LoadConfig
from vllm.model_executor.model_loader.gguf_adapters import dflash as adapter_module
from vllm.model_executor.model_loader.gguf_adapters.base import GGUFLoadSpec
from vllm.model_executor.model_loader.gguf_adapters.default import GGUFWeightsAdapter
from vllm.model_executor.model_loader.gguf_adapters.dflash import DFlashGGUFAdapter
from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead
from vllm.transformers_utils import gguf_dflash
from vllm.transformers_utils.gguf_dflash import (
    EXPECTED_METADATA,
    EXPECTED_TYPE_COUNTS,
    build_dflash_config_from_gguf,
    dflash_tensor_specs,
    validate_dflash_reader,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import _draft_load_config


class _Field:
    def __init__(self, value: object) -> None:
        self.value = value

    def contents(self) -> object:
        return self.value


class _Type:
    def __init__(self, name: str) -> None:
        self.name = name


class _Tensor:
    def __init__(self, name: str, shape: tuple[int, ...], kind: str) -> None:
        self.name = name
        self.shape = shape
        self.tensor_type = _Type(kind)


class _Reader:
    def __init__(self) -> None:
        self.fields = {name: _Field(value) for name, value in EXPECTED_METADATA.items()}
        self.tensors = [
            _Tensor(name, shape, kind)
            for name, (shape, kind) in dflash_tensor_specs().items()
        ]


def test_published_dflash_contract_is_exactly_81_tensors() -> None:
    reader = _Reader()

    validate_dflash_reader(reader)

    assert len(reader.fields) == 44
    assert len(reader.tensors) == 81
    assert Counter(t.tensor_type.name for t in reader.tensors) == Counter(
        EXPECTED_TYPE_COUNTS
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("target_layer", "target_layers"),
        ("tensor_type", "ffn_gate_exps"),
        ("missing_tensor", "tensor inventory"),
    ],
)
def test_dflash_contract_rejects_incompatible_artifacts(
    mutation: str, match: str
) -> None:
    reader = _Reader()
    if mutation == "target_layer":
        reader.fields["dflash.target_layers"] = _Field((40, 41, 42))
    elif mutation == "tensor_type":
        tensor = next(t for t in reader.tensors if "ffn_gate_exps" in t.name)
        tensor.tensor_type = _Type("Q8_0")
    else:
        reader.tensors.pop()

    with pytest.raises(ValueError, match=match):
        validate_dflash_reader(reader)


def test_dflash_config_converts_target_layers_back_to_zero_based(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_dflash_config_from_gguf.cache_clear()
    monkeypatch.setattr(gguf_dflash, "gguf_reader", lambda _path: _Reader())

    config = build_dflash_config_from_gguf("draft.gguf")

    assert config.architectures == ["DSparkDraftModel"]
    assert config.num_hidden_layers == 43
    assert config.n_mtp_layers == 3
    assert config.dspark_target_layer_ids == [40, 41, 42]
    assert config.dspark_block_size == 5
    assert config.dspark_markov_rank == 256
    assert config.dspark_noise_token_id == 128_799
    assert config.vocab_size == 129_280
    assert config.rope_parameters["rope_type"] == "yarn"
    build_dflash_config_from_gguf.cache_clear()


def test_dflash_adapter_restores_runtime_checkpoint_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_module, "gguf_reader", lambda _path: _Reader())
    adapter = DFlashGGUFAdapter(SimpleNamespace())

    name_map = adapter.build_name_map(SimpleNamespace(model="draft.gguf"))

    assert len(name_map) == 81
    assert name_map["fc.weight"] == "mtp.0.main_proj.weight"
    assert name_map["blk.1.ffn_gate_exps.weight"] == "mtp.1.ffn.experts.0.w1.weight"
    assert name_map["conf_proj.weight"] == "mtp.2.confidence_head.proj.weight"


def test_dflash_adapter_marks_dequantized_output_projection_by_runtime_prefix() -> None:
    adapter = DFlashGGUFAdapter(SimpleNamespace())
    base_spec = GGUFLoadSpec([], [], {})
    model_config = SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=43))

    with mock.patch.object(
        GGUFWeightsAdapter, "prepare_loading", return_value=base_spec
    ):
        spec = adapter.prepare_loading("draft.gguf", model_config)

    for layer in range(43, 46):
        assert f"model.layers.{layer}.attn.wo_a" in spec.unquantized_modules
    assert "model.layers.0.attn.wo_a" not in spec.unquantized_modules


def test_dflash_adapter_restores_confidence_projection_output_dimension() -> None:
    adapter = DFlashGGUFAdapter(SimpleNamespace())
    weight = torch.arange(4352)

    transformed = adapter.transform_weight("mtp.2.confidence_head.proj.weight", weight)

    assert transformed.shape == (1, 4352)
    assert transformed.data_ptr() == weight.data_ptr()


def test_standalone_dflash_uses_gguf_loader_not_target_loader(tmp_path) -> None:
    draft_path = tmp_path / "draft.gguf"
    draft_path.write_bytes(b"GGUF")
    target_load_config = LoadConfig(load_format="auto")
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_load_config=None),
        load_config=target_load_config,
        model_config=SimpleNamespace(model="target-model"),
    )

    draft_load_config = _draft_load_config(
        vllm_config, SimpleNamespace(model=str(draft_path))
    )

    assert draft_load_config.load_format == "gguf"
    assert target_load_config.load_format == "auto"


def test_dspark_confidence_head_concatenates_hidden_and_markov_features() -> None:
    class _Projection(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            assert features.tolist() == [[1.0, 2.0, 3.0]]
            return features.sum(dim=-1, keepdim=True)

    head = DSparkConfidenceHead.__new__(DSparkConfidenceHead)
    nn.Module.__init__(head)
    head.proj = _Projection()

    confidence = head(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0]]))

    torch.testing.assert_close(confidence, torch.tensor([[6.0]]).sigmoid())
