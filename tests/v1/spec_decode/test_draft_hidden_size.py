# SPDX-License-Identifier: Apache-2.0
"""The width of the V2 speculator's draft hidden buffer.

One rule (``_get_draft_hidden_size``): the draft config's ``hc_mult`` widens
the buffer. HC drafters that consume the per-stream residual (DeepSeek V4 MTP
and DSpark, Qwen4-exp MTP) carry their stream count there; GLM-5.3-Flash's
nextn head consumes the hc-combined state, and every path that builds its
draft config (safetensors checkpoint on CUDA, the ``glm5-next`` GGUF on
Metal) goes through ``SpeculativeConfig.hf_config_override``, which sets its
``hc_mult`` to 1.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.models.qwen4_exp.config import Qwen4ExpConfig, Qwen4ExpTextConfig
from vllm.v1.worker.gpu.spec_decode.speculator import _get_draft_hidden_size

HIDDEN = 4096


def _config(hf_config, hidden_size: int = HIDDEN):
    if isinstance(hf_config, dict):
        hf_config = SimpleNamespace(**hf_config)
    return SimpleNamespace(
        get_hidden_size=lambda: hidden_size,
        get_vocab_size=lambda: 64,
        hf_config=hf_config,
        hf_text_config=hf_config,
    )


def _qwen4_exp_text_config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 0,
        "hc_count": 4,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True},
    }
    values.update(kwargs)
    return Qwen4ExpTextConfig(**values)


def _qwen4_exp_draft_config():
    target = Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=_qwen4_exp_text_config(),
    )
    draft = SpeculativeConfig.compose_draft_hf_overrides(None)(target)
    assert draft.model_type == "qwen4_exp_mtp"
    return draft


def _gguf_text_fields() -> dict:
    layer_types = ["linear_attention", "deepseek_sparse_attention"]
    return {
        "vocab_size": 64,
        "hidden_size": 64,
        "num_hidden_layers": len(layer_types),
        "layer_types": layer_types,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "index_n_heads": 2,
        "index_head_dim": 16,
        "index_topk": 8,
        "index_kpool": 4,
        "hc_mult": 4,
        "num_nextn_predict_layers": 1,
    }


def _glm5_next_gguf_draft_config():
    # Metal glm53f-q2-1: speculator model "target", method glm5_next_mtp. The
    # draft ModelConfig re-reads the same glm5-next GGUF, so its hf_config is
    # what build_glm5_next_config_from_gguf returns (model_type glm5_next,
    # hc_mult=4 on the text config), and get_config then applies the draft's
    # hf_overrides - the composed SpeculativeConfig override.
    from vllm.transformers_utils import gguf_glm5_next

    with patch.object(
        gguf_glm5_next, "text_config_fields_from_gguf", lambda _: _gguf_text_fields()
    ):
        target = gguf_glm5_next.build_glm5_next_config_from_gguf.__wrapped__(
            "glm5-next.gguf"
        )
    assert target.model_type == "glm5_next"
    assert target.text_config.hc_mult == 4
    return SpeculativeConfig.compose_draft_hf_overrides(None)(target)


# --- HC drafters that consume the per-stream residual: widened -------------


def test_qwen4_exp_mtp_uses_expanded_hidden_size() -> None:
    # qwen38fn-fp8-8 / qwen38fn-nvfp4-4: the draft config is built by the
    # real override, which sets hc_mult from the text config's hc_count; the
    # MTP head consumes the hc_count-wide multi-stream feedback.
    draft = _qwen4_exp_draft_config()
    assert draft.hc_mult == 4
    assert _get_draft_hidden_size(_config(draft)) == 4 * HIDDEN


def test_deepseek_v4_mtp_uses_expanded_hidden_size() -> None:
    # DeepSeek V4's MTP consumes the pre-hc_head (T, hc_mult * D) residual.
    # The override renames deepseek_v4 to deepseek_mtp and keeps hc_mult.
    trunk = PretrainedConfig(
        num_nextn_predict_layers=1, hc_mult=4, architectures=["DeepseekV4ForCausalLM"]
    )
    trunk.model_type = "deepseek_v4"
    draft = SpeculativeConfig.hf_config_override(trunk)
    assert draft.model_type == "deepseek_mtp"
    assert draft.architectures == ["DeepSeekV4MTPModel"]
    assert _get_draft_hidden_size(_config(draft.to_dict())) == 4 * HIDDEN


def test_deepseek_v4_dspark_uses_expanded_hidden_size() -> None:
    # DSV4 DSpark reuses the full DeepSeek-V4 config (model_type deepseek_v4).
    config = _config(dict(model_type="deepseek_v4", hc_mult=4, compress_ratios=[4]))
    assert _get_draft_hidden_size(config) == 4 * HIDDEN


# --- GLM-5.3-Flash nextn: the hc-combined state, never widened --------------


def test_glm5_next_mtp_uses_contracted_hidden_size() -> None:
    # CUDA safetensors checkpoint: the trunk's hc_mult (4) stays on the
    # target; the draft override sets the draft's hc_mult to 1.
    trunk = PretrainedConfig(
        num_nextn_predict_layers=1, hc_mult=4, architectures=["Glm5NextForCausalLM"]
    )
    trunk.model_type = "glm5_next_text"
    draft = SpeculativeConfig.hf_config_override(trunk)
    assert draft.model_type == "glm5_next_mtp"
    assert _get_draft_hidden_size(_config(draft.to_dict())) == HIDDEN


def test_glm5_next_gguf_nextn_draft_uses_contracted_hidden_size() -> None:
    draft = _glm5_next_gguf_draft_config()
    assert draft.model_type == "glm5_next_mtp"
    assert draft.architectures == ["Glm5NextMTPModel"]
    assert draft.hc_mult == 1
    assert _get_draft_hidden_size(_config(draft, hidden_size=64)) == 64


def test_non_hc_draft_keeps_hidden_size() -> None:
    assert _get_draft_hidden_size(_config(dict(model_type="llama"))) == HIDDEN


# --- V2 MTPSpeculator construction ------------------------------------------


def _vllm_config(draft_model_config, max_num_tokens: int):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=2,
            draft_model_config=draft_model_config,
            use_local_argmax_reduction=False,
            draft_top_k_top_p=None,
            draft_sample_method="greedy",
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4, max_num_batched_tokens=max_num_tokens
        ),
        model_config=SimpleNamespace(
            max_model_len=128, dtype=torch.float32, use_fp64_gumbel=False
        ),
        parallel_config=SimpleNamespace(
            data_parallel_replicate_moe=False,
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
    )


@pytest.mark.parametrize("case", ["qwen4_exp_mtp", "glm5_next_gguf"])
def test_mtp_speculator_hidden_buffer_takes_target_feedback(case: str) -> None:
    # The target hands the drafter its hidden states and the speculator copies
    # them into its preallocated (max_num_tokens, hidden_size) buffer. With
    # the buffer hc_count times too narrow the Qwen4-exp copy fails.
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    hidden = 16
    if case == "qwen4_exp_mtp":
        draft_hf = _qwen4_exp_draft_config()
        feedback_width = draft_hf.text_config.hc_count * hidden
    else:
        draft_hf = _glm5_next_gguf_draft_config()
        feedback_width = hidden  # the post-final-norm, hc-combined state

    num_tokens = 8
    with patch(
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator."
        "MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        return_value=False,
    ):
        speculator = MTPSpeculator(
            _vllm_config(_config(draft_hf, hidden_size=hidden), num_tokens),
            torch.device("cpu"),
        )

    assert speculator.hidden_states.shape == (num_tokens, feedback_width)
    target_hidden = torch.randn(num_tokens, feedback_width)
    speculator.hidden_states[:num_tokens].copy_(target_hidden)
    assert torch.equal(speculator.hidden_states, target_hidden)
