# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.speculator import _get_draft_hidden_size


def _config(**hf_fields):
    return SimpleNamespace(
        get_hidden_size=lambda: 4096,
        hf_config=SimpleNamespace(**hf_fields),
    )


def test_glm5_next_mtp_uses_contracted_hidden_size() -> None:
    # GLM-5.3-Flash's target returns the contracted mHC state to its MTP
    # head. The trunk's hc_mult (4) must not widen the proposer buffer: the
    # draft config override sets the draft's hc_mult to 1.
    trunk = PretrainedConfig(
        num_nextn_predict_layers=1, hc_mult=4, architectures=["Glm5NextForCausalLM"]
    )
    trunk.model_type = "glm5_next_text"
    draft = SpeculativeConfig.hf_config_override(trunk)
    assert draft.model_type == "glm5_next_mtp"
    assert _get_draft_hidden_size(_config(**draft.to_dict())) == 4096


def test_deepseek_v4_mtp_uses_expanded_hidden_size() -> None:
    # DeepSeek V4's MTP consumes the pre-hc_head residual stream.
    config = _config(model_type="deepseek_mtp", hc_mult=4, compress_ratios=[4])
    assert _get_draft_hidden_size(config) == 16384


def test_deepseek_v4_dspark_uses_expanded_hidden_size() -> None:
    # DSV4 DSpark reuses the full DeepSeek-V4 config (model_type deepseek_v4).
    config = _config(model_type="deepseek_v4", hc_mult=4, compress_ratios=[4])
    assert _get_draft_hidden_size(config) == 16384


def test_qwen4_exp_mtp_uses_expanded_hidden_size() -> None:
    # SpeculativeConfig sets hc_mult from hc_count without compress_ratios;
    # Qwen MTP consumes the hc_mult-wide multi-stream feedback.
    config = _config(model_type="qwen4_exp_mtp", hc_mult=4)
    assert _get_draft_hidden_size(config) == 16384


def test_non_hc_draft_keeps_hidden_size() -> None:
    assert _get_draft_hidden_size(_config(model_type="llama")) == 4096
