from types import SimpleNamespace

from vllm.v1.worker.gpu.spec_decode.speculator import _get_draft_hidden_size


def _config(**hf_fields):
    return SimpleNamespace(
        get_hidden_size=lambda: 4096,
        hf_config=SimpleNamespace(**hf_fields),
    )


def test_glm5_next_mtp_uses_contracted_hidden_size() -> None:
    # GLM-5.3-Flash's target returns the mean-contracted mHC state to its MTP
    # head. hc_mult describes the trunk and must not widen the proposer buffer.
    assert _get_draft_hidden_size(_config(model_type="glm5_next_mtp", hc_mult=4)) == 4096


def test_deepseek_v4_mtp_uses_expanded_hidden_size() -> None:
    # DeepSeek V4's MTP consumes the pre-hc_head residual stream.
    config = _config(model_type="deepseek_v4_mtp", hc_mult=4, compress_ratios=[4])
    assert _get_draft_hidden_size(config) == 16384
