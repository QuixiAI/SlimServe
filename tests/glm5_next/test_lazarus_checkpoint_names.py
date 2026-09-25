# SPDX-License-Identifier: Apache-2.0
"""Exports of GLM-5.3-Flash disagree on tensor paths, not on tensors.

Lazarus-Ai's UnCut NVFP4 keeps the upstream spelling: the KDA forget gate in a
submodule, the mHC tensors as attn_hc/ffn_hc groups, and one pre-fused
convolution where RedHat's and nvidia's ship q/k/v separately. Shapes are
identical either way (the fused conv1d is exactly 3 x projection rows), so the
loader translates names and splits the fused tensor rather than anyone
rewriting a checkpoint.
"""
import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _make_fused_conv1d_weight_loader,
)
from vllm.model_executor.models.glm5_next import Glm5NextForCausalLM


def _translate(name: str) -> str:
    for pref, new in Glm5NextForCausalLM.hf_to_vllm_prefix.items():
        if name.startswith(pref):
            name = new + name[len(pref):]
            break
    for old, new in Glm5NextForCausalLM.hf_to_vllm_substr.items():
        if old in name:
            name = name.replace(old, new)
    return name


@pytest.mark.parametrize(
    "lazarus, canonical",
    [
        ("model.language_model.layers.5.self_attn.forget_gate.f_a_proj.weight",
         "model.layers.5.self_attn.f_a_proj.weight"),
        ("model.language_model.layers.5.self_attn.forget_gate.A_log",
         "model.layers.5.self_attn.A_log"),
        ("model.language_model.layers.5.self_attn.forget_gate.dt_bias",
         "model.layers.5.self_attn.dt_bias"),
        ("model.language_model.layers.5.attn_hc.base", "model.layers.5.hc_attn_base"),
        ("model.language_model.layers.5.attn_hc.fn", "model.layers.5.hc_attn_fn"),
        ("model.language_model.layers.5.ffn_hc.scale", "model.layers.5.hc_ffn_scale"),
    ],
)
def test_upstream_spellings_translate(lazarus, canonical):
    assert _translate(lazarus) == canonical


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.5.self_attn.q_conv1d.weight",
        "model.language_model.layers.5.self_attn.o_proj.weight",
        "model.language_model.layers.5.mlp.experts.3.down_proj.weight",
    ],
)
def test_other_spellings_are_untouched(name):
    assert _translate(name) == name.replace("model.language_model.", "model.")


@pytest.mark.parametrize("tp_size", [1, 4])
def test_a_prefused_convolution_loads_like_three_separate_ones(tp_size):
    projection, width, tp_rank = 8192, 4, 1 if tp_size > 1 else 0
    torch.manual_seed(0)
    q, k, v = (torch.randn(projection, 1, width) for _ in range(3))
    loader = _make_fused_conv1d_weight_loader([projection] * 3, tp_size, tp_rank)

    separate = torch.zeros(3 * projection // tp_size, 1, width)
    for shard_id, piece in enumerate((q, k, v)):
        loader(separate, piece, shard_id)

    # The pre-fused form: one tensor of q|k|v, split by the loader path.
    fused_ckpt = torch.cat([q, k, v], dim=0)
    fused = torch.zeros_like(separate)
    for shard_id, piece in enumerate(fused_ckpt.chunk(3, dim=0)):
        loader(fused, piece, shard_id)

    assert torch.equal(fused, separate)
    # And this rank really holds its own slice, not rank 0's.
    local = projection // tp_size
    assert torch.equal(fused[:local], q[tp_rank * local : (tp_rank + 1) * local])
