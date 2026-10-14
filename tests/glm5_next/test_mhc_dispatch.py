# SPDX-License-Identifier: Apache-2.0
"""Keep explicit normalization outside the measured SM80 fusion path."""

import pytest
import torch
from torch import nn

from vllm.model_executor.models.glm5_next import Glm5NextDecoderLayer


class CountingNorm(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.weight = nn.Parameter(torch.full((4,), float(factor)))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x * self.weight


class IdentityAttention(nn.Module):
    def forward(self, positions, x):
        return x


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("first", [False, True])
def test_norm_applied_once_on_both_paths(monkeypatch, fuse, first):
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer._fuse_mhc_norm = fuse
    layer._overlap_kda = layer._overlap_router = False
    layer.rms_norm_eps = layer.hc_eps = 1e-6
    layer.hc_post_alpha = 2.0
    layer.hc_sinkhorn_iters = 20
    layer.is_linear = False
    layer.input_layernorm = CountingNorm(2)
    layer.post_attention_layernorm = CountingNorm(3)
    layer.self_attn = IdentityAttention()
    layer.mlp = nn.Identity()
    for site in ("attn", "ffn"):
        for name in ("fn", "base", "scale"):
            setattr(layer, f"hc_{site}_{name}", torch.empty(0))
    passed_weights = []
    post, comb = torch.ones(1, 4, 1), torch.eye(4).unsqueeze(0)

    def pre(residual, *args):
        weight = args[-2]
        passed_weights.append(weight)
        x = residual.mean(dim=1)
        return post, comb, x if weight is None else x * weight

    def transition(x, residual, post, comb, *args):
        weight = args[-2]
        passed_weights.append(weight)
        return residual, post, comb, x if weight is None else x * weight

    monkeypatch.setattr(torch.ops.vllm, "glm5_mhc_pre", pre)
    monkeypatch.setattr(torch.ops.vllm, "glm5_mhc_fused_post_pre", transition)
    residual = torch.ones(1, 4, 4)
    x = residual if first else torch.ones(1, 4)
    output, *_ = layer(x, torch.zeros(1), None if first else residual, post, comb)
    torch.testing.assert_close(output, torch.full((1, 4), 6.0))
    assert layer.input_layernorm.calls == layer.post_attention_layernorm.calls == (
        0 if fuse else 1
    )
    assert len(passed_weights) == 2
    assert all((weight is not None) == fuse for weight in passed_weights)
