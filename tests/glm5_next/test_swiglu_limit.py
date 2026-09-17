# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash clamps the SwiGLU inputs (checkpoint field ``swiglu_limit``:
gate capped at the limit, up clamped to +-limit) in the dense MLP, the shared
experts and the routed experts. The MLP picks the clamped activation whenever
the limit is set; the clamped op matches the reference formula."""

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.models.deepseek_v2 import silu_and_mul


@pytest.fixture
def default_vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _reference(x: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def test_clamped_activation_matches_reference(default_vllm_config) -> None:
    x = torch.randn(64, 256) * 8.0  # plenty of values beyond the limit
    out = SiluAndMulWithClamp(10.0).forward_native(x)
    torch.testing.assert_close(out, _reference(x, 10.0))
    assert not torch.allclose(out, SiluAndMul().forward_native(x))


def test_mlp_activation_follows_the_config_field(default_vllm_config) -> None:
    assert type(silu_and_mul(None)) is SiluAndMul
    act = silu_and_mul(10.0)
    assert isinstance(act, SiluAndMulWithClamp)
    assert act.swiglu_limit == 10.0 and act.alpha == 1.0 and act.beta == 0.0
