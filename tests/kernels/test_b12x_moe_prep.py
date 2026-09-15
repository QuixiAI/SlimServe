# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The b12x NVFP4 MoE weight preparation and backend registration."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    map_nvfp4_backend,
)
from vllm.model_executor.layers.quantization.utils.b12x_moe import (
    prepare_nvfp4_moe_layer_for_b12x,
)

# The block-scale swizzle queries the device, so these run on a CUDA device.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _layer(num_experts: int, hidden: int, intermediate: int):
    fp8 = torch.float8_e4m3fn
    dev = "cuda"
    w13 = torch.randint(
        0,
        255,
        (num_experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device=dev,
    )
    w13_scale = torch.rand(num_experts, 2 * intermediate, hidden // 16, device=dev).to(
        fp8
    )
    w2 = torch.randint(
        0, 255, (num_experts, hidden, intermediate // 2), dtype=torch.uint8, device=dev
    )
    w2_scale = torch.rand(num_experts, hidden, intermediate // 16, device=dev).to(fp8)
    w13_scale_2 = torch.rand(num_experts, device=dev)
    w2_scale_2 = torch.rand(num_experts, device=dev)
    a13_scale = torch.rand(num_experts, 2, device=dev)
    a2_scale = torch.rand(num_experts, device=dev)
    return w13, w13_scale, w13_scale_2, a13_scale, w2, w2_scale, w2_scale_2, a2_scale


def test_backend_name_maps_to_b12x():
    assert map_nvfp4_backend("b12x") is NvFp4MoeBackend.B12X


def test_glm53_geometry_needs_no_padding():
    # GLM-5.3-Flash at TP4: 4096 hidden, 512 intermediate per rank.
    w13, w13_s, w13_s2, a13, w2, w2_s, w2_s2, a2 = _layer(4, 4096, 512)
    out = prepare_nvfp4_moe_layer_for_b12x(
        w13, w13_s, w13_s2, a13, w2, w2_s, w2_s2, a2, is_act_and_mul=True
    )
    p_w13, p_w13_s, p_w13_s2, p_a13, p_w2, p_w2_s, p_w2_s2, p_a2 = out
    assert p_w13.data_ptr() == w13.data_ptr() and p_w2.data_ptr() == w2.data_ptr()
    assert p_w13_s.shape == w13_s.shape and p_w2_s.shape == w2_s.shape
    assert p_w13_s2 is w13_s2 and p_w2_s2 is w2_s2
    # The two-column activation scale collapses to one value per expert.
    assert p_a13.shape == (4,) and p_a13.dtype == torch.float32
    assert torch.equal(p_a13, a13.amax(dim=1))
    assert p_a2.shape == (4,) and p_a2.dtype == torch.float32


def test_gated_rows_pad_to_the_tile_in_each_half():
    w13, w13_s, w13_s2, a13, w2, w2_s, w2_s2, a2 = _layer(2, 256, 96)
    out = prepare_nvfp4_moe_layer_for_b12x(
        w13, w13_s, w13_s2, a13, w2, w2_s, w2_s2, a2, is_act_and_mul=True
    )
    p_w13, p_w13_s, _, _, p_w2, p_w2_s, _, _ = out
    # 96 -> 128 rows per half: gate rows first, then zero rows, then up rows.
    assert p_w13.shape == (2, 256, 128)
    assert torch.equal(p_w13[:, :96], w13[:, :96])
    assert torch.equal(p_w13[:, 128:224], w13[:, 96:])
    assert not p_w13[:, 96:128].any() and not p_w13[:, 224:].any()
    assert p_w13_s.shape == (2, 256, 16)
    assert p_w2.shape == (2, 256, 64) and p_w2_s.shape == (2, 256, 8)
    assert torch.equal(p_w2[:, :, :48], w2)
    assert not p_w2[:, :, 48:].any()


def test_mismatched_shapes_are_rejected():
    w13, w13_s, w13_s2, a13, w2, w2_s, w2_s2, a2 = _layer(2, 256, 96)
    with pytest.raises(ValueError, match="w2 shape"):
        prepare_nvfp4_moe_layer_for_b12x(
            w13, w13_s, w13_s2, a13, w2[:, :, :40], w2_s, w2_s2, a2, is_act_and_mul=True
        )
    with pytest.raises(ValueError, match="one value per expert"):
        prepare_nvfp4_moe_layer_for_b12x(
            w13, w13_s, w13_s2, a13[:1], w2, w2_s, w2_s2, a2, is_act_and_mul=True
        )
