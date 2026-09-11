# SPDX-License-Identifier: Apache-2.0
"""CPU arithmetic contracts, not native GPU execution or full GEMM parity."""

import torch

from benchmarks.glm5_next_nvfp4_down_candidate import (
    down_epilogue_cpu,
    weight_bits_cpu,
)


def test_rebiased_weights_match_literal_marlin_bits_for_all_valid_codes_scales():
    codes = torch.arange(16, dtype=torch.int32)[:, None]
    # Nonzero S0E5M3 bytes from clipped, nonnegative finite E4M3 scales.
    scales = torch.tensor([0, *range(128, 247)], dtype=torch.int32)[None, :]
    tiny_fp4_bits = ((codes & 8) << 12) | ((codes & 7) << 6)
    packed_byte = scales << 8
    scale_bits = ((packed_byte & 0x8000) >> 1) | ((packed_byte & 0x7F00) >> 4)
    tiny_fp4 = tiny_fp4_bits.to(torch.int16).view(torch.bfloat16).double()
    scale = scale_bits.to(torch.int16).view(torch.bfloat16).double()
    # Float64 oracle preserves the subnormal FP4 operand before multiplication.
    expected = (tiny_fp4 * scale).bfloat16().view(torch.int16)
    actual = weight_bits_cpu(codes, scales)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    absolute = actual.to(torch.int32) & 0x7FFF
    assert ((absolute == 0) | (absolute >= 128)).all()


def test_signed_zero_and_clipped_scale_remain_exact():
    actual = weight_bits_cpu(torch.arange(16), torch.zeros(16)).to(torch.int32) & 0xFFFF
    torch.testing.assert_close(
        actual, (torch.arange(16, dtype=torch.int32) & 8) << 12, atol=0, rtol=0
    )


def test_scale_domain_matches_owned_converter_for_all_finite_nonnegative_fp8():
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_scales,
    )

    values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).half()
    source = torch.cat((values, torch.zeros(1, dtype=torch.float16))).view(2, 64)
    encoded, factor = nvfp4_marlin_process_scales(
        source, scale_factor=1.0, a_dtype=torch.bfloat16
    )
    assert factor == 1.0
    assert set(encoded.view(torch.uint8).flatten().tolist()) == {0, *range(128, 247)}


def test_down_epilogue_cannot_move_scale_before_bf16_rounding():
    tiny = torch.tensor([0.001, -0.001, 0.00001, -0.00001]) * 2.0**-119
    route = torch.full((4,), 0.3)
    global_scale = torch.full((4,), 2.0**119)
    expected = (
        tiny.bfloat16().float() * (route * global_scale).bfloat16().float()
    ).bfloat16()
    torch.testing.assert_close(
        down_epilogue_cpu(tiny, route, global_scale), expected, atol=0, rtol=0
    )
    incorrectly_fused = (tiny * route * global_scale).bfloat16()
    assert not torch.equal(expected, incorrectly_fused)


def test_fp32_factor_is_rounded_to_bf16_before_down_multiply():
    tiny = torch.tensor([0.3, 0.7, 1.01, 1.73]) * 2.0**-119
    route = torch.tensor([0.3031, 0.7017, 1.0141, 0.2873])
    global_scale = torch.full((4,), 2.0**119)
    native = down_epilogue_cpu(tiny, route, global_scale)
    wrong = (tiny.bfloat16().float() * route * global_scale).bfloat16()
    assert not torch.equal(native, wrong)
