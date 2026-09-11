# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from benchmarks.kernels.benchmark_glm53_mhc_norm import (
    norm_oracle,
    normalize_inplace,
    ulp_distance,
)


def test_bf16_ulp_order_handles_negative_values_and_signed_zero():
    values = torch.tensor([-2, -1, -0.0, 0.0, 1, 2], dtype=torch.bfloat16)
    adjacent = torch.nextafter(values, torch.full_like(values, float("inf")))
    assert ulp_distance(values, adjacent) == 1
    assert ulp_distance(values, values) == 0
    assert ulp_distance(values[2:3], values[3:4]) == 0
    with pytest.raises(ValueError, match="finite"):
        ulp_distance(values, torch.full_like(values, float("inf")))
    with pytest.raises(ValueError, match="matching BF16"):
        ulp_distance(values, values.float())


def test_norm_reference_is_inplace_with_fp32_weight_multiply():
    values = torch.tensor([[1, 2], [-3, 4]], dtype=torch.bfloat16)
    weight = torch.tensor([0.5, 2], dtype=torch.bfloat16)
    expected = norm_oracle(values, weight)
    output = normalize_inplace(values, weight)
    assert output.data_ptr() == values.data_ptr()
    assert ulp_distance(output, expected) <= 1
