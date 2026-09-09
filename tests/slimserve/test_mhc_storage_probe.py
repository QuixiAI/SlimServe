# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from benchmarks.kernels.benchmark_glm53_mhc_storage import exact, lossless_bf16


def test_storage_conversion_requires_exact_checkpoint_values():
    value = torch.tensor([0.0, -1.0, 0.125, 1e-20], dtype=torch.bfloat16).float()
    assert torch.equal(lossless_bf16(value).float(), value)
    with pytest.raises(ValueError, match="alter checkpoint"):
        lossless_bf16(torch.tensor([0.1], dtype=torch.float32))


def test_storage_probe_rejects_nan_and_any_output_difference():
    with pytest.raises(ValueError):
        lossless_bf16(torch.tensor([float("nan")]))
    value = torch.ones(2, dtype=torch.bfloat16)
    exact([value], [value.clone()])
    with pytest.raises(AssertionError, match="bit-exact"):
        exact([value], [value + 0.01])
    with pytest.raises(AssertionError, match="bit-exact"):
        exact([value], [value.float()])
