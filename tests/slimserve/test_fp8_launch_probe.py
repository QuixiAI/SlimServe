# SPDX-License-Identifier: Apache-2.0
import argparse

import pytest
import torch

from benchmarks.kernels.benchmark_glm53_fp8_launch import (
    CONFIGS,
    oracle_errors,
    parse_config,
)


def test_fixed_configs_include_every_current_shared_expert_dispatch():
    assert len(CONFIGS) == len(set(CONFIGS)) == 12
    assert {(8, 8, 8), (16, 8, 8), (32, 8, 4)} <= set(CONFIGS)
    for config in CONFIGS:
        assert parse_config(",".join(map(str, config))) == config
        assert config[0] in (8, 16, 32)
        assert config[1] in (4, 8)
        assert config[2] in (4, 8)


@pytest.mark.parametrize("value", ["auto", "16,8", "16,8,3", "16,2,4", "8,8,8,8"])
def test_unprescribed_config_is_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_config(value)


def test_independent_oracle_records_and_rejects_large_or_nonfinite_errors():
    expected = torch.tensor([[1.0, -2.0], [0.0, 0.0]], dtype=torch.float64)
    good = oracle_errors(expected.bfloat16(), expected)
    assert good == {
        "finite": True,
        "normalized_rms": 0.0,
        "row_peak_error": 0.0,
        "passed": True,
    }
    bad = expected.clone()
    bad[0, 0] += 0.1
    assert not oracle_errors(bad, expected)["passed"]
    bad[0, 0] = float("nan")
    assert not oracle_errors(bad, expected)["passed"]
    bad = expected.clone()
    bad[1, 0] = 1e-8
    assert not oracle_errors(bad, expected)["passed"]


def test_oracle_rejects_broadcastable_output_shape():
    with pytest.raises(ValueError, match="shapes differ"):
        oracle_errors(torch.ones(1, 2), torch.ones(3, 2))
