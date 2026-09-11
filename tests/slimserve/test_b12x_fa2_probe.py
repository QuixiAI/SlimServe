# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[2] / "benchmarks/kernels/probe_b12x_fa2.py"
    spec = importlib.util.spec_from_file_location("b12x_fa2_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_oracle_keeps_variable_sequences_separate(probe):
    qkv = torch.zeros(3, 3, 16, 64, dtype=torch.bfloat16)
    qkv[0, 2].fill_(2)
    qkv[1, 2].fill_(4)
    qkv[2, 2].fill_(8)
    expected = torch.empty(3, 16, 64, dtype=torch.float64)
    expected[0].fill_(2)
    expected[1:].fill_(6)
    assert torch.equal(probe.oracle(qkv, (1, 2)), expected)


def test_gates_reject_wrong_nonfinite_and_wrong_shape(probe):
    expected = torch.ones(2, 16, 64)
    assert probe.errors(expected, expected)["passed"]
    assert not probe.errors(expected + 0.1, expected)["passed"]
    assert not probe.errors(expected * float("nan"), expected)["passed"]
    with pytest.raises(ValueError, match="shape mismatch"):
        probe.errors(expected[0], expected)


def test_input_digest_is_value_and_order_sensitive(probe):
    value = torch.arange(24, dtype=torch.float32).bfloat16().reshape(2, 3, 4)
    assert probe.tensor_digest(value) == probe.tensor_digest(value.clone())
    assert probe.tensor_digest(value) != probe.tensor_digest(value.flip(0))
    assert probe.tensor_digest(value) != probe.tensor_digest(value + 1)
