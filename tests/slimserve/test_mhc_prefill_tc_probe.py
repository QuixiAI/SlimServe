# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the isolated mHC tensor-core oracle, not GPU parity."""

import pytest
import torch

from benchmarks.kernels.benchmark_glm53_mhc_prefill_tc import (
    check_outputs,
    oracle_rows,
    partial_errors,
)


def fixture():
    # Nonuniform stream/split coordinates, but exactly representable products
    # and sums: catches mixing up stream, split, or output axes in the oracle.
    residual = torch.empty(65, 4, 4096, dtype=torch.bfloat16)
    fn = torch.empty(24, 16384, dtype=torch.bfloat16)
    partial = torch.empty(65, 32, 25)
    for split in range(32):
        square = 0
        for stream in range(4):
            v = (stream + 1) * (1 + split % 3)
            residual[:, stream, split * 128 : (split + 1) * 128] = v
            square += 128 * v * v
        partial[:, split, 24] = square
        for output in range(24):
            dot = 0
            for stream in range(4):
                w = (output + 1) * (stream + 1) * 0.125
                start = stream * 4096 + split * 128
                fn[output, start : start + 128] = w
                dot += 128 * ((stream + 1) * (1 + split % 3)) * w
            partial[:, split, output] = dot
    return residual, fn, partial


def test_oracle_checks_stream_split_and_mix_coordinates():
    result = partial_errors(*fixture())
    assert result["passed"]
    assert result["dot_normalized_rms"] == 0
    assert result["dot_row_peak_error"] == 0
    assert result["square_relative_error"] == 0


@pytest.mark.parametrize("column", [0, 23, 24])
def test_oracle_rejects_bad_dot_or_norm_at_tail(column):
    residual, fn, partial = fixture()
    partial[-1, -1, column] *= 1.1
    assert not partial_errors(residual, fn, partial)["passed"]


@pytest.mark.parametrize("bad_input", [0, 1, 2])
def test_oracle_rejects_nonfinite_inputs_without_nan_metrics(bad_input):
    data = fixture()
    data[bad_input].flatten()[0] = float("nan")
    result = partial_errors(*data)
    assert result["finite"] is False
    assert result["passed"] is False
    assert "dot_normalized_rms" not in result


def test_oracle_checks_zero_case_and_shapes():
    data = fixture()
    for tensor in data:
        tensor.zero_()
    assert partial_errors(*data)["passed"]
    with pytest.raises(ValueError, match="shapes"):
        partial_errors(data[0], data[1], data[2][:, :, :24])


def test_row_selection_covers_tile_boundaries_and_last_row():
    assert oracle_rows(65) == [0, 1, 15, 16, 31, 32, 63, 64]
    assert oracle_rows(7616)[-1] == 7615
    assert oracle_rows(64)[-1] == 63
    with pytest.raises(ValueError, match="at least 64"):
        oracle_rows(63)


def test_output_gate_preserves_residual_bits_and_output_arity():
    reference = [
        torch.zeros(1, dtype=torch.bfloat16),
        torch.ones(1),
        torch.ones(1),
        torch.ones(1, dtype=torch.bfloat16),
    ]
    assert check_outputs(reference, reference)
    candidate = [tensor.clone() for tensor in reference]
    candidate[0].fill_(-0.0)
    with pytest.raises(AssertionError, match="different bits"):
        check_outputs(reference, candidate)
    with pytest.raises(ValueError, match="output count"):
        check_outputs(reference, candidate[:3])
