# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.kernels.repro_marlin_shared_boundary import isolated_source


def test_only_requested_boundary_changes():
    original = """  auto thread_block_reduce = [&]() {
    compute();
  };
  auto write_result = [&](bool last) {
    write();
  };"""
    assert isolated_source(original, "original") == original
    assert isolated_source(original, "compute").count("__syncthreads();") == 1
    assert isolated_source(original, "output").count("__syncthreads();") == 1
    assert isolated_source(original, "both").count("__syncthreads();") == 2
    with pytest.raises(ValueError, match="unknown boundary"):
        isolated_source(original, "typo")
    with pytest.raises(ValueError, match="no longer matches"):
        isolated_source(original + original, "both")
