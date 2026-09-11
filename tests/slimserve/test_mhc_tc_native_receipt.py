# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from benchmarks.kernels.check_glm53_mhc_tc_native import qualified_kernel_body


@pytest.mark.parametrize("mutation", [None, "arithmetic", "layout"])
def test_native_kernel_body_is_unchanged_except_namespace(mutation):
    root = Path(__file__).resolve().parents[2]
    probe = (root / "benchmarks/kernels/mhc_prefill_tc_probe.cu").read_text()
    native = (root / "csrc/quixicore/serving/glm53_mhc_prefill_tc.cuh").read_text()
    if mutation == "arithmetic":
        native = native.replace("sum += v.x * v.x;", "sum += v.x;")
    if mutation == "layout":
        native = native.replace("ROW = 520", "ROW = 512")
    if mutation is None:
        qualified_kernel_body(probe, native)
    else:
        with pytest.raises(ValueError, match="kernel body changed"):
            qualified_kernel_body(probe, native)
