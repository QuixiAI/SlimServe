# SPDX-License-Identifier: Apache-2.0
"""The public binding rejects an oversized bitmap before launching any kernel."""

import pytest
import torch

pytest.importorskip("vllm._quixicore_C")
from vllm.quixicore.ops import quixicore_ops as qc  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_stable_align_rejects_oversized_ids_before_output_checks():
    ids = torch.zeros(8193, 8, device="cuda", dtype=torch.int32)
    empty = torch.empty(0, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match=r"expects int32 \[17\.\.8192,8\]"):
        qc.glm_stable_align(ids, empty, empty, empty, empty, 32)
