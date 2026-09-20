# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Programmatic dependent launch for the Triton decode kernels: with QC_PDL on
(the default; "0" is the A/B baseline) they are launched with the
programmatic-serialization attribute and start with a trigger for their own
successor plus a wait for their predecessor. Both are no-ops without the
attribute, and are kept out of the kernel entirely on non-CUDA targets and for
the baseline through each kernel's PDL constexpr."""

import os

import torch

from vllm.triton_utils import triton


def _pdl_enabled() -> bool:
    if os.environ.get("QC_PDL", "4") == "0":
        return False
    try:
        return torch.cuda.is_available() and torch.version.hip is None
    except Exception:
        return False


PDL = _pdl_enabled()
if PDL:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
else:

    @triton.jit
    def gdc_launch_dependents():
        pass

    @triton.jit
    def gdc_wait():
        pass


__all__ = ["PDL", "gdc_launch_dependents", "gdc_wait"]
