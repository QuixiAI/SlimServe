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
        from vllm.platforms import current_platform

        # griddepcontrol is an sm_90+ instruction: ptxas rejects the kernels
        # on sm_80/sm_86 (A100, RTX 3090) when it is emitted there.
        return current_platform.is_cuda() and current_platform.is_arch_support_pdl()
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


# `launch_pdl` is a Triton launch option only the CUDA backend accepts; ROCm's
# raises "Keyword argument launch_pdl was specified but unrecognised" even
# for False. Splat this into the launch instead of passing the keyword.
PDL_LAUNCH: dict[str, bool] = {"launch_pdl": True} if PDL else {}

__all__ = ["PDL", "PDL_LAUNCH", "gdc_launch_dependents", "gdc_wait"]
