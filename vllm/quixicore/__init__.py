# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuixiCore-CUDA kernel access for the NVIDIA/Ampere serving path.

The compiled extension (`vllm._quixicore_C`, built from the vendored sources
under `csrc/quixicore/`) plays the role AITER plays on ROCm: the sparse-MLA
decode kernel and its supporting ops, called directly as Python functions
rather than through `torch.ops`.
"""

from vllm.quixicore.ops import quixicore_ops

__all__ = ["quixicore_ops"]
