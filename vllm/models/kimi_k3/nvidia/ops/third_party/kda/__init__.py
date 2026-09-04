# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA re-export of the vendored KDA Triton kernels.

The kernels under ``..amd.ops.third_party.kda`` are pure Triton and
platform-aware internally (autotune warp counts switch on AMD); the
``amd``/``nvidia`` split is organizational. glm5_next's KDA layers import
this path on CUDA via the shared GDN layer.
"""

from vllm.models.kimi_k3.amd.ops.third_party.kda import (
    chunk_kda_with_fused_gate,
    fused_recurrent_kda,
    fused_recurrent_kda_packed_decode,
)

__all__ = [
    "chunk_kda_with_fused_gate",
    "fused_recurrent_kda",
    "fused_recurrent_kda_packed_decode",
]
