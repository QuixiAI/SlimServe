# SPDX-License-Identifier: Apache-2.0
"""Opaque large-M stable alignment, used only by the opt-in GLM53 diagnostic.

Native dispatch retains count256/scatter256 for M17..32 and uses direct1024/
scatter512 for M33..8192. No route scores, IDs or weights are changed.
"""

import torch

from vllm.model_executor.layers.fused_moe.router.glm_route_align import (
    alignment_geometry,
)
from vllm.quixicore.ops import quixicore_ops
from vllm.utils.torch_utils import direct_register_custom_op


def _outputs(ids: torch.Tensor, block: int) -> list[torch.Tensor]:
    if not (
        ids.ndim == 2
        and 17 <= ids.shape[0] <= 8192
        and ids.shape[1] == 8
        and ids.dtype == torch.int32
        and ids.is_contiguous()
        and block in (8, 16, 32, 48, 64)
    ):
        raise ValueError("stable GLM alignment requires contiguous int32 [17..8192,8]")
    capacity, blocks = alignment_geometry(ids.shape[0], 8, 288, block)
    return [ids.new_empty((size,)) for size in (capacity, blocks, 1)]


def _glm_stable_align_impl(ids: torch.Tensor, block: int) -> list[torch.Tensor]:
    outputs = _outputs(ids, block)
    offsets = ids.new_empty((289,))
    quixicore_ops.glm_stable_align(ids, *outputs, offsets, block)
    return outputs


def _glm_stable_align_fake(ids: torch.Tensor, block: int) -> list[torch.Tensor]:
    return _outputs(ids, block)


direct_register_custom_op(
    op_name="glm_stable_align",
    op_func=_glm_stable_align_impl,
    fake_impl=_glm_stable_align_fake,
)


def align(ids: torch.Tensor, block: int) -> list[torch.Tensor]:
    return torch.ops.vllm.glm_stable_align(ids, block)
