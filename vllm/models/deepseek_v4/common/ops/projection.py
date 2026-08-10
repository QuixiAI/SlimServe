# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.quixicore import quixicore_ops
from vllm.utils.torch_utils import direct_register_custom_op


def dsv4_ampere_projection_gemv_impl(
    x: torch.Tensor, weight: torch.Tensor, bf16_output: bool = False
) -> torch.Tensor:
    return quixicore_ops.dsv4_projection_gemv(x, weight, bf16_output)


def dsv4_ampere_projection_gemv_fake(
    x: torch.Tensor, weight: torch.Tensor, bf16_output: bool = False
) -> torch.Tensor:
    dtype = torch.bfloat16 if bf16_output else torch.float32
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=dtype)


direct_register_custom_op(
    op_name="dsv4_ampere_projection_gemv",
    op_func=dsv4_ampere_projection_gemv_impl,
    fake_impl=dsv4_ampere_projection_gemv_fake,
)
