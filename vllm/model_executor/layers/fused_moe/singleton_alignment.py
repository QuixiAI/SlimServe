# SPDX-License-Identifier: Apache-2.0
"""Read-only Marlin row metadata for one token with distinct routed experts.

Experts need not appear in numeric order: the GEMM writes each result back
to its original top-k position. EP/invalid-expert and multi-token routing
must use the general aligner. Metadata belongs to an expert-module instance,
is allocated before graph capture, and never aliases another model's state.
"""

import torch


class SingletonAlignment:
    def __init__(self, topk: int, block_size: int, device):
        if min(topk, block_size) <= 0:
            raise ValueError("topk and block size must be positive")
        self.topk = topk
        self.block_size = block_size
        rows = torch.full((topk, block_size), topk, dtype=torch.int32)
        rows[:, 0] = torch.arange(topk, dtype=torch.int32)
        self.rows = rows.flatten().to(device)
        self.padded = torch.tensor(
            [topk * block_size], dtype=torch.int32, device=device
        )

    def compatible(self, topk_ids: torch.Tensor, block_size: int) -> bool:
        return (
            topk_ids.shape == (1, self.topk)
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
            and topk_ids.device == self.rows.device
            and block_size == self.block_size
        )

    def get(self, topk_ids: torch.Tensor):
        if not self.compatible(topk_ids, self.block_size):
            raise ValueError(
                "expected one contiguous int32 top-k row on the owner device"
            )
        return self.rows, topk_ids.view(-1), self.padded


def make_singleton_alignment(moe, quant, extra, input_dtype, sm80):
    enabled = (
        extra.get("glm5_next_singleton_marlin_alignment", False)
        if isinstance(extra, dict)
        else False
    )
    if not isinstance(enabled, bool):
        raise ValueError("glm5_next_singleton_marlin_alignment must be a boolean")
    if not (
        enabled
        and sm80
        and quant.use_nvfp4_w4a16
        and input_dtype is None
        and moe.hidden_dim == 4096
        and moe.num_experts == 288
        and not moe.moe_parallel_config.use_ep
        and not moe.is_lora_enabled
    ):
        return None
    return SingletonAlignment(moe.experts_per_token, 8, moe.device)
