# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up v1 block-table Triton kernels."""

import torch

_SLOT_MAPPING_WARMUP_TOKENS = 8
_SLOT_MAPPING_WARMUP_BLOCK_SIZES = (3, 16)
_SLOT_MAPPING_WARMUP_CP_KV_CACHE_INTERLEAVE_SIZE = 1


def warm_v1_block_table_kernels(
    device: torch.device,
    max_tokens: int,
) -> None:
    from vllm.utils.bootstamp import bootstamp
    from vllm.v1.worker.block_table import BlockTable

    bootstamp("block_table warmup: BlockTable imported")

    num_tokens = max(0, min(_SLOT_MAPPING_WARMUP_TOKENS, max_tokens))
    if num_tokens <= 0:
        return

    bootstamp(f"block_table warmup: device={device!r}")
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32).to(device)
    bootstamp("block_table warmup: query_start_loc via .to(device)")
    positions = torch.arange(num_tokens, dtype=torch.int64).to(device)
    bootstamp("block_table warmup: positions via .to(device)")
    for block_size in _SLOT_MAPPING_WARMUP_BLOCK_SIZES:
        max_num_blocks_per_req = max(
            1, (max(num_tokens, max_tokens) + block_size - 1) // block_size
        )
        max_num_blocks_per_req = ((max_num_blocks_per_req + 15) // 16) * 16
        block_table = BlockTable(
            block_size=block_size,
            max_num_reqs=1,
            max_num_blocks_per_req=max_num_blocks_per_req,
            max_num_batched_tokens=max(num_tokens, max_tokens),
            pin_memory=False,
            device=device,
            kernel_block_size=block_size,
            cp_kv_cache_interleave_size=(
                _SLOT_MAPPING_WARMUP_CP_KV_CACHE_INTERLEAVE_SIZE
            ),
        )
        block_table.add_row(list(range(max_num_blocks_per_req)), 0)
        block_table.commit_block_table(1)
        block_table.compute_slot_mapping(1, query_start_loc, positions)
        bootstamp(f"block_table warmup: block_size={block_size} done")
